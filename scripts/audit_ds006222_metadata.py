#!/usr/bin/env python3
"""Metadata and event audit for OpenNeuro ds006222 without signal download."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd


BIOSEMI_BASES = (61440, 49152)


def normalize_event_value(value: object) -> int | None:
    text = str(value).strip()
    match = re.fullmatch(r"condition\s+(\d+)", text, flags=re.I)
    if match:
        return int(match.group(1))
    try:
        number = int(float(text))
    except ValueError:
        return None
    for base in BIOSEMI_BASES:
        if base <= number <= base + 255:
            return number - base
    return number


def annex_declared_size(path: Path) -> int:
    if path.is_symlink():
        target = os.readlink(path)
        match = re.search(r"SHA256E-s(\d+)--", target)
        if match:
            return int(match.group(1))
    return path.stat().st_size


def audit(root: Path, output: Path) -> dict:
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(root / "participants.tsv", sep="\t", dtype=str)
    event_files = sorted(root.glob("sub-*/ses-*/eeg/*_events.tsv"))
    recording_rows = []
    event_counter: Counter[int | str] = Counter()

    for event_path in event_files:
        participant_id = event_path.parts[-4]
        session_id = event_path.parts[-3]
        stem = event_path.name.replace("_events.tsv", "")
        eeg_json = event_path.parent / f"{stem}_eeg.json"
        set_path = event_path.parent / f"{stem}_eeg.set"
        fdt_path = event_path.parent / f"{stem}_eeg.fdt"
        sidecar = json.loads(eeg_json.read_text(encoding="utf-8"))
        events = pd.read_csv(event_path, sep="\t", dtype=str)
        normalized = events["value"].map(normalize_event_value)
        event_counter.update("unparsed" if pd.isna(value) else int(value) for value in normalized)
        recording_rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "event_rows": len(events),
                "event_codes_parsed": int(normalized.notna().sum()),
                "recording_duration_s": float(sidecar["RecordingDuration"]),
                "sampling_frequency_hz": float(sidecar["SamplingFrequency"]),
                "eeg_channel_count": int(sidecar["EEGChannelCount"]),
                "misc_channel_count": int(sidecar.get("MiscChannelCount", 0)),
                "set_declared_bytes": annex_declared_size(set_path),
                "fdt_declared_bytes": annex_declared_size(fdt_path),
            }
        )

    records = pd.DataFrame(recording_rows).merge(participants, on="participant_id", how="left")
    records.to_csv(output / "recording_inventory.csv", index=False)
    event_counts = pd.DataFrame(
        [{"normalized_event_code": key, "count": count} for key, count in event_counter.items()]
    ).sort_values("count", ascending=False)
    event_counts.to_csv(output / "normalized_event_counts.csv", index=False)

    group_counts = participants["Group"].value_counts(dropna=False).to_dict()
    duration = records["recording_duration_s"]
    summary = {
        "dataset_id": "ds006222",
        "dataset_version": "1.0.1",
        "dataset_doi": "10.18112/openneuro.ds006222.v1.0.1",
        "license": "CC0",
        "participants_rows": int(len(participants)),
        "unique_participants": int(participants["participant_id"].nunique()),
        "recordings": int(len(records)),
        "participants_with_multiple_sessions": int(
            records.groupby("participant_id")["session_id"].nunique().gt(1).sum()
        ),
        "group_counts_as_released": {str(key): int(value) for key, value in group_counts.items()},
        "age_missing_or_na": int(
            (participants["Age"].isna() | participants["Age"].isin(["n/a", "", "NA"])).sum()
        ),
        "sampling_frequencies_hz": sorted(records["sampling_frequency_hz"].unique().tolist()),
        "eeg_channel_counts": sorted(records["eeg_channel_count"].unique().tolist()),
        "recording_duration_s": {
            "min": float(duration.min()),
            "median": float(duration.median()),
            "max": float(duration.max()),
        },
        "event_rows_total": int(records["event_rows"].sum()),
        "event_values_unparsed": int(event_counter.get("unparsed", 0)),
        "declared_signal_size_gib": float(
            (records["set_declared_bytes"].sum() + records["fdt_declared_bytes"].sum()) / 2**30
        ),
        "metadata_gate": "pass_with_issues",
        "issues": [
            "Released participants table has 69 rows whereas the final paper analysed 62 participants.",
            "One released participant has group label 'cut?'.",
            "One participant has two sessions; records are not independent participants.",
            "Recording durations vary substantially and require an explicit minimum-exposure rule.",
            "BIDS sidecars report 512 Hz although the paper reports acquisition at 2048 Hz; the release appears downsampled.",
            "Raw trigger values use multiple BioSemi status-bit offsets and require normalization.",
            "The final paper conditions further analyses on a 40-Hz-response threshold and excludes behaviour below 80%; neither rule should define the new primary analysis.",
        ],
    }
    (output / "metadata_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
