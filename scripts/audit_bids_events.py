#!/usr/bin/env python3
"""Audit event metadata for the two core public EEG datasets.

The script reads only BIDS sidecars (TSV/JSON), never the EEG signal files.
It is intentionally standard-library-only so that metadata QA can run before
the scientific Python environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


PHOTO_RE = re.compile(r"^PHOTO\s+(\d+(?:\.\d+)?)Hz$", re.IGNORECASE)
SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)")


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "n/a", "-"):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def subject_id(path: Path) -> str:
    match = SUBJECT_RE.search(path.as_posix())
    if not match:
        raise ValueError(f"Cannot infer subject ID from {path}")
    return match.group(1)


def participant_ids(root: Path) -> Set[str]:
    path = root / "participants.tsv"
    if not path.exists():
        return set()
    return {row["participant_id"] for row in read_tsv(path)}


def participants_by_id(root: Path) -> Dict[str, Dict[str, str]]:
    path = root / "participants.tsv"
    if not path.exists():
        return {}
    return {row["participant_id"]: row for row in read_tsv(path)}


def summarize_numbers(
    values: Iterable[float],
) -> Dict[str, Union[float, int, None]]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(finite),
        "min": round(finite[0], 6),
        "median": round(statistics.median(finite), 6),
        "max": round(finite[-1], 6),
    }


def audit_auditory(root: Path) -> Dict[str, Any]:
    event_files = sorted(root.glob("sub-*/eeg/*_events.tsv"))
    eeg_json_by_subject = {
        subject_id(path): path for path in root.glob("sub-*/eeg/*_eeg.json")
    }
    records: List[Dict[str, Any]] = []

    for path in event_files:
        sid = subject_id(path)
        rows = read_tsv(path)
        parsed = []
        for row in rows:
            onset = as_float(row.get("onset"))
            duration = as_float(row.get("duration"))
            value = (row.get("value") or "").strip()
            trial_type = (row.get("trial_type") or "").strip()
            if onset is None or duration is None:
                continue
            parsed.append((onset, duration, value, trial_type))

        stim = [row for row in parsed if row[2] == "2" or row[3] == "Stimulus"]
        rest = [row for row in parsed if row[2] == "1" or row[3] == "Rest"]
        labels = [row[3] for row in parsed]
        alternating = all(a != b for a, b in zip(labels, labels[1:]))
        duration = None
        if sid in eeg_json_by_subject:
            duration = as_float(
                str(read_json(eeg_json_by_subject[sid]).get("RecordingDuration", ""))
            )
        event_end = max((onset + length for onset, length, _, _ in parsed), default=0.0)
        records.append(
            {
                "subject": sid,
                "n_rows": len(parsed),
                "n_stimulus": len(stim),
                "n_rest": len(rest),
                "alternating": alternating,
                "stimulus_durations": sorted({row[1] for row in stim}),
                "rest_durations": sorted({row[1] for row in rest}),
                "first_onset_s": min((row[0] for row in parsed), default=None),
                "event_end_s": event_end,
                "recording_duration_s": duration,
                "event_within_recording": duration is None or event_end <= duration + 1e-6,
            }
        )

    ids = {record["subject"] for record in records}
    participant_table = participants_by_id(root)
    participants = set(participant_table)
    return {
        "dataset_kind": "auditory_40hz",
        "root": str(root.resolve()),
        "n_event_files": len(event_files),
        "n_participants_tsv": len(participants),
        "participants_without_events": sorted(participants - ids),
        "events_without_participant_row": sorted(ids - participants),
        "stimulus_block_count": dict(
            sorted(Counter(record["n_stimulus"] for record in records).items())
        ),
        "recording_duration_s": summarize_numbers(
            record["recording_duration_s"]
            for record in records
            if record["recording_duration_s"] is not None
        ),
        "all_blocks_alternate": all(record["alternating"] for record in records),
        "all_events_within_recording": all(
            record["event_within_recording"] for record in records
        ),
        "subjects": records,
    }


def contiguous_pulse_segment(
    pulse_onsets: List[float], label_onset: float, frequency_hz: float
) -> List[float]:
    """Return the pulse train surrounding a PHOTO label.

    A gap longer than five expected cycles (and at least 0.5 s) terminates the
    train. The rule tolerates a missing pulse but not the approximately 10 s
    inter-frequency pause in this dataset.
    """
    after_label = [onset for onset in pulse_onsets if onset >= label_onset - 0.05]
    if not after_label:
        return []
    gap_limit = max(0.5, 5.0 / frequency_hz)
    segment = [after_label[0]]
    for onset in after_label[1:]:
        if onset - segment[-1] > gap_limit:
            break
        segment.append(onset)
    return segment


def first_event_between(
    rows: List[Tuple[float, str]], label: str, start: float, stop: float
) -> Optional[float]:
    return next(
        (onset for onset, value in rows if start <= onset < stop and value == label),
        None,
    )


def audit_visual(root: Path) -> Dict[str, Any]:
    event_files = sorted(root.glob("sub-*/eeg/*_events.tsv"))
    records: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []

    for path in event_files:
        sid = subject_id(path)
        rows_raw = read_tsv(path)
        rows = []
        for row in rows_raw:
            onset = as_float(row.get("onset"))
            if onset is not None:
                rows.append((onset, (row.get("value") or "").strip()))
        labels = []
        for index, (onset, value) in enumerate(rows):
            match = PHOTO_RE.match(value)
            if match:
                labels.append((index, onset, float(match.group(1))))
        subject_blocks: List[Dict[str, Any]] = []
        all_pulses = [onset for onset, value in rows if value == "Photo/HV mark"]

        for label_index, (_, label_onset, frequency) in enumerate(labels):
            next_label = (
                labels[label_index + 1][1]
                if label_index + 1 < len(labels)
                else float("inf")
            )
            pulse_segment = contiguous_pulse_segment(
                all_pulses, label_onset=label_onset, frequency_hz=frequency
            )
            # Eye-state annotations must belong to the current hardware pulse
            # train. Without this bound, a malformed final block can be paired
            # to an unrelated eye event minutes later.
            pulse_stop = (
                pulse_segment[-1] + max(0.1, 1.5 / frequency)
                if pulse_segment
                else next_label
            )
            block_stop = min(next_label, pulse_stop)
            open_onset = first_event_between(rows, "open eyes", label_onset, block_stop)
            closed_onset = (
                first_event_between(rows, "closed eyes", open_onset, block_stop)
                if open_onset is not None
                else None
            )
            intervals = [b - a for a, b in zip(pulse_segment, pulse_segment[1:])]
            estimated_frequency = (
                1.0 / statistics.median(intervals) if intervals else None
            )
            block = {
                "subject": sid,
                "frequency_hz": frequency,
                "label_onset_s": label_onset,
                "open_onset_s": open_onset,
                "closed_onset_s": closed_onset,
                "open_duration_s": (
                    closed_onset - open_onset
                    if open_onset is not None and closed_onset is not None
                    else None
                ),
                "n_hardware_pulses": len(pulse_segment),
                "pulse_train_duration_s": (
                    pulse_segment[-1] - pulse_segment[0]
                    if len(pulse_segment) >= 2
                    else None
                ),
                "estimated_pulse_frequency_hz": estimated_frequency,
                "frequency_error_pct": (
                    100.0 * abs(estimated_frequency - frequency) / frequency
                    if estimated_frequency is not None
                    else None
                ),
                "open_after_stimulus_label_s": (
                    open_onset - label_onset if open_onset is not None else None
                ),
            }
            subject_blocks.append(block)
            blocks.append(block)

        frequencies = sorted({block["frequency_hz"] for block in subject_blocks})
        records.append(
            {
                "subject": sid,
                "n_rows": len(rows),
                "frequencies_hz": frequencies,
                "n_frequency_blocks": len(subject_blocks),
                "missing_open_close_pairs": sum(
                    block["open_onset_s"] is None or block["closed_onset_s"] is None
                    for block in subject_blocks
                ),
            }
        )

    ids = {record["subject"] for record in records}
    participant_table = participants_by_id(root)
    participants = set(participant_table)
    frequency_coverage = Counter(
        frequency
        for record in records
        for frequency in set(record["frequencies_hz"])
    )
    usable_blocks = [
        block
        for block in blocks
        if block["open_duration_s"] is not None and block["open_duration_s"] > 0
    ]
    frequency_error = [
        block["frequency_error_pct"]
        for block in blocks
        if block["frequency_error_pct"] is not None
    ]
    usable_by_frequency = {
        frequency: {
            block["subject"]
            for block in usable_blocks
            if block["frequency_hz"] == frequency
        }
        for frequency in sorted(frequency_coverage)
    }
    common_frequencies = {5.0, 10.0, 15.0, 20.0}
    complete_common_subjects = {
        sid
        for sid in ids
        if all(sid in usable_by_frequency.get(frequency, set()) for frequency in common_frequencies)
    }
    group_totals = Counter(
        (participant_table[sid].get("Group") or "missing").strip()
        for sid in participants
    )
    complete_common_by_group = Counter(
        (participant_table[sid].get("Group") or "missing").strip()
        for sid in complete_common_subjects
    )
    return {
        "dataset_kind": "visual_photostimulation",
        "root": str(root.resolve()),
        "n_event_files": len(event_files),
        "n_participants_tsv": len(participants),
        "participants_without_events": sorted(participants - ids),
        "events_without_participant_row": sorted(ids - participants),
        "frequency_coverage_n_subjects": {
            str(key): value for key, value in sorted(frequency_coverage.items())
        },
        "usable_open_eye_coverage_n_subjects": {
            str(key): len(value) for key, value in sorted(usable_by_frequency.items())
        },
        "n_subjects_with_usable_5_10_15_20_hz": len(complete_common_subjects),
        "complete_common_frequencies_by_group": {
            group: {
                "complete": complete_common_by_group.get(group, 0),
                "total": total,
            }
            for group, total in sorted(group_totals.items())
        },
        "n_frequency_blocks": len(blocks),
        "n_usable_open_eye_blocks": len(usable_blocks),
        "n_blocks_missing_open_close_pair": len(blocks) - len(usable_blocks),
        "open_eye_duration_s": summarize_numbers(
            block["open_duration_s"] for block in usable_blocks
        ),
        "open_after_stimulus_label_s": summarize_numbers(
            block["open_after_stimulus_label_s"]
            for block in usable_blocks
            if block["open_after_stimulus_label_s"] is not None
        ),
        "pulse_frequency_error_pct": summarize_numbers(frequency_error),
        "subjects": records,
        "blocks": blocks,
    }


def infer_kind(root: Path) -> str:
    names = {path.name for path in root.glob("task-*_events.json")}
    if "task-photomark_events.json" in names:
        return "visual"
    if "task-40HzAuditoryEntrainment_events.json" in names:
        return "auditory"
    raise ValueError(
        "Cannot infer dataset kind. Pass --kind auditory or --kind visual."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="BIDS dataset root")
    parser.add_argument(
        "--kind", choices=("auditory", "visual"), help="Override auto-detection"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include subject/block-level records in JSON output",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    kind = args.kind or infer_kind(root)
    result = audit_auditory(root) if kind == "auditory" else audit_visual(root)
    if not args.details:
        result.pop("subjects", None)
        result.pop("blocks", None)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
