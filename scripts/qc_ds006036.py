#!/usr/bin/env python3
"""Signal-level foundation gate for OpenNeuro ds006036.

This deliberately answers only whether short, reconstructed eyes-open pulse
windows contain frequency-specific sensor-level responses. It does not test
clinical hypotheses and does not treat eyes-closed data as a clean baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bids_events import PHOTO_RE, contiguous_pulse_segment, first_event_between  # noqa: E402


CORE_FREQUENCIES = (5.0, 10.0, 15.0, 20.0)
OCCIPITAL_ROI = ("O1", "O2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_snr_db(freqs: np.ndarray, psd: np.ndarray, target_hz: float) -> float:
    """Target-bin power relative to symmetric 1–2 Hz neighboring bins."""
    target = np.abs(freqs - target_hz) <= 0.26
    noise = ((freqs >= target_hz - 2.0) & (freqs <= target_hz - 1.0)) | (
        (freqs >= target_hz + 1.0) & (freqs <= target_hz + 2.0)
    )
    signal_power = float(np.mean(psd[target]))
    noise_power = float(np.mean(psd[noise]))
    return 10.0 * np.log10(signal_power / noise_power)


def blocks_from_annotations(raw: mne.io.BaseRaw, subject: str) -> list[dict]:
    """Reconstruct open-eye pulse windows from the annotations in this file.

    This is essential for the derivative files because ASR removed samples and
    EEGLAB adjusted event latencies; raw BIDS TSV onsets are no longer valid.
    """
    rows = sorted(
        (float(onset), str(description).strip())
        for onset, description in zip(raw.annotations.onset, raw.annotations.description)
    )
    labels = [
        (index, onset, float(match.group(1)))
        for index, (onset, value) in enumerate(rows)
        if (match := PHOTO_RE.match(value))
    ]
    pulses = [onset for onset, value in rows if value == "Photo/HV mark"]
    blocks = []
    for block_index, (_, label_onset, frequency) in enumerate(labels):
        next_label = labels[block_index + 1][1] if block_index + 1 < len(labels) else float("inf")
        pulse_segment = contiguous_pulse_segment(pulses, label_onset, frequency)
        pulse_stop = pulse_segment[-1] + max(0.1, 1.5 / frequency) if pulse_segment else next_label
        block_stop = min(next_label, pulse_stop)
        open_onset = first_event_between(rows, "open eyes", label_onset, block_stop)
        closed_onset = first_event_between(rows, "closed eyes", open_onset, block_stop) if open_onset is not None else None
        if open_onset is not None and closed_onset is not None:
            blocks.append(
                {
                    "subject": subject,
                    "block_index": block_index,
                    "frequency_hz": frequency,
                    "open_onset_s": open_onset,
                    "closed_onset_s": closed_onset,
                    "open_duration_s": closed_onset - open_onset,
                }
            )
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(args.root / "participants.tsv", sep="\t", encoding="utf-8-sig")
    participants.columns = [column.strip() for column in participants.columns]
    participant_rows = participants.set_index("participant_id").to_dict("index")
    manifest, recording_qc, metrics, all_channel_metrics = [], [], [], []
    for set_path in sorted(args.root.glob("sub-*/eeg/*_eeg.set")):
        sid = set_path.parts[-3]
        manifest.append(
            {"subject": sid, "path": str(set_path), "bytes": set_path.stat().st_size, "sha256": sha256(set_path)}
        )
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data()
        finite = bool(np.isfinite(data).all())
        # Average reference is explicit and reproducible; native Cz-reference
        # results will be a later sensitivity analysis.
        data = data - data.mean(axis=0, keepdims=True)
        subject_blocks = [
            block
            for block in blocks_from_annotations(raw, sid)
            if block["frequency_hz"] in CORE_FREQUENCIES and block["open_duration_s"] >= 2.5
        ]
        recording_qc.append(
            {
                "subject": sid,
                "n_channels": len(raw.ch_names),
                "sfreq_hz": sfreq,
                "duration_s": raw.times[-1],
                "all_finite": finite,
                "has_occipital_roi": all(channel in raw.ch_names for channel in OCCIPITAL_ROI),
                "n_usable_core_blocks": len(subject_blocks),
                "p99_abs_uv": float(np.nanpercentile(np.abs(data), 99) * 1e6),
            }
        )
        for block in subject_blocks:
            # Remove the first 0.5 s after eye opening and the last 0.1 s to
            # reduce state-transition and marker-edge contamination.
            start = int(round((block["open_onset_s"] + 0.5) * sfreq))
            stop = int(round((block["closed_onset_s"] - 0.1) * sfreq))
            if stop - start < int(2.0 * sfreq):
                continue
            segment = data[:, start:stop]
            nperseg = min(int(2.0 * sfreq), segment.shape[1])
            freqs, psd = welch(segment, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
            for index, channel in enumerate(raw.ch_names):
                all_channel_metrics.append(
                    {
                        "subject": sid,
                        "block_index": block["block_index"],
                        "group": str(participant_rows[sid]["Group"]).strip(),
                        "channel": channel,
                        "frequency_hz": block["frequency_hz"],
                        "local_snr_db": local_snr_db(freqs, psd[index], block["frequency_hz"]),
                    }
                )
            for channel in OCCIPITAL_ROI:
                index = raw.ch_names.index(channel)
                target = block["frequency_hz"]
                metrics.append(
                    {
                        "subject": sid,
                        "block_index": block["block_index"],
                        "group": str(participant_rows[sid]["Group"]).strip(),
                        "mmse": float(participant_rows[sid]["MMSE"]),
                        "channel": channel,
                        "frequency_hz": target,
                        "open_duration_s": block["open_duration_s"],
                        "analyzed_duration_s": (stop - start) / sfreq,
                        "p99_abs_uv": float(np.percentile(np.abs(segment[index]), 99) * 1e6),
                        "peak_to_peak_uv": float(np.ptp(segment[index]) * 1e6),
                        "local_snr_db": local_snr_db(freqs, psd[index], target),
                    }
                )

    manifest_df = pd.DataFrame(manifest)
    qc_df = pd.DataFrame(recording_qc)
    metrics_df = pd.DataFrame(metrics)
    all_channel_df = pd.DataFrame(all_channel_metrics)
    subject_frequency = (
        metrics_df.groupby(["subject", "group", "mmse", "frequency_hz"], as_index=False)
        .agg(local_snr_db=("local_snr_db", "mean"), analyzed_duration_s=("analyzed_duration_s", "mean"))
    )
    frequency_summary = (
        subject_frequency.groupby("frequency_hz", as_index=False)
        .agg(n_subjects=("subject", "nunique"), median_snr_db=("local_snr_db", "median"), min_snr_db=("local_snr_db", "min"), max_snr_db=("local_snr_db", "max"))
    )

    manifest_df.to_csv(args.output / "file_manifest_sha256.csv", index=False)
    qc_df.to_csv(args.output / "recording_qc.csv", index=False)
    metrics_df.to_csv(args.output / "occipital_block_metrics.csv", index=False)
    all_channel_df.to_csv(args.output / "all_channel_block_metrics.csv", index=False)
    (
        all_channel_df.groupby(["channel", "frequency_hz"], as_index=False)
        .agg(n_subjects=("subject", "nunique"), median_snr_db=("local_snr_db", "median"))
        .to_csv(args.output / "channel_frequency_summary.csv", index=False)
    )
    subject_frequency.to_csv(args.output / "subject_frequency_summary.csv", index=False)
    frequency_summary.to_csv(args.output / "frequency_summary.csv", index=False)
    summary = {
        "n_subjects": int(qc_df["subject"].nunique()),
        "n_usable_raw_blocks": int(metrics_df[["subject", "block_index"]].drop_duplicates().shape[0]),
        "n_usable_subject_frequency_cells": int(subject_frequency.shape[0]),
        "all_structural_qc_pass": bool(
            (qc_df["n_channels"] == 19).all()
            and (qc_df["sfreq_hz"] == 500).all()
            and qc_df["all_finite"].all()
            and qc_df["has_occipital_roi"].all()
        ),
        "analysis_note": "Frequency-specific SNR within reconstructed eyes-open pulse windows; no efficacy inference.",
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(frequency_summary.to_string(index=False))


if __name__ == "__main__":
    main()
