#!/usr/bin/env python3
"""Signal-level gate for a small, preselected ds006222 QC sample.

The released BIDS events encode task colour changes but not trial outcomes or
reaction times.  This script therefore limits itself to data integrity, event
integrity, and spectral target-engagement checks; it does not fabricate
behavioural endpoints from EEG triggers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


SUBJECTS = ("S060", "S018", "S058", "S055", "S081", "S084")
REPORT_ROI = ("Fp1", "Cz", "Oz")


def normalize_event(value: object) -> int | None:
    match = re.search(r"-?\d+", str(value))
    if match is None:
        return None
    code = int(match.group())
    if 61440 <= code <= 61695:
        code -= 61440
    elif 49152 <= code <= 49407:
        code -= 49152
    return code


def annex_size(path: Path) -> int | None:
    """Read the expected byte count from a git-annex symlink target."""
    if not path.is_symlink():
        return None
    match = re.search(r"SHA256E-s(\d+)--", str(path.readlink()))
    return int(match.group(1)) if match else None


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def spectral_metrics(raw: mne.io.BaseRaw) -> tuple[pd.DataFrame, int]:
    codes = np.array([normalize_event(x) for x in raw.annotations.description])
    onsets = np.asarray(raw.annotations.onset)
    trial_onsets = onsets[codes == 2]
    trial_onsets = trial_onsets[trial_onsets >= 4.0]
    sfreq = float(raw.info["sfreq"])
    rows: list[dict[str, float | str]] = []
    eeg_channels = [
        name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"
    ]
    for channel in eeg_channels:
        epochs = []
        for onset in trial_onsets:
            start = int(round((onset - 4.0) * sfreq))
            stop = int(round(onset * sfreq))
            segment = raw.get_data(picks=[channel], start=start, stop=stop)[0]
            if segment.size == int(round(4.0 * sfreq)) and np.isfinite(segment).all():
                epochs.append(segment)
        data = np.asarray(epochs)
        centered = data - np.median(data, axis=1, keepdims=True)
        frequencies, psd = welch(
            data,
            fs=sfreq,
            window="hamming",
            nperseg=int(2 * sfreq),
            noverlap=int(sfreq),
            nfft=1024,
            axis=-1,
        )
        mean_psd = np.mean(psd, axis=0)
        target = (frequencies >= 39) & (frequencies <= 41)
        flank = ((frequencies >= 31) & (frequencies < 39)) | (
            (frequencies > 41) & (frequencies <= 49)
        )
        peak = float(mean_psd[np.argmin(np.abs(frequencies - 40.0))])
        flank_values = mean_psd[flank]
        rows.append(
            {
                "channel": channel,
                "preselected_report_roi": channel in REPORT_ROI,
                "n_attention_epochs": int(data.shape[0]),
                "median_abs_centered_uv": float(np.median(np.abs(centered)) * 1e6),
                "p99_abs_centered_uv": float(np.quantile(np.abs(centered), 0.99) * 1e6),
                "power_40hz": peak,
                "flank_mean": float(np.mean(flank_values)),
                "flank_sd": float(np.std(flank_values, ddof=1)),
                "peak_over_flank": float(peak / np.mean(flank_values)),
                "peak_z_flank": float(
                    (peak - np.mean(flank_values)) / np.std(flank_values, ddof=1)
                ),
                "band_39_41_fraction": float(
                    np.trapezoid(mean_psd[target], frequencies[target])
                    / np.trapezoid(mean_psd[(frequencies >= 2) & (frequencies <= 55)],
                                   frequencies[(frequencies >= 2) & (frequencies <= 55)])
                ),
            }
        )
    return pd.DataFrame(rows), len(trial_onsets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, default=Path("data/public/ds006222/v1.0.1"))
    parser.add_argument("--signal-root", type=Path, default=Path("data/public/ds006222/qc_sample"))
    parser.add_argument("--output", type=Path, default=Path("outputs/planning/ds006222_signal_qc"))
    parser.add_argument("--hash", action="store_true", help="Compute SHA-256 for complete files.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(args.metadata_root / "participants.tsv", sep="\t", keep_default_na=False)
    groups = dict(zip(participants.participant_id.str.removeprefix("sub-"), participants.Group))
    inventory: list[dict[str, object]] = []
    spectra: list[pd.DataFrame] = []

    for subject in SUBJECTS:
        stem = f"sub-{subject}_ses-1_task-PVT_eeg"
        signal_dir = args.signal_root / f"sub-{subject}" / "ses-1" / "eeg"
        metadata_dir = args.metadata_root / f"sub-{subject}" / "ses-1" / "eeg"
        set_path = signal_dir / f"{stem}.set"
        fdt_path = signal_dir / f"{stem}.fdt"
        expected_set = annex_size(metadata_dir / f"{stem}.set")
        expected_fdt = annex_size(metadata_dir / f"{stem}.fdt")
        complete = (
            set_path.exists() and fdt_path.exists()
            and set_path.stat().st_size == expected_set
            and fdt_path.stat().st_size == expected_fdt
        )
        row: dict[str, object] = {
            "subject": subject,
            "group": groups.get(subject),
            "complete": complete,
            "set_bytes": set_path.stat().st_size if set_path.exists() else None,
            "set_expected_bytes": expected_set,
            "fdt_bytes": fdt_path.stat().st_size if fdt_path.exists() else None,
            "fdt_expected_bytes": expected_fdt,
        }
        if not complete:
            inventory.append(row)
            continue
        if args.hash:
            row["set_sha256"] = sha256(set_path)
            row["fdt_sha256"] = sha256(fdt_path)
        raw = mne.io.read_raw_eeglab(set_path, preload=False, verbose="error")
        metrics, n_trials = spectral_metrics(raw)
        row.update(
            {
                "n_channels": len(raw.ch_names),
                "n_eeg": sum(kind == "eeg" for kind in raw.get_channel_types()),
                "n_misc": sum(kind == "misc" for kind in raw.get_channel_types()),
                "sfreq": float(raw.info["sfreq"]),
                "n_times": int(raw.n_times),
                "duration_seconds": float(raw.times[-1]),
                "n_annotations": len(raw.annotations),
                "n_color_change_events": n_trials,
            }
        )
        inventory.append(row)
        metrics.insert(0, "group", groups.get(subject))
        metrics.insert(0, "subject", subject)
        spectra.append(metrics)

    inventory_frame = pd.DataFrame(inventory)
    spectra_frame = pd.concat(spectra, ignore_index=True) if spectra else pd.DataFrame()
    response_rows = []
    if not spectra_frame.empty:
        for (subject, group), values in spectra_frame.groupby(["subject", "group"]):
            modulated = values.loc[values.peak_z_flank >= 3]
            left = modulated.channel.str.match(r".*[13579]$").sum()
            right = modulated.channel.str.match(r".*[02468]$").sum()
            response_rows.append(
                {
                    "subject": subject,
                    "group": group,
                    "n_modulated_channels_z_ge_3": len(modulated),
                    "n_left_modulated": int(left),
                    "n_right_modulated": int(right),
                    "published_response_rule_proxy": bool(len(modulated) >= 3 and left >= 1 and right >= 1),
                }
            )
    response_frame = pd.DataFrame(response_rows)
    inventory_frame.to_csv(args.output / "file_and_recording_qc.csv", index=False)
    spectra_frame.to_csv(args.output / "spectral_target_engagement_qc.csv", index=False)
    response_frame.to_csv(args.output / "subject_response_rule_qc.csv", index=False)
    summary = {
        "subjects_preselected": len(SUBJECTS),
        "subjects_complete": int(inventory_frame.complete.sum()),
        "groups_complete": inventory_frame.loc[inventory_frame.complete].group.value_counts().to_dict(),
        "published_response_rule_proxy_pass": (
            int(response_frame.published_response_rule_proxy.sum()) if not response_frame.empty else 0
        ),
        "behavior_reconstructable_from_released_events": False,
        "reason": "Reaction times and trial outcomes reside in unreleased PsychoPy CSV files; BIDS events contain colour-change/task-boundary triggers only.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
