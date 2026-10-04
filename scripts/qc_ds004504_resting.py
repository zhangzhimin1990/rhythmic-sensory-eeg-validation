#!/usr/bin/env python3
"""Robust resting-state feature gate for ds004504 derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


POSTERIOR_ROI = ("P3", "Pz", "P4", "O1", "O2")
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}


def band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(np.trapezoid(psd[mask], freqs[mask]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--visual-summary", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(args.root / "participants.tsv", sep="\t", encoding="utf-8-sig")
    participants.columns = [column.strip() for column in participants.columns]
    participant_rows = participants.set_index("participant_id").to_dict("index")
    qc_rows, feature_rows, manifests = [], [], []

    for set_path in sorted(args.root.glob("sub-*/eeg/*_eeg.set")):
        sid = set_path.parts[-3]
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data()
        data = data - data.mean(axis=0, keepdims=True)
        window_samples = int(round(4.0 * sfreq))
        n_windows = data.shape[1] // window_samples
        windows = data[:, : n_windows * window_samples].reshape(len(raw.ch_names), n_windows, window_samples).transpose(1, 0, 2)
        peak_to_peak = np.ptp(windows, axis=-1).max(axis=1) * 1e6
        keep = peak_to_peak <= 300.0
        if not keep.any():
            keep = np.ones(n_windows, dtype=bool)
        freqs, psd = welch(
            windows[keep], fs=sfreq, nperseg=window_samples, noverlap=0, detrend="constant", axis=-1
        )
        median_psd = np.median(psd, axis=0)
        posterior_indices = [raw.ch_names.index(channel) for channel in POSTERIOR_ROI]
        posterior_psd = np.median(median_psd[posterior_indices], axis=0)
        total = band_power(freqs, posterior_psd, 1.0, 45.0)
        powers = {name: band_power(freqs, posterior_psd, *limits) for name, limits in BANDS.items()}
        alpha_mask = (freqs >= 7.0) & (freqs <= 13.0)
        iaf = float(freqs[alpha_mask][np.argmax(posterior_psd[alpha_mask])])
        exponent_mask = (freqs >= 2.0) & (freqs <= 30.0) & ~((freqs >= 7.0) & (freqs <= 13.0))
        exponent = float(-np.polyfit(np.log10(freqs[exponent_mask]), np.log10(posterior_psd[exponent_mask]), 1)[0])
        row = {
            "subject": sid,
            "group": str(participant_rows[sid]["Group"]).strip(),
            "age": float(participant_rows[sid]["Age"]),
            "mmse": float(participant_rows[sid]["MMSE"]),
            "posterior_iaf_hz": iaf,
            "iaf_distance_to_10hz": abs(iaf - 10.0),
            "slowing_ratio": (powers["delta"] + powers["theta"]) / (powers["alpha"] + powers["beta"]),
            "spectral_exponent": exponent,
        }
        row.update({f"relative_{name}": power / total for name, power in powers.items()})
        feature_rows.append(row)
        qc_rows.append(
            {
                "subject": sid,
                "n_channels": len(raw.ch_names),
                "sfreq_hz": sfreq,
                "duration_s": raw.times[-1],
                "all_finite": bool(np.isfinite(data).all()),
                "has_posterior_roi": all(channel in raw.ch_names for channel in POSTERIOR_ROI),
                "n_windows": n_windows,
                "n_windows_kept_300uv": int(keep.sum()),
                "fraction_windows_kept_300uv": float(keep.mean()),
            }
        )
        manifests.append({"subject": sid, "path": str(set_path), "bytes": set_path.stat().st_size, "sha256": sha256(set_path)})

    qc = pd.DataFrame(qc_rows)
    features = pd.DataFrame(feature_rows)
    pd.DataFrame(manifests).to_csv(args.output / "file_manifest_sha256.csv", index=False)
    qc.to_csv(args.output / "recording_qc.csv", index=False)
    features.to_csv(args.output / "resting_features.csv", index=False)
    if args.visual_summary:
        visual = pd.read_csv(args.visual_summary)
        paired = features.merge(visual, on="subject", suffixes=("_rest", "_visual"))
        paired.to_csv(args.output / "paired_rest_visual_features.csv", index=False)
    summary = {
        "n_subjects": int(features.subject.nunique()),
        "all_structural_qc_pass": bool((qc.n_channels == 19).all() and (qc.sfreq_hz == 500).all() and qc.all_finite.all() and qc.has_posterior_roi.all()),
        "median_fraction_windows_kept_300uv": float(qc.fraction_windows_kept_300uv.median()),
        "analysis_note": "Foundation feature extraction from dataset-provided derivatives; not a clinical model.",
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(features.to_string(index=False))


if __name__ == "__main__":
    main()
