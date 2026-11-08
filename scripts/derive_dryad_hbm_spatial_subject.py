#!/usr/bin/env python3
"""Derive frozen channel-level Dryad robustness metrics for one participant.

This is a post-result robustness analysis. It reproduces the frozen artifact
rules and primary analysis window, then writes only condition-by-channel
summaries. It does not modify the original confirmatory outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

try:
    from scripts.audit_dryad_bdf_pilot import EXTERNAL_CHANNELS, find_status_events, reconstruct_trials
    from scripts.derive_dryad_confirmatory_subject import (
        CONDITIONS,
        evoked_local_snr_db,
        load_locked_subject,
        mean_itpc,
    )
    from scripts.develop_dryad_r2b_technical_subject import (
        ANALYSIS_WINDOW_END_SECONDS,
        ANALYSIS_WINDOW_START_SECONDS,
        FRONTOCENTRAL_ROI,
        ROI_FIXED_PEAK_TO_PEAK_UV,
        SCALP_FIXED_PEAK_TO_PEAK_UV,
        eeg_sequence_mask,
        robust_upper,
        sha256,
    )
    from scripts.qc_dryad_signal_pilot import preprocess_epoch
except ModuleNotFoundError:
    from audit_dryad_bdf_pilot import EXTERNAL_CHANNELS, find_status_events, reconstruct_trials
    from derive_dryad_confirmatory_subject import CONDITIONS, evoked_local_snr_db, load_locked_subject, mean_itpc
    from develop_dryad_r2b_technical_subject import (
        ANALYSIS_WINDOW_END_SECONDS,
        ANALYSIS_WINDOW_START_SECONDS,
        FRONTOCENTRAL_ROI,
        ROI_FIXED_PEAK_TO_PEAK_UV,
        SCALP_FIXED_PEAK_TO_PEAK_UV,
        eeg_sequence_mask,
        robust_upper,
        sha256,
    )
    from qc_dryad_signal_pilot import preprocess_epoch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIRMATORY_PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
DEFAULT_HBM_PROTOCOL = ROOT / "configs/hbm_additional_analysis_protocol_v1.json"
DEFAULT_TECHNICAL_ROOT = ROOT / "outputs/qc/dryad_raw_stream/R2b_validation_technical_v1"


def channel_condition_metrics(
    epochs: np.ndarray,
    sfreq: float,
    target_hz: float,
    analysis_slice: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ITPC and evoked local-SNR for trials x channels x samples."""
    array = np.asarray(epochs, dtype=float)
    if array.ndim != 3 or array.shape[0] < 4:
        raise ValueError("epochs must be trials x channels x samples with at least four trials")
    itpc = np.asarray([
        mean_itpc(array[:, channel, :], sfreq, target_hz, analysis_slice)
        for channel in range(array.shape[1])
    ])
    start, stop = analysis_slice
    snr = np.asarray([
        evoked_local_snr_db(array[:, channel, start:stop], sfreq, target_hz)
        for channel in range(array.shape[1])
    ])
    return itpc, snr


def run_subject(
    bdf_path: Path,
    subject: int,
    individual_theta_hz: float,
    technical_root: Path,
    confirmatory_protocol: Path,
    hbm_protocol: Path,
    output_dir: Path,
) -> dict[str, object]:
    confirmatory = load_locked_subject(confirmatory_protocol, subject)
    hbm = json.loads(hbm_protocol.read_text(encoding="utf-8"))
    expected_subjects = set(map(int, hbm["dryad_spatial_robustness"]["subjects"]))
    if subject not in expected_subjects:
        raise ValueError("subject is not in the HBM spatial robustness cohort")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")

    technical_dir = technical_root / f"S{subject}"
    technical = json.loads(
        (technical_dir / "technical_summary_r2b_development.json").read_text(encoding="utf-8")
    )
    if not technical["technical_pass_development_candidate"]:
        raise RuntimeError("subject did not pass the frozen technical gate")
    source_hash = sha256(bdf_path)
    if source_hash != technical["source_bdf_sha256"]:
        raise RuntimeError("BDF hash does not match frozen technical validation")

    channel_table = pd.read_csv(technical_dir / "channel_metrics_r2b_development.csv")
    good_channels = channel_table.loc[
        ~channel_table["r2b_bad_channel_development"].astype(bool), "channel"
    ].astype(str).tolist()
    roi = [channel for channel in FRONTOCENTRAL_ROI if channel in good_channels]
    if len(roi) < 3:
        raise RuntimeError("fewer than three frozen-QC frontocentral channels")

    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="error")
    scalp = [name for name in raw.ch_names if name not in EXTERNAL_CHANNELS | {"Status"}]
    if set(scalp) != set(channel_table["channel"].astype(str)):
        raise RuntimeError("BDF scalp-channel identity changed")
    events, _, _ = find_status_events(raw)
    trials = reconstruct_trials(events, float(raw.info["sfreq"]))
    trials = trials.loc[eeg_sequence_mask(trials)].copy().reset_index(drop=True)

    sfreq_raw = float(raw.info["sfreq"])
    sfreq = sfreq_raw / 4.0
    roi_indices = [good_channels.index(channel) for channel in roi]
    epoch_rows: list[dict[str, object]] = []
    active_epochs: dict[int, np.ndarray] = {}
    a0 = a1 = -1
    for index, row in enumerate(trials.itertuples(index=False)):
        onset = int(row.stimulation_sample)
        start = onset - int(round(3.0 * sfreq_raw))
        stop = onset + int(round(4.0 * sfreq_raw))
        if start < 0 or stop > raw.n_times:
            epoch_rows.append({"within_bounds": False, "median_ptp": np.nan, "roi_ptp": np.nan})
            continue
        filtered, sfreq = preprocess_epoch(
            raw.get_data(picks=good_channels, start=start, stop=stop), sfreq_raw
        )
        a0 = int(round((3.0 + ANALYSIS_WINDOW_START_SECONDS) * sfreq))
        a1 = int(round((3.0 + ANALYSIS_WINDOW_END_SECONDS) * sfreq))
        analysis = filtered[:, a0:a1]
        roi_signal = np.mean(analysis[roi_indices], axis=0)
        epoch_rows.append({
            "within_bounds": True,
            "median_ptp": float(np.median(np.ptp(analysis, axis=1))),
            "roi_ptp": float(np.ptp(roi_signal)),
        })
        if row.condition in CONDITIONS:
            active_epochs[index] = np.asarray(filtered, dtype=np.float32)

    artifacts = pd.DataFrame(epoch_rows)
    within = artifacts["within_bounds"].astype(bool)
    scalp_limit = robust_upper(artifacts.loc[within, "median_ptp"].to_numpy())
    roi_limit = robust_upper(artifacts.loc[within, "roi_ptp"].to_numpy())
    artifacts["usable"] = within & ~(
        (artifacts["median_ptp"] > SCALP_FIXED_PEAK_TO_PEAK_UV)
        | (artifacts["roi_ptp"] > ROI_FIXED_PEAK_TO_PEAK_UV)
        | (artifacts["median_ptp"] > scalp_limit)
        | (artifacts["roi_ptp"] > roi_limit)
    )
    if int(artifacts["usable"].sum()) != int(technical["n_technically_usable_epochs"]):
        raise RuntimeError("artifact reconstruction disagrees with frozen technical validation")
    trials = pd.concat([trials, artifacts], axis=1)

    target = 1.33 * float(individual_theta_hz)
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        indices = trials.index[(trials["condition"] == condition) & trials["usable"]].tolist()
        epochs = np.asarray([active_epochs[index] for index in indices], dtype=np.float32)
        itpc, evoked_snr = channel_condition_metrics(epochs, sfreq, target, (a0, a1))
        for channel, itpc_value, snr_value in zip(good_channels, itpc, evoked_snr):
            rows.append({
                "subject": subject,
                "condition": condition,
                "channel": channel,
                "target_frequency_hz": target,
                "n_usable_trials": len(indices),
                "mean_itpc_in_stimulation_window": float(itpc_value),
                "evoked_local_log_snr_db": float(snr_value),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "channel_condition_metrics.csv", index=False)
    summary = {
        "protocol_id": hbm["protocol_id"],
        "classification": hbm["dryad_spatial_robustness"]["classification"],
        "subject": subject,
        "source_bdf_sha256": source_hash,
        "confirmatory_protocol_id": confirmatory["protocol_id"],
        "individual_theta_hz": float(individual_theta_hz),
        "analysis_frequency_hz": target,
        "n_good_frozen_qc_channels": len(good_channels),
        "n_technical_usable_epochs_reproduced": int(artifacts["usable"].sum()),
        "status": "hbm_spatial_subject_complete",
    }
    (output_dir / "spatial_subject_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bdf", type=Path)
    parser.add_argument("subject", type=int)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--individual-theta-hz", type=float, required=True)
    parser.add_argument("--technical-root", type=Path, default=DEFAULT_TECHNICAL_ROOT)
    parser.add_argument("--confirmatory-protocol", type=Path, default=DEFAULT_CONFIRMATORY_PROTOCOL)
    parser.add_argument("--hbm-protocol", type=Path, default=DEFAULT_HBM_PROTOCOL)
    args = parser.parse_args()
    result = run_subject(
        args.bdf,
        args.subject,
        args.individual_theta_hz,
        args.technical_root,
        args.confirmatory_protocol,
        args.hbm_protocol,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
