#!/usr/bin/env python3
"""Derive frozen confirmatory neural and behavioural features for one Dryad subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal

try:
    from scripts.audit_dryad_bdf_pilot import EXTERNAL_CHANNELS, find_status_events, reconstruct_trials
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
    from scripts.qc_dryad_signal_pilot import local_log_snr_db, preprocess_epoch
except ModuleNotFoundError:  # Direct execution via ``python scripts/...``.
    from audit_dryad_bdf_pilot import EXTERNAL_CHANNELS, find_status_events, reconstruct_trials
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
    from qc_dryad_signal_pilot import local_log_snr_db, preprocess_epoch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
DEFAULT_TECHNICAL_ROOT = ROOT / "outputs/qc/dryad_raw_stream/R2b_validation_technical_v1"
CONDITIONS = ("f_theta_plus", "non_rhythmic")


def narrowband_unit_phase(epochs: np.ndarray, sfreq: float, centre_hz: float) -> np.ndarray:
    """Return unit analytic phase for trials x samples using the frozen ±1-Hz band."""
    array = np.asarray(epochs, dtype=float)
    if array.ndim != 2 or not len(array):
        raise ValueError("epochs must be a non-empty trials-by-samples array")
    low, high = centre_hz - 1.0, centre_hz + 1.0
    if low <= 0 or high >= sfreq / 2:
        raise ValueError("analysis band is outside valid filter limits")
    sos = signal.butter(3, [low, high], btype="bandpass", fs=sfreq, output="sos")
    analytic = signal.hilbert(signal.sosfiltfilt(sos, array, axis=1), axis=1)
    return analytic / np.maximum(np.abs(analytic), 1e-30)


def _trim(values: np.ndarray, analysis_slice: tuple[int, int] | None) -> np.ndarray:
    if analysis_slice is None:
        return values
    start, stop = analysis_slice
    if not 0 <= start < stop <= values.shape[-1]:
        raise ValueError("analysis slice is outside the epoch")
    return values[..., start:stop]


def mean_itpc(
    epochs: np.ndarray,
    sfreq: float,
    centre_hz: float,
    analysis_slice: tuple[int, int] | None = None,
) -> float:
    unit = narrowband_unit_phase(epochs, sfreq, centre_hz)
    unit = _trim(unit, analysis_slice)
    return float(np.mean(np.abs(np.mean(unit, axis=0))))


def leave_one_trial_out_phase_alignment(
    epochs: np.ndarray,
    sfreq: float,
    centre_hz: float,
    analysis_slice: tuple[int, int] | None = None,
) -> np.ndarray:
    """Score each trial against the circular mean phase of all other trials."""
    unit = narrowband_unit_phase(epochs, sfreq, centre_hz)
    if len(unit) < 3:
        return np.full(len(unit), np.nan)
    total = np.sum(unit, axis=0, keepdims=True)
    reference = total - unit
    reference /= np.maximum(np.abs(reference), 1e-30)
    unit = _trim(unit, analysis_slice)
    reference = _trim(reference, analysis_slice)
    return np.mean(np.real(unit * np.conjugate(reference)), axis=1)


def total_power_local_snr_db(epochs: np.ndarray, sfreq: float, target: float) -> float:
    freqs, power = signal.welch(
        np.asarray(epochs, dtype=float), fs=sfreq,
        nperseg=np.asarray(epochs).shape[1], axis=1, detrend="linear"
    )
    return local_log_snr_db(freqs, np.mean(power, axis=0), target)


def evoked_local_snr_db(epochs: np.ndarray, sfreq: float, target: float) -> float:
    evoked = np.mean(np.asarray(epochs, dtype=float), axis=0)
    freqs, power = signal.welch(evoked, fs=sfreq, nperseg=len(evoked), detrend="linear")
    return local_log_snr_db(freqs, power, target)


def induced_local_snr_db(epochs: np.ndarray, sfreq: float, target: float) -> float:
    array = np.asarray(epochs, dtype=float)
    residual = array - np.mean(array, axis=0, keepdims=True)
    freqs, power = signal.welch(
        residual, fs=sfreq, nperseg=array.shape[1], axis=1, detrend="linear"
    )
    return local_log_snr_db(freqs, np.mean(power, axis=0), target)


def condition_metrics(
    epochs: np.ndarray,
    sfreq: float,
    target: float,
    analysis_slice: tuple[int, int] | None = None,
) -> dict[str, float]:
    epochs = np.asarray(epochs, dtype=float)
    if len(epochs) < 4:
        raise ValueError("at least four usable trials are required")
    analysis_epochs = _trim(epochs, analysis_slice)
    target_itpc = mean_itpc(epochs, sfreq, target, analysis_slice)
    midpoint = len(epochs) // 2
    return {
        "mean_itpc_in_stimulation_window": target_itpc,
        "evoked_local_log_snr_db": evoked_local_snr_db(analysis_epochs, sfreq, target),
        "total_power_local_log_snr_db": total_power_local_snr_db(analysis_epochs, sfreq, target),
        "induced_local_log_snr_db": induced_local_snr_db(analysis_epochs, sfreq, target),
        "odd_trial_itpc": mean_itpc(epochs[::2], sfreq, target, analysis_slice),
        "even_trial_itpc": mean_itpc(epochs[1::2], sfreq, target, analysis_slice),
        "first_half_itpc": mean_itpc(epochs[:midpoint], sfreq, target, analysis_slice),
        "second_half_itpc": mean_itpc(epochs[midpoint:], sfreq, target, analysis_slice),
        "odd_trial_evoked_local_log_snr_db": evoked_local_snr_db(analysis_epochs[::2], sfreq, target),
        "even_trial_evoked_local_log_snr_db": evoked_local_snr_db(analysis_epochs[1::2], sfreq, target),
        "first_half_evoked_local_log_snr_db": evoked_local_snr_db(analysis_epochs[:midpoint], sfreq, target),
        "second_half_evoked_local_log_snr_db": evoked_local_snr_db(analysis_epochs[midpoint:], sfreq, target),
    }


def load_locked_subject(protocol_path: Path, subject: int) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if subject not in set(map(int, protocol["confirmatory_subjects"])):
        raise ValueError("subject is not in the locked 35-person confirmatory set")
    return protocol


def run_subject(
    bdf_path: Path,
    subject: int,
    individual_theta_hz: float,
    technical_root: Path,
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    protocol = load_locked_subject(protocol_path, subject)
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
        raise RuntimeError("BDF hash does not match the technical-validation source")
    channels = pd.read_csv(technical_dir / "channel_metrics_r2b_development.csv")
    good_channels = channels.loc[~channels["r2b_bad_channel_development"].astype(bool), "channel"].tolist()
    roi = [channel for channel in FRONTOCENTRAL_ROI if channel in good_channels]
    if len(roi) < 3:
        raise RuntimeError("fewer than three frozen-QC ROI channels")

    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="error")
    scalp = [name for name in raw.ch_names if name not in EXTERNAL_CHANNELS | {"Status"}]
    if set(scalp) != set(channels["channel"].astype(str)):
        raise RuntimeError("BDF scalp-channel identity changed")
    events, _, _ = find_status_events(raw)
    trials = reconstruct_trials(events, float(raw.info["sfreq"]))
    trials = trials.loc[eeg_sequence_mask(trials)].copy().reset_index(drop=True)
    sfreq_raw = float(raw.info["sfreq"])
    roi_indices = [good_channels.index(channel) for channel in roi]
    epoch_rows: list[dict[str, object]] = []
    roi_epochs: list[np.ndarray | None] = []
    sfreq = sfreq_raw / 4.0
    for row in trials.itertuples(index=False):
        onset = int(row.stimulation_sample)
        start = onset - int(round(3.0 * sfreq_raw))
        stop = onset + int(round(4.0 * sfreq_raw))
        if start < 0 or stop > raw.n_times:
            epoch_rows.append({"within_bounds": False, "median_ptp": np.nan, "roi_ptp": np.nan})
            roi_epochs.append(None)
            continue
        filtered, sfreq = preprocess_epoch(raw.get_data(picks=good_channels, start=start, stop=stop), sfreq_raw)
        a0 = int(round((3.0 + ANALYSIS_WINDOW_START_SECONDS) * sfreq))
        a1 = int(round((3.0 + ANALYSIS_WINDOW_END_SECONDS) * sfreq))
        analysis = filtered[:, a0:a1]
        roi_signal = np.mean(analysis[roi_indices], axis=0)
        roi_signal_full = np.mean(filtered[roi_indices], axis=0)
        epoch_rows.append({
            "within_bounds": True,
            "median_ptp": float(np.median(np.ptp(analysis, axis=1))),
            "roi_ptp": float(np.ptp(roi_signal)),
        })
        roi_epochs.append(roi_signal_full)
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
        raise RuntimeError("confirmatory artifact reconstruction disagrees with technical validation")
    trials = pd.concat([trials, artifacts], axis=1)

    condition_rows = []
    trial_rows = []
    target = 1.33 * float(individual_theta_hz)
    for condition in CONDITIONS:
        all_indices = trials.index[trials["condition"] == condition].tolist()
        indices = trials.index[(trials["condition"] == condition) & trials["usable"]].tolist()
        epochs = np.asarray([roi_epochs[index] for index in indices])
        metrics = condition_metrics(epochs, sfreq, target, (a0, a1))
        condition_rows.append({
            "subject": subject,
            "condition": condition,
            "target_frequency_hz": target,
            "n_event_valid_trials": len(all_indices),
            "n_usable_trials": len(indices),
            "frontocentral_roi": ";".join(roi),
            **metrics,
        })
        alignment = leave_one_trial_out_phase_alignment(epochs, sfreq, target, (a0, a1))
        score_by_index = {index: float(score) for index, score in zip(indices, alignment)}
        for index in all_indices:
            row = trials.loc[index]
            trial_rows.append({
                "subject": subject,
                "trial_index": int(row["trial_index"]),
                "condition": condition,
                "congruency": row["congruency"],
                "response_class": row["response_class"],
                "correct": row["response_class"] == "correct",
                "rt_ms": row["rt_ms"],
                "neural_usable": bool(row["usable"]),
                "phase_alignment_loo": score_by_index.get(index, np.nan),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(condition_rows).to_csv(output_dir / "condition_neural_metrics.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(output_dir / "trial_neural_behaviour.csv", index=False)
    summary = {
        "subject": subject,
        "source_bdf_name": bdf_path.name,
        "source_bdf_sha256": source_hash,
        "individual_theta_hz": float(individual_theta_hz),
        "analysis_frequency_hz": target,
        "n_good_frozen_qc_channels": len(good_channels),
        "frontocentral_roi": roi,
        "n_technical_usable_epochs_reproduced": int(artifacts["usable"].sum()),
        "confirmatory_conditions": list(CONDITIONS),
        "primary_metric": protocol["primary_neural_estimand"]["metric"],
        "status": "confirmatory_subject_derivation_complete",
    }
    (output_dir / "confirmatory_subject_summary.json").write_text(
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
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result = run_subject(
        args.bdf, args.subject, args.individual_theta_hz,
        args.technical_root, args.protocol, args.output_dir
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
