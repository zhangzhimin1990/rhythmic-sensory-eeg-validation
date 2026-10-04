#!/usr/bin/env python3
"""Automated signal-QC and target-frequency pilot for one Dryad stimulation BDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal


EXTERNAL_CHANNELS = {"UP", "DOWN", "LEFT", "RIGHT", "EXG5", "EXG6", "EXG7", "EXG8"}
FRONTOCENTRAL_ROI = ["FC3", "FC1", "FCz", "FC2", "FC4"]
CONDITION_FREQUENCY = {
    "2_hz": lambda theta: 2.0,
    "f_theta": lambda theta: theta,
    "f_theta_plus": lambda theta: 1.33 * theta,
    "non_rhythmic": lambda theta: 1.33 * theta,
}
ANALYSIS_WINDOW_START_SECONDS = 0.25
ANALYSIS_WINDOW_END_SECONDS = 1.75


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(values)
    return 0.67448975 * (values - median) / mad


def band_mean(freqs: np.ndarray, values: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        raise ValueError(f"frequency band [{low}, {high}] has no bins")
    return np.nanmean(values[..., mask], axis=-1)


def local_log_snr_db(freqs: np.ndarray, power: np.ndarray, target: float) -> float:
    target_power = band_mean(freqs, power, target - 0.5, target + 0.5)
    left = band_mean(freqs, power, max(0.5, target - 2.0), max(0.75, target - 1.0))
    right = band_mean(freqs, power, target + 1.0, target + 2.0)
    background = np.nanmean(np.asarray([left, right]), axis=0)
    return float(10.0 * np.log10((target_power + 1e-30) / (background + 1e-30)))


def local_snr_interpretation(condition: str, target: float) -> str:
    if target < 3.0:
        return "descriptive_only_low_frequency_1f_boundary"
    if condition == "non_rhythmic":
        return "same_analysis_frequency_readout_not_stimulus_frequency_specificity"
    return "primary_frequency_specificity_check"


def flag_channels(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    result["z_log_rms"] = robust_z(np.log(np.maximum(result["rms_uv"], 1e-12)))
    result["z_line_noise_db"] = robust_z(result["line_noise_db"])
    result["z_high_frequency_db"] = robust_z(result["high_frequency_db"])
    result["z_low_correlation"] = robust_z(-result["median_channel_correlation"])
    reasons = []
    for row in result.itertuples(index=False):
        row_reasons = []
        if row.finite_fraction < 0.999:
            row_reasons.append("nonfinite")
        if row.rms_uv < 0.1:
            row_reasons.append("flat")
        if abs(row.z_log_rms) > 5:
            row_reasons.append("rms_outlier")
        if row.z_line_noise_db > 5 and row.line_noise_db > 3.0:
            row_reasons.append("line_noise_outlier")
        if row.z_high_frequency_db > 5 and row.high_frequency_db > -3.0:
            row_reasons.append("high_frequency_outlier")
        if row.z_low_correlation > 5 and row.median_channel_correlation < 0.2:
            row_reasons.append("low_correlation_outlier")
        reasons.append(";".join(row_reasons))
    result["bad_channel_reason"] = reasons
    result["bad_channel"] = result["bad_channel_reason"].ne("")
    return result


def channel_metrics(raw: mne.io.BaseRaw, scalp_names: list[str]) -> pd.DataFrame:
    sfreq = float(raw.info["sfreq"])
    window_samples = int(round(10.0 * sfreq))
    latest = max(0, raw.n_times - window_samples)
    starts = np.unique(np.linspace(0, latest, 18, dtype=int))
    chunks = []
    for start in starts:
        chunk = raw.get_data(picks=scalp_names, start=int(start), stop=int(start + window_samples))
        chunk = chunk * 1e6
        chunk = chunk - np.nanmean(chunk, axis=0, keepdims=True)
        chunk = chunk - np.nanmedian(chunk, axis=1, keepdims=True)
        chunk = signal.resample_poly(chunk, 1, 4, axis=1)
        chunks.append(chunk)
    data = np.concatenate(chunks, axis=1)
    sfreq_qc = sfreq / 4.0
    freqs, psd = signal.welch(
        data,
        fs=sfreq_qc,
        nperseg=int(4 * sfreq_qc),
        noverlap=int(2 * sfreq_qc),
        axis=1,
        detrend="linear",
    )
    line = band_mean(freqs, psd, 49.0, 51.0)
    line_neighbors = np.nanmean(
        np.vstack([band_mean(freqs, psd, 45.0, 47.0), band_mean(freqs, psd, 53.0, 55.0)]),
        axis=0,
    )
    high = band_mean(freqs, psd, 70.0, 100.0)
    physiological = band_mean(freqs, psd, 1.0, 45.0)
    correlations = np.corrcoef(data)
    rows = []
    for index, name in enumerate(scalp_names):
        other = np.delete(correlations[index], index)
        rows.append(
            {
                "channel": name,
                "finite_fraction": float(np.mean(np.isfinite(data[index]))),
                "rms_uv": float(np.sqrt(np.nanmean(data[index] ** 2))),
                "mad_uv": float(np.nanmedian(np.abs(data[index] - np.nanmedian(data[index])))),
                "line_noise_db": float(10 * np.log10((line[index] + 1e-30) / (line_neighbors[index] + 1e-30))),
                "high_frequency_db": float(10 * np.log10((high[index] + 1e-30) / (physiological[index] + 1e-30))),
                "median_channel_correlation": float(np.nanmedian(other)),
            }
        )
    return flag_channels(pd.DataFrame(rows))


def preprocess_epoch(data_v: np.ndarray, original_sfreq: float) -> tuple[np.ndarray, float]:
    data_uv = data_v * 1e6
    data_uv = data_uv - np.nanmean(data_uv, axis=0, keepdims=True)
    data_uv = signal.resample_poly(data_uv, 1, 4, axis=1)
    sfreq = original_sfreq / 4.0
    sos = signal.butter(4, [1.0, 45.0], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, data_uv, axis=1), sfreq


def select_epochs_by_trial_index(
    indexed_epochs: list[tuple[int, np.ndarray]],
    usable_trial_indices: list[int],
) -> np.ndarray:
    """Select retained epochs by trial identity, not their pre-rejection position."""
    epoch_by_trial: dict[int, np.ndarray] = {}
    for trial_index, epoch in indexed_epochs:
        trial_index = int(trial_index)
        if trial_index in epoch_by_trial:
            raise RuntimeError(f"duplicate epoch for trial_index={trial_index}")
        epoch_by_trial[trial_index] = epoch
    missing = [index for index in usable_trial_indices if index not in epoch_by_trial]
    if missing:
        raise RuntimeError(f"usable trials missing retained epochs: {missing}")
    if not usable_trial_indices:
        return np.empty((0, 0))
    return np.asarray([epoch_by_trial[index] for index in usable_trial_indices])


def trial_metrics(
    raw: mne.io.BaseRaw,
    trials: pd.DataFrame,
    good_channels: list[str],
    individual_theta: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sfreq_raw = float(raw.info["sfreq"])
    roi = [name for name in FRONTOCENTRAL_ROI if name in good_channels]
    if len(roi) < 3:
        raise RuntimeError(f"fewer than three usable frontocentral ROI channels: {roi}")
    good_indices = {name: index for index, name in enumerate(good_channels)}
    roi_indices = [good_indices[name] for name in roi]
    rows = []
    roi_epochs: dict[str, list[tuple[int, np.ndarray]]] = {
        condition: [] for condition in CONDITION_FREQUENCY
    }
    for trial in trials.itertuples(index=False):
        onset = int(trial.stimulation_sample)
        start = onset - int(round(3.0 * sfreq_raw))
        stop = onset + int(round(4.0 * sfreq_raw))
        if start < 0 or stop > raw.n_times:
            rows.append({"trial_index": trial.trial_index, "condition": trial.condition, "usable": False, "rejection_reason": "out_of_bounds"})
            continue
        data = raw.get_data(picks=good_channels, start=start, stop=stop)
        filtered, sfreq = preprocess_epoch(data, sfreq_raw)
        analysis_start = int(round((3.0 + ANALYSIS_WINDOW_START_SECONDS) * sfreq))
        analysis_stop = int(round((3.0 + ANALYSIS_WINDOW_END_SECONDS) * sfreq))
        analysis = filtered[:, analysis_start:analysis_stop]
        channel_ptp = np.ptp(analysis, axis=1)
        roi_full = np.nanmean(filtered[roi_indices], axis=0)
        roi_signal = roi_full[analysis_start:analysis_stop]
        freqs, power = signal.welch(roi_signal, fs=sfreq, nperseg=len(roi_signal), detrend="linear")
        target = CONDITION_FREQUENCY[trial.condition](individual_theta)
        rows.append(
            {
                "trial_index": int(trial.trial_index),
                "condition": trial.condition,
                "response_class": trial.response_class,
                "rt_ms": trial.rt_ms,
                "target_frequency_hz": target,
                "max_channel_peak_to_peak_uv": float(np.nanmax(channel_ptp)),
                "median_channel_peak_to_peak_uv": float(np.nanmedian(channel_ptp)),
                "roi_rms_uv": float(np.sqrt(np.nanmean(roi_signal ** 2))),
                "roi_peak_to_peak_uv": float(np.ptp(roi_signal)),
                "roi_local_log_snr_db": local_log_snr_db(freqs, power, target),
                "usable": True,
                "rejection_reason": "",
            }
        )
        roi_epochs[trial.condition].append((int(trial.trial_index), roi_full))
    result = pd.DataFrame(rows)
    valid = result["usable"].fillna(False)
    fixed_bad = (
        (result["median_channel_peak_to_peak_uv"] > 250.0)
        | (result["roi_peak_to_peak_uv"] > 150.0)
    )
    scalp_center = result.loc[valid, "median_channel_peak_to_peak_uv"].median()
    scalp_mad = (
        result.loc[valid, "median_channel_peak_to_peak_uv"] - scalp_center
    ).abs().median()
    scalp_robust_limit = float(scalp_center + 6.0 * scalp_mad)
    roi_center = result.loc[valid, "roi_peak_to_peak_uv"].median()
    roi_mad = (result.loc[valid, "roi_peak_to_peak_uv"] - roi_center).abs().median()
    roi_robust_limit = float(roi_center + 6.0 * roi_mad)
    robust_bad = (
        (result["median_channel_peak_to_peak_uv"] > scalp_robust_limit)
        | (result["roi_peak_to_peak_uv"] > roi_robust_limit)
    )
    artifact = valid & (fixed_bad | robust_bad)
    result.loc[artifact, "usable"] = False
    result.loc[artifact, "rejection_reason"] = np.where(
        fixed_bad[artifact] & robust_bad[artifact],
        "peak_to_peak_fixed_and_robust",
        np.where(fixed_bad[artifact], "peak_to_peak_fixed", "peak_to_peak_robust"),
    )
    summaries = []
    for condition, condition_rows in result.groupby("condition", sort=False):
        usable_indices = condition_rows.loc[condition_rows["usable"], "trial_index"].astype(int).tolist()
        epochs = select_epochs_by_trial_index(roi_epochs[condition], usable_indices)
        target = CONDITION_FREQUENCY[condition](individual_theta)
        sfreq = sfreq_raw / 4.0

        def aggregate_epoch_metrics(epoch_array: np.ndarray) -> tuple[float, float]:
            if not len(epoch_array):
                return np.nan, np.nan
            low = max(0.5, target - 1.0)
            high = target + 1.0
            sos = signal.butter(3, [low, high], btype="bandpass", fs=sfreq, output="sos")
            narrow = signal.sosfiltfilt(sos, epoch_array, axis=1)
            narrow = narrow[
                :,
                int(round((3.0 + ANALYSIS_WINDOW_START_SECONDS) * sfreq)):
                int(round((3.0 + ANALYSIS_WINDOW_END_SECONDS) * sfreq)),
            ]
            analytic = signal.hilbert(narrow, axis=1)
            itpc = np.abs(np.nanmean(analytic / np.maximum(np.abs(analytic), 1e-30), axis=0))
            itpc_mean = float(np.nanmean(itpc))
            evoked = np.nanmean(
                epoch_array[
                    :,
                    int(round((3.0 + ANALYSIS_WINDOW_START_SECONDS) * sfreq)):
                    int(round((3.0 + ANALYSIS_WINDOW_END_SECONDS) * sfreq)),
                ],
                axis=0,
            )
            freqs, evoked_power = signal.welch(evoked, fs=sfreq, nperseg=len(evoked), detrend="linear")
            evoked_snr = local_log_snr_db(freqs, evoked_power, target)
            return itpc_mean, evoked_snr

        itpc_mean, evoked_snr = aggregate_epoch_metrics(epochs)
        odd_itpc, odd_evoked_snr = aggregate_epoch_metrics(epochs[::2])
        even_itpc, even_evoked_snr = aggregate_epoch_metrics(epochs[1::2])
        summaries.append(
            {
                "condition": condition,
                "target_frequency_hz": target,
                "n_trials": int(len(condition_rows)),
                "n_usable_trials": int(condition_rows["usable"].sum()),
                "rejection_fraction": float(1 - condition_rows["usable"].mean()),
                "median_trial_local_log_snr_db": float(condition_rows.loc[condition_rows["usable"], "roi_local_log_snr_db"].median()),
                "evoked_local_log_snr_db": evoked_snr,
                "mean_itpc_in_stimulation_window": itpc_mean,
                "odd_trial_itpc": odd_itpc,
                "even_trial_itpc": even_itpc,
                "odd_trial_evoked_local_log_snr_db": odd_evoked_snr,
                "even_trial_evoked_local_log_snr_db": even_evoked_snr,
                "local_snr_interpretation": local_snr_interpretation(
                    condition, target
                ),
                "frontocentral_roi": ";".join(roi),
                "scalp_median_fixed_peak_to_peak_limit_uv": 250.0,
                "roi_fixed_peak_to_peak_limit_uv": 150.0,
                "scalp_median_robust_peak_to_peak_limit_uv": scalp_robust_limit,
                "roi_robust_peak_to_peak_limit_uv": roi_robust_limit,
            }
        )
    return result, pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bdf", type=Path)
    parser.add_argument("trials_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--individual-theta-hz", type=float, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_bdf(args.bdf, preload=False, verbose="error")
    scalp_names = [name for name in raw.ch_names if name not in EXTERNAL_CHANNELS | {"Status"}]
    if len(scalp_names) != 64:
        raise RuntimeError(f"expected 64 scalp channels, found {len(scalp_names)}")
    metrics = channel_metrics(raw, scalp_names)
    bad_channels = metrics.loc[metrics["bad_channel"], "channel"].tolist()
    good_channels = [name for name in scalp_names if name not in bad_channels]
    trials = pd.read_csv(args.trials_csv)
    trial_table, condition_table = trial_metrics(
        raw, trials, good_channels, args.individual_theta_hz
    )
    metrics.to_csv(args.output_dir / "channel_metrics.csv", index=False)
    trial_table.to_csv(args.output_dir / "trial_signal_qc.csv", index=False)
    condition_table.to_csv(args.output_dir / "condition_target_engagement.csv", index=False)
    summary = {
        "source_bdf": args.bdf.name,
        "individual_theta_hz": args.individual_theta_hz,
        "n_scalp_channels": len(scalp_names),
        "n_bad_channels": len(bad_channels),
        "bad_channels": bad_channels,
        "n_good_channels": len(good_channels),
        "n_trials": int(len(trial_table)),
        "n_usable_trials": int(trial_table["usable"].sum()),
        "trial_rejection_fraction": float(1 - trial_table["usable"].mean()),
        "analysis_window_seconds_after_stimulation_onset": [
            ANALYSIS_WINDOW_START_SECONDS,
            ANALYSIS_WINDOW_END_SECONDS,
        ],
        "preprocessing_identity": "automated R1 pilot; not an exact reproduction of manual EEGLAB/ICA cleaning",
    }
    (args.output_dir / "qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
