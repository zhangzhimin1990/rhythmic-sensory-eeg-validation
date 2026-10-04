#!/usr/bin/env python3
"""Derive run-level ASSR and behavior features for OpenNeuro ds006780."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import tfr_array_morlet
from scipy.stats import norm

from derive_ds007648_trial_features import robust_upper, sample_channel_quality


ROI = ("FCz", "FC3", "FC4")
REFERENCE = ("TP7", "TP8")
FEATURE_ALGORITHM_VERSION = "3.0.0"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_responses(events: pd.DataFrame, response_window: float = 1.0) -> pd.DataFrame:
    stimuli = events.loc[
        events.trial_type.str.match(r"^(27|40)_Hz_(Standard|Oddball)$", na=False)
    ].copy()
    stimuli = stimuli.sort_values("onset").reset_index(drop=True)
    responses = events.loc[events.trial_type.eq("Response_button"), "onset"].to_numpy()
    assigned = np.zeros(len(stimuli), dtype=bool)
    reaction_time = np.full(len(stimuli), np.nan)
    onsets = stimuli.onset.to_numpy()
    for response in responses:
        candidate = np.where(
            (onsets <= response) & ((response - onsets) <= response_window)
        )[0]
        if len(candidate):
            index = candidate[-1]
            if not assigned[index]:
                assigned[index] = True
                reaction_time[index] = response - onsets[index]
    stimuli["response_assigned"] = assigned
    stimuli["reaction_time"] = reaction_time
    stimuli["standard_frequency_hz"] = np.where(
        stimuli.trial_type.isin(["40_Hz_Standard", "27_Hz_Oddball"]), 40, 27
    )
    stimuli["is_oddball"] = stimuli.trial_type.str.contains("Oddball")
    stimuli.attrs["unassigned_responses"] = int(len(responses) - assigned.sum())
    return stimuli


def behavior_by_frequency(
    events: pd.DataFrame, response_window: float = 1.0
) -> list[dict[str, float | int]]:
    stimuli = assign_responses(events, response_window=response_window)
    rows = []
    for frequency, group in stimuli.groupby("standard_frequency_hz"):
        oddball = group.is_oddball.to_numpy()
        response = group.response_assigned.to_numpy()
        hits = int((oddball & response).sum())
        misses = int((oddball & ~response).sum())
        false_alarms = int((~oddball & response).sum())
        correct_rejections = int((~oddball & ~response).sum())
        hit_rate = (hits + 0.5) / (hits + misses + 1)
        false_alarm_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
        rows.append(
            {
                "frequency_hz": int(frequency),
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_rejections": correct_rejections,
                "dprime": float(norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)),
                "median_hit_rt": float(group.loc[oddball & response, "reaction_time"].median()),
                "unassigned_responses_total": stimuli.attrs["unassigned_responses"],
            }
        )
    return rows


def morlet_metrics(
    epochs_uv: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    target_hz: int,
) -> dict[str, float]:
    frequencies = np.arange(target_hz - 5, target_hz + 6, dtype=float)
    n_cycles = 0.166 * np.pi * frequencies / np.sqrt(2 * np.log(2))
    complex_tfr = tfr_array_morlet(
        epochs_uv,
        sfreq=sfreq,
        freqs=frequencies,
        n_cycles=n_cycles,
        output="complex",
        use_fft=True,
        decim=1,
        n_jobs=1,
        verbose="error",
    )
    power = np.abs(complex_tfr) ** 2
    baseline = (times >= -0.15) & (times <= -0.05)
    response = (times >= 0.2) & (times <= 0.5)
    baseline_power = power[..., baseline].mean(axis=-1, keepdims=True)
    percent_change = (power - baseline_power) / np.maximum(
        baseline_power, np.finfo(float).tiny
    )
    trial_power_change = 100 * percent_change[..., response].mean(axis=(1, 2, 3))
    target_index = int(np.where(frequencies == target_hz)[0][0])
    response_signal = epochs_uv[..., response].mean(axis=0)
    response_signal = response_signal - response_signal.mean(axis=-1, keepdims=True)
    tapered = response_signal * np.hanning(response_signal.shape[-1])
    spectrum = np.abs(np.fft.rfft(tapered, axis=-1)) ** 2
    fft_frequencies = np.fft.rfftfreq(tapered.shape[-1], d=1 / sfreq)
    fft_target_index = int(np.argmin(np.abs(fft_frequencies - target_hz)))
    fft_flank_indices = np.array(
        [fft_target_index - 3, fft_target_index - 2,
         fft_target_index + 2, fft_target_index + 3]
    )
    target_response_power = spectrum[..., fft_target_index]
    flank_response_power = spectrum[..., fft_flank_indices].mean(axis=-1)
    channel_local_log_snr_db = 10 * np.log10(
        np.maximum(target_response_power, np.finfo(float).tiny)
        / np.maximum(flank_response_power, np.finfo(float).tiny)
    )
    target_complex = complex_tfr[:, :, target_index, :]
    unit_phase = target_complex / np.maximum(np.abs(target_complex), np.finfo(float).tiny)
    itpc_time_channel = np.abs(unit_phase[..., response].mean(axis=0))
    odd_itpc = np.abs(unit_phase[::2, ..., response].mean(axis=0)).mean()
    even_itpc = np.abs(unit_phase[1::2, ..., response].mean(axis=0)).mean()
    return {
        "morlet_power_percent_change_mean": float(trial_power_change.mean()),
        "morlet_power_percent_change_median": float(np.median(trial_power_change)),
        "local_log_snr_db_mean": float(channel_local_log_snr_db.mean()),
        "local_log_snr_db_median": float(np.median(channel_local_log_snr_db)),
        "itpc": float(itpc_time_channel.mean()),
        "itpc_odd_trials": float(odd_itpc),
        "itpc_even_trials": float(even_itpc),
    }


def derive_run(
    bdf_path: Path,
    events_path: Path,
    channels_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    stem = bdf_path.name.removesuffix("_eeg.bdf")
    subject = stem.split("_", 1)[0]
    run = stem.split("run-")[-1]
    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="error")
    events = pd.read_csv(events_path, sep="\t")
    channels = pd.read_csv(channels_path, sep="\t")
    eeg_names = channels.loc[channels.type.eq("EEG"), "name"].tolist()
    eeg_names = [name for name in eeg_names if name in raw.ch_names]
    if len(eeg_names) != 64:
        raise RuntimeError(f"{stem}: expected 64 EEG channels, found {len(eeg_names)}")
    channel_qc = sample_channel_quality(raw, eeg_names, analysis_stop=raw.n_times)
    good_eeg = channel_qc.loc[~channel_qc.bad, "channel"].tolist()
    missing_roi = sorted(set(ROI) - set(good_eeg))
    if len(good_eeg) < 50 or missing_roi:
        raise RuntimeError(f"{stem}: unusable channels; n_good={len(good_eeg)}, ROI={missing_roi}")

    sfreq = float(raw.info["sfreq"])
    tmin, tmax = -0.6, 0.8
    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))
    times = np.arange(stop_offset - start_offset) / sfreq + tmin
    index = {name: idx for idx, name in enumerate(good_eeg)}
    roi_indices = [index[name] for name in ROI]
    reference_names = [name for name in REFERENCE if name in index]
    reference_indices = [index[name] for name in reference_names]
    epoch_baseline = (times >= -0.2) & (times <= 0)

    cache: list[dict[str, object]] = []
    standard_events = events.loc[
        events.trial_type.str.match(r"^(27|40)_Hz_Standard$", na=False)
    ].copy()
    for _, event in standard_events.iterrows():
        sample = int(event["sample"])
        start, stop = sample + start_offset, sample + stop_offset
        row: dict[str, object] = {
            "frequency_hz": int(event.trial_type.split("_")[0]),
            "sample": sample,
            "in_bounds": start >= 0 and stop <= raw.n_times,
        }
        if not row["in_bounds"]:
            cache.append(row)
            continue
        eeg = raw.get_data(picks=good_eeg, start=start, stop=stop) * 1e6
        finite = np.isfinite(eeg).all(axis=1)
        roi_finite = bool(finite[roi_indices].all())
        enough_finite = int(finite.sum()) >= 50
        reference_finite = bool(reference_indices) and bool(finite[reference_indices].all())
        if enough_finite and reference_finite:
            reference_signal = eeg[reference_indices].mean(axis=0, keepdims=True)
            eeg[finite] -= reference_signal
            eeg[finite] -= eeg[finite][:, epoch_baseline].mean(axis=1, keepdims=True)
            p2p_99 = float(np.quantile(np.ptp(eeg[finite], axis=1), 0.99))
        else:
            p2p_99 = np.nan
        row.update(
            {
                "p2p_99_uv": p2p_99,
                "enough_finite": enough_finite,
                "roi_finite": roi_finite,
                "reference_finite": reference_finite,
                "roi_epoch_uv": eeg[roi_indices] if roi_finite and reference_finite else None,
            }
        )
        cache.append(row)

    candidate_p2p = np.array(
        [x.get("p2p_99_uv", np.nan) for x in cache if x.get("in_bounds", False)]
    )
    artifact_limit = min(500.0, robust_upper(candidate_p2p))
    run_rows = []
    for frequency in (40, 27):
        selected = [
            x
            for x in cache
            if x.get("frequency_hz") == frequency
            and x.get("in_bounds", False)
            and x.get("enough_finite", False)
            and x.get("roi_finite", False)
            and x.get("reference_finite", False)
            and np.isfinite(x.get("p2p_99_uv", np.nan))
            and x["p2p_99_uv"] <= artifact_limit
        ]
        row: dict[str, object] = {
            "subject": subject,
            "run": run,
            "frequency_hz": frequency,
            "n_standard_expected": int((standard_events.trial_type == f"{frequency}_Hz_Standard").sum()),
            "n_standard_valid": len(selected),
            "artifact_limit_uv": artifact_limit,
        }
        if len(selected) >= 20:
            epochs = np.stack([x["roi_epoch_uv"] for x in selected])
            row.update(morlet_metrics(epochs, sfreq, times, frequency))
        run_rows.append(row)

    behavior = {row["frequency_hz"]: row for row in behavior_by_frequency(events)}
    for row in run_rows:
        row.update(behavior.get(row["frequency_hz"], {}))
        row.update(
            {
                "feature_algorithm_version": FEATURE_ALGORITHM_VERSION,
                "reference_channels": ";".join(reference_names),
                "n_reference_channels": len(reference_names),
                "sfreq": sfreq,
                "n_bad_channels": int(channel_qc.bad.sum()),
                "bad_channels": ";".join(channel_qc.loc[channel_qc.bad, "channel"]),
                "signal_bytes": bdf_path.stat().st_size,
                "input_bdf_sha256": sha256_file(bdf_path),
                "input_events_sha256": sha256_file(events_path),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(output_dir / f"{stem}_features.csv", index=False)
    channel_qc.insert(0, "subject", subject)
    channel_qc.insert(1, "run", run)
    channel_qc.to_csv(output_dir / f"{stem}_channel_qc.csv", index=False)
    summary = {
        "subject": subject,
        "run": run,
        "n_rows": len(run_rows),
        "n_bad_channels": int(channel_qc.bad.sum()),
        "artifact_limit_uv": artifact_limit,
        "features": run_rows,
    }
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdf", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--channels", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/derived/ds006780_assr_features")
    )
    args = parser.parse_args()
    stem = args.bdf.name.removesuffix("_eeg.bdf")
    events = args.events or args.bdf.with_name(f"{stem}_events.tsv")
    channels = args.channels or args.bdf.with_name(f"{stem}_channels.tsv")
    summary = derive_run(args.bdf, events, channels, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
