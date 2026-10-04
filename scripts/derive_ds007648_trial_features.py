#!/usr/bin/env python3
"""Derive leakage-safe trial features for OpenNeuro ds007648.

The script never uses correctness or response time for EEG quality control. All
neural predictors are computed before target onset (0 s is tagging/cue onset;
the target follows the 3 s tagging interval).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin, hilbert, welch


CENTRAL_ROI = ("Cz", "FC1", "FC2", "C1", "C2", "FCz")
OCCIPITAL_ROI = ("POz", "O1", "O2", "Oz")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_upper(values: np.ndarray, z: float = 6.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("inf")
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    return float(median + z * 1.4826 * mad) if mad > 0 else float(median)


def sample_channel_quality(
    raw: mne.io.BaseRaw, picks: list[str], analysis_stop: int | None = None
) -> pd.DataFrame:
    """Estimate channel quality from fixed, behavior-blind recording snippets."""
    sfreq = float(raw.info["sfreq"])
    snippet_samples = int(round(20 * sfreq))
    stop = min(raw.n_times, analysis_stop or raw.n_times)
    starts = np.linspace(0, max(0, stop - snippet_samples), 10).astype(int)
    chunks = [raw.get_data(picks=picks, start=int(x), stop=int(x + snippet_samples)) for x in starts]
    data = np.concatenate(chunks, axis=1)
    finite_fraction = np.isfinite(data).mean(axis=1)
    sanitized = data.copy()
    sanitized[~np.isfinite(sanitized)] = np.nan
    channel_median = np.nanmedian(sanitized, axis=1, keepdims=True)
    sanitized = np.where(np.isfinite(sanitized), sanitized, channel_median)
    scale_uv = 1e6
    std_uv = np.nanstd(sanitized, axis=1) * scale_uv
    freq, psd = welch(sanitized, fs=sfreq, nperseg=int(2 * sfreq), axis=-1)
    line = psd[:, (freq >= 49) & (freq <= 51)].mean(axis=1)
    flank = psd[:, ((freq >= 45) & (freq <= 48)) | ((freq >= 52) & (freq <= 55))].mean(axis=1)
    line_ratio_db = 10 * np.log10(line / flank)
    log_std = np.log(np.maximum(std_uv, np.finfo(float).tiny))
    noisy_limit = robust_upper(log_std)
    line_limit = robust_upper(line_ratio_db)
    # A narrow 50-Hz peak is tracked but is not itself a bad-channel rule:
    # central electrodes can legitimately have a different line-noise profile,
    # and neither target metric (36/40 Hz) overlaps 50 Hz.
    bad = (
        (~np.isfinite(std_uv))
        | (finite_fraction < 0.95)
        | (std_uv < 0.1)
        | (log_std > noisy_limit)
    )
    line_warning = np.isfinite(line_ratio_db) & (line_ratio_db > line_limit)
    reason = np.select(
        [~np.isfinite(std_uv), finite_fraction < 0.95, std_uv < 0.1, log_std > noisy_limit],
        ["nonfinite", "low_finite_fraction", "flat", "extreme_variance"],
        default="",
    )
    return pd.DataFrame(
        {
            "channel": picks,
            "std_uv": std_uv,
            "finite_fraction": finite_fraction,
            "line_ratio_db": line_ratio_db,
            "line_noise_warning": line_warning,
            "bad": bad,
            "reason": reason,
        }
    )


def narrowband_log_ratio(
    epoch_uv: np.ndarray,
    sfreq: float,
    target_hz: float,
    baseline_slice: slice,
    late_slice: slice,
) -> tuple[float, float, float, complex]:
    """Return log amplitude change and a late-window complex target coefficient."""
    taps = firwin(
        117,
        [target_hz - 0.5, target_hz + 0.5],
        pass_zero=False,
        fs=sfreq,
        window="blackman",
    )
    filtered = filtfilt(taps, [1.0], epoch_uv, axis=-1)
    envelope = np.abs(hilbert(filtered, axis=-1))
    baseline = float(envelope[:, baseline_slice].mean())
    late = float(envelope[:, late_slice].mean())
    log_ratio = float(np.log(late / baseline)) if baseline > 0 and late > 0 else np.nan
    late_data = epoch_uv[:, late_slice]
    late_data = late_data - late_data.mean(axis=-1, keepdims=True)
    time = np.arange(late_data.shape[-1]) / sfreq
    coefficient = (late_data * np.exp(-2j * np.pi * target_hz * time)).mean(axis=-1).mean()
    return log_ratio, baseline, late, complex(coefficient)


def derive_subject(vhdr_path: Path, events_path: Path, output_dir: Path) -> dict[str, object]:
    subject = vhdr_path.name.split("_", 1)[0]
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="error")
    events = pd.read_csv(events_path, sep="\t")
    sfreq = float(raw.info["sfreq"])
    tmin, tmax = -1.5, 3.5
    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))
    analysis_stop = int(events["sample"].max()) + stop_offset
    eeg_picks = [name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"]
    # Some releases contain trailing bytes after the final planned epoch. They
    # are audited separately and must not contaminate behavior-linked epochs.
    channel_qc = sample_channel_quality(raw, eeg_picks, analysis_stop=analysis_stop)
    good_eeg = channel_qc.loc[~channel_qc.bad, "channel"].tolist()
    if len(good_eeg) < 50:
        raise RuntimeError(f"{subject}: fewer than 50 usable EEG channels ({len(good_eeg)})")

    missing_roi = sorted((set(CENTRAL_ROI) | set(OCCIPITAL_ROI)) - set(good_eeg))
    if missing_roi:
        raise RuntimeError(f"{subject}: predefined ROI channel(s) failed QC: {missing_roi}")

    zero_index = -start_offset
    baseline_slice = slice(zero_index + int(round(-0.7 * sfreq)), zero_index + int(round(-0.2 * sfreq)))
    late_slice = slice(zero_index + int(round(2.5 * sfreq)), zero_index + int(round(3.0 * sfreq)))

    epoch_cache: list[dict[str, object]] = []
    for trial_index, row in events.reset_index(drop=True).iterrows():
        sample = int(row["sample"])
        start, stop = sample + start_offset, sample + stop_offset
        record: dict[str, object] = {
            "subject": subject,
            "trial_index": trial_index,
            "sample": sample,
            "window_in_bounds": start >= 0 and stop <= raw.n_times,
        }
        if not record["window_in_bounds"]:
            epoch_cache.append(record)
            continue
        eeg = raw.get_data(picks=good_eeg, start=start, stop=stop) * 1e6
        finite_channels = np.isfinite(eeg).all(axis=1)
        finite_names = {name for name, keep in zip(good_eeg, finite_channels) if keep}
        roi_finite = (set(CENTRAL_ROI) | set(OCCIPITAL_ROI)).issubset(finite_names)
        enough_finite = int(finite_channels.sum()) >= 50
        if enough_finite:
            reference = eeg[finite_channels].mean(axis=0, keepdims=True)
            eeg[finite_channels] = eeg[finite_channels] - reference
            eeg[finite_channels] = eeg[finite_channels] - eeg[finite_channels].mean(
                axis=1, keepdims=True
            )
            p2p_99 = float(np.quantile(np.ptp(eeg[finite_channels], axis=1), 0.99))
        else:
            p2p_99 = np.nan
        eog_p2p = np.nan
        if "EOG" in raw.ch_names:
            # BrainVision labels this channel as misc with unit "n/a", so MNE
            # returns the numeric microvolt values without SI conversion.
            eog = raw.get_data(picks=["EOG"], start=start, stop=stop)[0]
            eog_p2p = float(np.ptp(eog - np.mean(eog)))
        record.update(
            {
                "eeg_p2p_99_uv": p2p_99,
                "eog_p2p_uv": eog_p2p,
                "n_nonfinite_eeg_channels": int((~finite_channels).sum()),
                "roi_finite": bool(roi_finite),
                "enough_finite_channels": bool(enough_finite),
                "eeg": eeg,
            }
        )
        epoch_cache.append(record)

    bounded = [x for x in epoch_cache if x["window_in_bounds"]]
    eeg_limit = min(500.0, robust_upper(np.array([x["eeg_p2p_99_uv"] for x in bounded])))
    eog_values = np.array([x["eog_p2p_uv"] for x in bounded], dtype=float)
    eog_limit = min(300.0, robust_upper(eog_values[np.isfinite(eog_values)]))

    trial_rows = []
    phase_rows = {36.0: [], 40.0: []}
    for record, (_, behavior) in zip(epoch_cache, events.reset_index(drop=True).iterrows()):
        result = {**behavior.to_dict(), **{k: v for k, v in record.items() if k != "eeg"}}
        if not record["window_in_bounds"]:
            result.update({"eeg_valid": False, "eeg_qc_flags": "window_out_of_bounds"})
            trial_rows.append(result)
            continue
        eeg_valid = (
            record["enough_finite_channels"]
            and record["roi_finite"]
            and np.isfinite(record["eeg_p2p_99_uv"])
            and record["eeg_p2p_99_uv"] <= eeg_limit
        )
        eog_clean = not np.isfinite(record["eog_p2p_uv"]) or record["eog_p2p_uv"] <= eog_limit
        reasons = []
        if record["eeg_p2p_99_uv"] > eeg_limit:
            reasons.append("extreme_eeg")
        if not record["enough_finite_channels"]:
            reasons.append("too_few_finite_channels")
        elif not record["roi_finite"]:
            reasons.append("nonfinite_roi")
        if np.isfinite(record["eog_p2p_uv"]) and record["eog_p2p_uv"] > eog_limit:
            reasons.append("extreme_eog")
        result.update(
            {
                "eeg_valid": bool(eeg_valid),
                "eog_clean": bool(eog_clean),
                "eeg_valid_strict_eog": bool(eeg_valid and eog_clean),
                "eeg_qc_flags": ";".join(reasons),
            }
        )
        if eeg_valid:
            eeg = record["eeg"]
            index = {name: idx for idx, name in enumerate(good_eeg)}
            for target_hz, roi in [(40.0, CENTRAL_ROI), (36.0, OCCIPITAL_ROI)]:
                roi_data = eeg[[index[name] for name in roi]]
                log_ratio, baseline, late, coefficient = narrowband_log_ratio(
                    roi_data, sfreq, target_hz, baseline_slice, late_slice
                )
                prefix = f"hz{int(target_hz)}"
                result[f"{prefix}_log_amp_ratio"] = log_ratio
                result[f"{prefix}_baseline_amp_uv"] = baseline
                result[f"{prefix}_late_amp_uv"] = late
                result[f"{prefix}_coefficient_real"] = coefficient.real
                result[f"{prefix}_coefficient_imag"] = coefficient.imag
                phase_rows[target_hz].append(coefficient)
        trial_rows.append(result)

    trial_table = pd.DataFrame(trial_rows)
    valid = trial_table.eeg_valid.fillna(False)
    summary: dict[str, object] = {
        "subject": subject,
        "sfreq": sfreq,
        "n_channels": len(raw.ch_names),
        "n_eeg_channels": len(eeg_picks),
        "n_bad_channels": int(channel_qc.bad.sum()),
        "bad_channels": channel_qc.loc[channel_qc.bad, "channel"].tolist(),
        "n_trials": int(len(trial_table)),
        "n_trials_in_bounds": int(trial_table.window_in_bounds.sum()),
        "n_trials_valid": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "n_trials_valid_strict_eog": int(
            trial_table.get("eeg_valid_strict_eog", pd.Series(False, index=trial_table.index))
            .fillna(False)
            .sum()
        ),
        "eeg_p2p_99_limit_uv": eeg_limit,
        "eog_p2p_limit_uv": eog_limit,
        "accuracy_all_trials": float(events.Correct.mean()),
        "median_correct_rt": float(events.loc[events.Correct.eq(1), "response_time"].median()),
        "input_vhdr": str(vhdr_path),
        "signal_bytes": vhdr_path.with_suffix(".eeg").stat().st_size,
        "signal_frame_remainder_bytes": vhdr_path.with_suffix(".eeg").stat().st_size
        % (len(raw.ch_names) * 4),
        "input_eeg_sha256": sha256_file(vhdr_path.with_suffix(".eeg")),
        "input_events_sha256": sha256_file(events_path),
    }
    for target_hz in (36.0, 40.0):
        prefix = f"hz{int(target_hz)}"
        if not valid.any():
            for suffix in ("coefficient_real", "coefficient_imag", "log_amp_ratio"):
                if f"{prefix}_{suffix}" not in trial_table:
                    trial_table[f"{prefix}_{suffix}"] = np.nan
            trial_table[f"{prefix}_coefficient_abs_uv"] = np.nan
            trial_table[f"{prefix}_phase_projected_uv"] = np.nan
            summary[f"hz{int(target_hz)}_itpc"] = np.nan
            summary[f"hz{int(target_hz)}_median_log_amp_ratio"] = np.nan
            continue
        coefficients_array = (
            trial_table.loc[valid, f"{prefix}_coefficient_real"].to_numpy()
            + 1j * trial_table.loc[valid, f"{prefix}_coefficient_imag"].to_numpy()
        )
        phase = coefficients_array / np.maximum(np.abs(coefficients_array), np.finfo(float).tiny)
        phase_sum = phase.sum()
        leave_one_out_reference = phase_sum - phase
        leave_one_out_reference /= np.maximum(
            np.abs(leave_one_out_reference), np.finfo(float).tiny
        )
        trial_table.loc[valid, f"{prefix}_coefficient_abs_uv"] = np.abs(coefficients_array)
        trial_table.loc[valid, f"{prefix}_phase_projected_uv"] = np.real(
            coefficients_array * np.conj(leave_one_out_reference)
        )
        summary[f"hz{int(target_hz)}_itpc"] = float(np.abs(phase.mean()))
        summary[f"hz{int(target_hz)}_median_log_amp_ratio"] = float(
            trial_table.loc[valid, f"hz{int(target_hz)}_log_amp_ratio"].median()
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    trial_table.drop(columns=["duration"], errors="ignore").to_csv(
        output_dir / f"{subject}_trial_features.csv", index=False
    )
    channel_qc.insert(0, "subject", subject)
    channel_qc.to_csv(output_dir / f"{subject}_channel_qc.csv", index=False)
    (output_dir / f"{subject}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vhdr", type=Path, required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/derived/ds007648_trial_features")
    )
    args = parser.parse_args()
    events = args.events or args.vhdr.with_name(args.vhdr.name.replace("_eeg.vhdr", "_events.tsv"))
    summary = derive_subject(args.vhdr, events, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
