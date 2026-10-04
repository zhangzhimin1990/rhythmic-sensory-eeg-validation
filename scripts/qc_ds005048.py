#!/usr/bin/env python3
"""Foundation-gate QC for a small ds005048 auditory 40 Hz sample.

This is a data-integrity and physiological-signal check, not a confirmatory
group analysis. It validates BIDS/EEGLAB consistency, computes subject-level
40 Hz response summaries, and writes auditable tables and figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr


EXPECTED_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]
FRONTOCENTRAL_ROI = ["Fz", "Cz", "C3", "C4"]
TARGET_HZ = 40.0
WELCH_WINDOW_S = 4.0
WELCH_OVERLAP = 0.5
NARROW_NOISE_BANDS = ((38.0, 39.5), (40.5, 42.0))
BROAD_NOISE_BANDS = ((35.0, 39.0), (41.0, 45.0))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_participants(dataset_root: Path) -> pd.DataFrame:
    participants = pd.read_csv(dataset_root / "participants.tsv", sep="\t", dtype=str)
    participants = participants.rename(
        columns={"Gender": "sex", "Age": "age", "Group": "group", "MMSE": "mmse"}
    )
    participants["age"] = pd.to_numeric(participants["age"], errors="coerce")
    participants["mmse"] = pd.to_numeric(participants["mmse"], errors="coerce")
    return participants


def event_path_for(set_path: Path) -> Path:
    return set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv"))


def sidecar_path_for(set_path: Path) -> Path:
    return set_path.with_name(set_path.name.replace("_eeg.set", "_eeg.json"))


def channel_path_for(set_path: Path) -> Path:
    return set_path.with_name(set_path.name.replace("_eeg.set", "_channels.tsv"))


def fdt_path_for(set_path: Path) -> Path:
    return set_path.with_suffix(".fdt")


def parse_subject(set_path: Path) -> str:
    return set_path.parts[-3]


def read_raw(set_path: Path, preload: bool = True) -> mne.io.BaseRaw:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Complex objects .* are not supported", category=UserWarning
        )
        raw = mne.io.read_raw_eeglab(set_path, preload=preload, verbose="ERROR")
    return raw


def normalize_marker(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def validate_annotations(raw: mne.io.BaseRaw, events: pd.DataFrame) -> dict[str, Any]:
    tsv = [
        (float(row.onset), normalize_marker(row.value))
        for row in events.itertuples(index=False)
    ]
    annotations = [
        (float(onset), normalize_marker(description))
        for onset, description in zip(raw.annotations.onset, raw.annotations.description)
    ]
    n_compared = min(len(tsv), len(annotations))
    onset_error = [abs(tsv[i][0] - annotations[i][0]) for i in range(n_compared)]
    values_match = all(tsv[i][1] == annotations[i][1] for i in range(n_compared))
    max_error = max(onset_error, default=np.nan)
    tolerance = 1.1 / float(raw.info["sfreq"])
    return {
        "n_events_tsv": len(tsv),
        "n_annotations_set": len(annotations),
        "event_values_match": bool(values_match and len(tsv) == len(annotations)),
        "max_event_onset_error_s": max_error,
        "event_onsets_within_one_sample": bool(max_error <= tolerance),
    }


def psd_for_segment(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(round(WELCH_WINDOW_S * sfreq))
    noverlap = int(round(nperseg * WELCH_OVERLAP))
    if data.shape[-1] < nperseg:
        raise ValueError(f"Segment has {data.shape[-1]} samples; need at least {nperseg}.")
    frequencies, psd = welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        axis=-1,
        scaling="density",
    )
    return frequencies, psd


def frequency_bin(frequencies: np.ndarray, target_hz: float) -> int:
    index = int(np.argmin(np.abs(frequencies - target_hz)))
    if abs(float(frequencies[index]) - target_hz) > 1e-9:
        raise ValueError(f"Target {target_hz} Hz is not represented exactly in Welch bins.")
    return index


def noise_mask(
    frequencies: np.ndarray, bands: tuple[tuple[float, float], ...]
) -> np.ndarray:
    mask = np.zeros(frequencies.shape, dtype=bool)
    for low, high in bands:
        mask |= (frequencies >= low) & (frequencies <= high)
    return mask


def snr_db(
    psd: np.ndarray,
    frequencies: np.ndarray,
    target_hz: float,
    bands: tuple[tuple[float, float], ...],
) -> np.ndarray:
    target_index = frequency_bin(frequencies, target_hz)
    mask = noise_mask(frequencies, bands)
    numerator = psd[..., target_index]
    denominator = np.mean(psd[..., mask], axis=-1)
    return 10.0 * np.log10(numerator / denominator)


def channel_psd_rows(
    raw: mne.io.BaseRaw,
    events: pd.DataFrame,
    subject: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sfreq = float(raw.info["sfreq"])
    block_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    condition_counts = {1: 0, 2: 0}
    for row in events.itertuples(index=False):
        marker = normalize_marker(row.value)
        if marker not in (1, 2):
            continue
        duration_s = float(row.duration)
        # Exclude the final approximately 5 s rest tail: one Welch window is
        # much less stable than the standard 20 s inter-block rest. Stimulus
        # blocks must still clear the 4 s Welch minimum.
        if (marker == 1 and duration_s < 19.0) or duration_s < WELCH_WINDOW_S:
            continue
        condition_counts[marker] += 1
        condition = "stimulus" if marker == 2 else "rest"
        block_index = condition_counts[marker]
        start = int(round(float(row.onset) * sfreq))
        stop = min(raw.n_times, int(round((float(row.onset) + duration_s) * sfreq)))
        segment = raw.get_data(start=start, stop=stop)
        frequencies, psd = psd_for_segment(segment, sfreq)
        target_index = frequency_bin(frequencies, TARGET_HZ)
        narrow = snr_db(psd, frequencies, TARGET_HZ, NARROW_NOISE_BANDS)
        broad = snr_db(psd, frequencies, TARGET_HZ, BROAD_NOISE_BANDS)

        for channel_index, channel in enumerate(raw.ch_names):
            block_rows.append(
                {
                    "participant_id": subject,
                    "condition": condition,
                    "block_index": block_index,
                    "onset_s": float(row.onset),
                    "duration_s": duration_s,
                    "channel": channel,
                    "target_power_v2_hz": float(psd[channel_index, target_index]),
                    "target_power_db_v2_hz": float(
                        10.0 * np.log10(psd[channel_index, target_index])
                    ),
                    "snr_narrow_db": float(narrow[channel_index]),
                    "snr_broad_db": float(broad[channel_index]),
                }
            )
            band = (frequencies >= 25.0) & (frequencies <= 55.0)
            curve_rows.extend(
                {
                    "participant_id": subject,
                    "condition": condition,
                    "block_index": block_index,
                    "channel": channel,
                    "frequency_hz": float(frequency),
                    "psd_v2_hz": float(value),
                }
                for frequency, value in zip(frequencies[band], psd[channel_index, band])
            )
    return pd.DataFrame(block_rows), pd.DataFrame(curve_rows)


def summarize_subjects(blocks: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    roi_blocks = blocks[blocks["channel"].isin(FRONTOCENTRAL_ROI)].copy()
    roi_by_block = (
        roi_blocks.groupby(["participant_id", "condition", "block_index"], as_index=False)
        .agg(
            snr_narrow_db=("snr_narrow_db", "mean"),
            snr_broad_db=("snr_broad_db", "mean"),
            target_power_db_v2_hz=("target_power_db_v2_hz", "mean"),
        )
    )
    # The first six stimulus blocks are the future full-cohort primary scope.
    # Up to six full inter-block rests are comparators. Short sessions supply
    # five full 20 s rests because their final 5 s tail is excluded.
    primary = roi_by_block[roi_by_block["block_index"] <= 6]
    wide = (
        primary.groupby(["participant_id", "condition"], as_index=False)
        .agg(
            snr_narrow_db=("snr_narrow_db", "median"),
            snr_broad_db=("snr_broad_db", "median"),
            target_power_db_v2_hz=("target_power_db_v2_hz", "median"),
        )
        .pivot(index="participant_id", columns="condition")
    )
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    wide = wide.reset_index()
    for metric in ("snr_narrow_db", "snr_broad_db", "target_power_db_v2_hz"):
        wide[f"{metric}_stim_minus_rest"] = (
            wide[f"{metric}_stimulus"] - wide[f"{metric}_rest"]
        )
    keep = ["participant_id", "sex", "age", "group", "mmse"]
    return participants[keep].merge(wide, on="participant_id", how="right", validate="one_to_one")


def plot_paired_snr(subjects: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 5.2))
    n_subjects = len(subjects)
    for row in subjects.itertuples(index=False):
        axis.plot(
            [0, 1],
            [row.snr_narrow_db_rest, row.snr_narrow_db_stimulus],
            marker="o",
            color=None if n_subjects <= 12 else "#777777",
            linewidth=1.6 if n_subjects <= 12 else 0.8,
            alpha=0.85 if n_subjects <= 12 else 0.32,
            label=row.participant_id if n_subjects <= 12 else None,
        )
    if n_subjects > 12:
        medians = [subjects["snr_narrow_db_rest"].median(), subjects["snr_narrow_db_stimulus"].median()]
        axis.plot([0, 1], medians, color="#D62728", marker="D", linewidth=2.8, label="Median")
    axis.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    axis.set_xticks([0, 1], ["Rest", "40 Hz auditory stimulus"])
    axis.set_ylabel("40 Hz SNR (dB), frontocentral ROI")
    axis.set_title(f"Subject-level 40 Hz response check (first six blocks, n={n_subjects})")
    axis.grid(axis="y", alpha=0.25)
    if n_subjects <= 12:
        axis.legend(title="Participant", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def subject_condition_curves(curves: pd.DataFrame) -> pd.DataFrame:
    roi = curves[curves["channel"].isin(FRONTOCENTRAL_ROI)].copy()
    roi = roi[roi["block_index"] <= 6]
    return (
        roi.groupby(["participant_id", "condition", "frequency_hz"], as_index=False)
        .agg(psd_v2_hz=("psd_v2_hz", "median"))
    )


def plot_psd_curves(curves: pd.DataFrame, output_path: Path) -> None:
    summarized = subject_condition_curves(curves)
    n_subjects = summarized["participant_id"].nunique()
    group = (
        summarized.groupby(["condition", "frequency_hz"])["psd_v2_hz"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(7.8, 4.9))
    style = {
        "rest": ("#4C78A8", "Rest"),
        "stimulus": ("#E45756", "40 Hz stimulus"),
    }
    for condition, (color, label) in style.items():
        frame = group[group["condition"] == condition]
        frequency = frame["frequency_hz"].to_numpy(float)
        median = 10.0 * np.log10(frame["median"].to_numpy(float))
        q25 = 10.0 * np.log10(frame["q25"].to_numpy(float))
        q75 = 10.0 * np.log10(frame["q75"].to_numpy(float))
        axis.plot(frequency, median, color=color, linewidth=2, label=label)
        axis.fill_between(frequency, q25, q75, color=color, alpha=0.18)
    axis.axvline(40, color="#222222", linewidth=1, linestyle="--", label="Target: 40 Hz")
    axis.set_xlim(30, 50)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (dB V²/Hz), median and IQR across subjects")
    axis.set_title(f"Frontocentral spectrum (n={n_subjects})")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_block_trajectory(blocks: pd.DataFrame, output_path: Path) -> None:
    roi = blocks[
        (blocks["channel"].isin(FRONTOCENTRAL_ROI)) & (blocks["condition"] == "stimulus")
    ]
    trajectory = (
        roi.groupby(["participant_id", "block_index"], as_index=False)
        .agg(snr_narrow_db=("snr_narrow_db", "mean"))
    )
    n_subjects = trajectory["participant_id"].nunique()
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    if n_subjects <= 12:
        for participant, frame in trajectory.groupby("participant_id"):
            axis.plot(
                frame["block_index"], frame["snr_narrow_db"], marker="o", linewidth=1.3,
                alpha=0.8, label=participant,
            )
    else:
        for _, frame in trajectory.groupby("participant_id"):
            axis.plot(
                frame["block_index"], frame["snr_narrow_db"], color="#888888",
                linewidth=0.6, alpha=0.18,
            )
        aggregate = (
            trajectory.groupby("block_index")["snr_narrow_db"]
            .agg(
                median="median",
                q25=lambda values: values.quantile(0.25),
                q75=lambda values: values.quantile(0.75),
            )
            .reset_index()
        )
        axis.fill_between(
            aggregate["block_index"], aggregate["q25"], aggregate["q75"],
            color="#4C78A8", alpha=0.22, label="IQR",
        )
        axis.plot(
            aggregate["block_index"], aggregate["median"], color="#1F4E79",
            marker="o", linewidth=2.4, label="Median",
        )
    axis.axvspan(6.5, 10.5, color="#BBBBBB", alpha=0.15, label="Long-session-only scope")
    axis.set_xticks(range(1, 11))
    axis.set_xlabel("Stimulus block")
    axis.set_ylabel("40 Hz SNR (dB), frontocentral ROI")
    axis.set_title(f"Block-to-block 40 Hz SNR (n={n_subjects}; blocks 7–10 long-session only)")
    axis.grid(alpha=0.2)
    if n_subjects <= 12:
        axis.legend(title="Participant", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    else:
        axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize_channels(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = blocks[blocks["block_index"] <= 6]
    condition = (
        primary.groupby(["participant_id", "channel", "condition"], as_index=False)
        .agg(snr_narrow_db=("snr_narrow_db", "median"))
        .pivot(index=["participant_id", "channel"], columns="condition", values="snr_narrow_db")
        .reset_index()
    )
    condition["snr_delta_db"] = condition["stimulus"] - condition["rest"]
    summary = (
        condition.groupby("channel", as_index=False)["snr_delta_db"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
    )
    order = {channel: index for index, channel in enumerate(EXPECTED_CHANNELS)}
    summary["channel_order"] = summary["channel"].map(order)
    summary = summary.sort_values("channel_order").drop(columns="channel_order")
    return condition, summary


def split_half_stability(blocks: pd.DataFrame) -> dict[str, float]:
    roi = blocks[
        (blocks["channel"].isin(FRONTOCENTRAL_ROI)) & (blocks["condition"] == "stimulus")
    ]
    block_level = (
        roi.groupby(["participant_id", "block_index"], as_index=False)
        .agg(
            snr_narrow_db=("snr_narrow_db", "mean"),
            snr_broad_db=("snr_broad_db", "mean"),
            target_power_db_v2_hz=("target_power_db_v2_hz", "mean"),
        )
    )
    result: dict[str, float] = {}
    for metric in ("snr_narrow_db", "snr_broad_db", "target_power_db_v2_hz"):
        first = (
            block_level[block_level["block_index"] <= 3]
            .groupby("participant_id")[metric]
            .mean()
        )
        second = (
            block_level[
                (block_level["block_index"] >= 4) & (block_level["block_index"] <= 6)
            ]
            .groupby("participant_id")[metric]
            .mean()
        )
        aligned = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
        rho = spearmanr(aligned["first"], aligned["second"]).statistic
        result[f"split_half_spearman_{metric}"] = float(rho)
    return result


def plot_channel_response(blocks: pd.DataFrame, output_path: Path) -> None:
    subject_channels, summary = summarize_channels(blocks)
    n_subjects = subject_channels["participant_id"].nunique()
    colors = ["#E45756" if channel in FRONTOCENTRAL_ROI else "#4C78A8" for channel in summary["channel"]]
    x = np.arange(len(summary))
    median = summary["median"].to_numpy(float)
    lower = median - summary["q25"].to_numpy(float)
    upper = summary["q75"].to_numpy(float) - median
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar(x, median, color=colors, alpha=0.9)
    axis.errorbar(x, median, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=2)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks(x, summary["channel"], rotation=45, ha="right")
    axis.set_ylabel("Stimulus − rest 40 Hz SNR (dB)")
    axis.set_title(f"Scalp-channel response pattern (median and IQR, n={n_subjects})")
    axis.grid(axis="y", alpha=0.2)
    axis.text(
        0.99, 0.98, "Red: prespecified frontocentral ROI", transform=axis.transAxes,
        ha="right", va="top", fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze(dataset_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root = output_root / "figures"
    figure_root.mkdir(exist_ok=True)
    participants = read_participants(dataset_root)
    set_files = sorted(dataset_root.glob("sub-*/eeg/*_eeg.set"))
    if not set_files:
        raise FileNotFoundError(f"No EEGLAB .set files found under {dataset_root}")

    qc_rows: list[dict[str, Any]] = []
    block_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    for set_path in set_files:
        subject = parse_subject(set_path)
        fdt_path = fdt_path_for(set_path)
        event_path = event_path_for(set_path)
        sidecar_path = sidecar_path_for(set_path)
        channel_path = channel_path_for(set_path)
        required = [set_path, fdt_path, event_path, sidecar_path, channel_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{subject} is missing required files: {missing}")

        for path in (set_path, fdt_path):
            manifest_rows.append(
                {
                    "participant_id": subject,
                    "file": str(path.relative_to(dataset_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

        events = pd.read_csv(event_path, sep="\t")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        channels = pd.read_csv(channel_path, sep="\t")
        raw = read_raw(set_path, preload=True)
        annotation_check = validate_annotations(raw, events)
        data = raw.get_data()
        raw_duration = raw.n_times / float(raw.info["sfreq"])
        sidecar_duration = float(sidecar["RecordingDuration"])
        channel_names_tsv = channels["name"].astype(str).tolist()
        qc_rows.append(
            {
                "participant_id": subject,
                "sfreq_hz": float(raw.info["sfreq"]),
                "n_channels": len(raw.ch_names),
                "n_times": raw.n_times,
                "raw_duration_s": raw_duration,
                "sidecar_duration_s": sidecar_duration,
                "duration_error_s": abs(raw_duration - sidecar_duration),
                "channels_match_expected": raw.ch_names == EXPECTED_CHANNELS,
                "channels_match_tsv": raw.ch_names == channel_names_tsv,
                "finite_fraction": float(np.isfinite(data).mean()),
                "median_abs_amplitude_uv": float(np.median(np.abs(data)) * 1e6),
                "p99_abs_amplitude_uv": float(np.quantile(np.abs(data), 0.99) * 1e6),
                "n_stimulus_blocks": int((events["value"].map(normalize_marker) == 2).sum()),
                "n_rest_blocks": int((events["value"].map(normalize_marker) == 1).sum()),
                **annotation_check,
            }
        )
        blocks, curves = channel_psd_rows(raw, events, subject)
        block_frames.append(blocks)
        curve_frames.append(curves)

    qc = pd.DataFrame(qc_rows)
    blocks = pd.concat(block_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    subjects = summarize_subjects(blocks, participants)
    channel_subjects, channel_summary = summarize_channels(blocks)
    manifest = pd.DataFrame(manifest_rows)

    integrity_columns = [
        "channels_match_expected", "channels_match_tsv", "event_values_match",
        "event_onsets_within_one_sample",
    ]
    integrity_pass = bool(
        qc[integrity_columns].all(axis=None)
        and (qc["sfreq_hz"] == 250.0).all()
        and (qc["n_channels"] == 19).all()
        and (qc["finite_fraction"] == 1.0).all()
        and (qc["duration_error_s"] <= 1.0 / 250.0 + 1e-9).all()
    )
    deltas = subjects["snr_narrow_db_stim_minus_rest"].dropna()
    broad_deltas = subjects["snr_broad_db_stim_minus_rest"].dropna()
    power_deltas = subjects["target_power_db_v2_hz_stim_minus_rest"].dropna()
    n_positive = int((deltas > 0).sum())
    n_positive_broad = int((broad_deltas > 0).sum())
    n_positive_power = int((power_deltas > 0).sum())
    n_subjects = int(len(subjects))
    response_summary = {
        "n_subjects": n_subjects,
        "n_short_sessions": int((qc["n_stimulus_blocks"] == 6).sum()),
        "n_long_sessions": int((qc["n_stimulus_blocks"] == 10).sum()),
        "n_with_mmse": int(subjects["mmse"].notna().sum()),
        "n_positive_snr_delta": n_positive,
        "median_snr_delta_db": float(deltas.median()),
        "min_snr_delta_db": float(deltas.min()),
        "max_snr_delta_db": float(deltas.max()),
        "n_positive_broad_snr_delta": n_positive_broad,
        "median_broad_snr_delta_db": float(broad_deltas.median()),
        "n_positive_target_power_delta": n_positive_power,
        "median_target_power_delta_db": float(power_deltas.median()),
        "provisional_signal_gate": (
            "supportive"
            if (
                n_positive / n_subjects >= 0.75
                and n_positive_broad / n_subjects >= 0.75
                and float(deltas.median()) > 0
                and float(broad_deltas.median()) > 0
            )
            else "inconclusive"
        ),
        "interpretation_limit": (
            "Foundation-gate analysis; clinical associations require prespecified modeling, "
            "and no result supports an efficacy claim."
        ),
        **split_half_stability(blocks),
    }
    run_summary = {
        "dataset": "OpenNeuro ds005048 v1.0.1",
        "source": "https://openneuro.org/datasets/ds005048/versions/1.0.1",
        "analysis_scope": (
            "Six long-session records selected before signal inspection"
            if n_subjects == 6
            else f"Full available ds005048 cohort foundation analysis (n={n_subjects})"
        ),
        "frontocentral_roi": FRONTOCENTRAL_ROI,
        "primary_blocks": (
            "first six stimulus blocks and up to first six full 20-second rest blocks; "
            "short sessions contribute five full rest blocks"
        ),
        "welch_window_s": WELCH_WINDOW_S,
        "welch_overlap": WELCH_OVERLAP,
        "target_hz": TARGET_HZ,
        "integrity_gate_pass": integrity_pass,
        **response_summary,
        "software": {
            "mne": mne.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    qc.to_csv(output_root / "recording_qc.csv", index=False)
    blocks.to_csv(output_root / "block_channel_metrics.csv", index=False)
    subjects.to_csv(output_root / "subject_summary.csv", index=False)
    channel_subjects.to_csv(output_root / "subject_channel_deltas.csv", index=False)
    channel_summary.to_csv(output_root / "channel_response_summary.csv", index=False)
    manifest.to_csv(output_root / "file_manifest_sha256.csv", index=False)
    subject_condition_curves(curves).to_csv(
        output_root / "subject_condition_psd.csv", index=False
    )
    (output_root / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_paired_snr(subjects, figure_root / "01_paired_40hz_snr.png")
    plot_psd_curves(curves, figure_root / "02_frontocentral_psd.png")
    plot_block_trajectory(blocks, figure_root / "03_block_trajectory.png")
    plot_channel_response(blocks, figure_root / "04_channel_response.png")
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    summary = analyze(args.dataset_root.resolve(), args.output_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
