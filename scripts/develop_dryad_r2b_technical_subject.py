#!/usr/bin/env python3
"""Run the result-blind Dryad R2b technical-development pass for one BDF.

The script reconstructs only the event structure needed to locate stimulation
epochs, applies condition-blind channel and epoch artifact rules, and writes
aggregate technical outputs. It does not emit condition labels, responses,
reaction times, accuracy, target-frequency responses, or behavioural models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

try:
    from scripts.audit_dryad_bdf_pilot import (
        CONDITION_NAMES,
        EXTERNAL_CHANNELS,
        find_status_events,
        reconstruct_trials,
    )
    from scripts.develop_dryad_r2b_spatial_qc import (
        DEFAULT_ADJACENCY,
        require_technical_only_schema,
        r2b_channel_metrics,
        validate_adjacency,
    )
    from scripts.qc_dryad_signal_pilot import (
        ANALYSIS_WINDOW_END_SECONDS,
        ANALYSIS_WINDOW_START_SECONDS,
        FRONTOCENTRAL_ROI,
        preprocess_epoch,
    )
except ModuleNotFoundError:  # Direct execution via ``python scripts/...``.
    from audit_dryad_bdf_pilot import (
        CONDITION_NAMES,
        EXTERNAL_CHANNELS,
        find_status_events,
        reconstruct_trials,
    )
    from develop_dryad_r2b_spatial_qc import (
        DEFAULT_ADJACENCY,
        require_technical_only_schema,
        r2b_channel_metrics,
        validate_adjacency,
    )
    from qc_dryad_signal_pilot import (
        ANALYSIS_WINDOW_END_SECONDS,
        ANALYSIS_WINDOW_START_SECONDS,
        FRONTOCENTRAL_ROI,
        preprocess_epoch,
    )


EXPECTED_TRIALS = 384
MAXIMUM_BAD_CHANNELS = 12
MINIMUM_GOOD_CHANNELS = 52
MINIMUM_ROI_CHANNELS = 3
MINIMUM_EVENT_COVERAGE = 0.95
MINIMUM_OVERALL_USABLE = 0.90
MINIMUM_CELL_USABLE = 0.80
MINIMUM_PRE_TARGET_MARGIN_MS = 100.0
SCALP_FIXED_PEAK_TO_PEAK_UV = 250.0
ROI_FIXED_PEAK_TO_PEAK_UV = 150.0
ROBUST_MAD_MULTIPLIER = 6.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eeg_sequence_mask(trials: pd.DataFrame) -> pd.Series:
    """Identify trials with sufficient EEG event structure, independent of response."""
    known_conditions = set(CONDITION_NAMES.values())
    return (
        trials["condition"].isin(known_conditions)
        & trials["stimulation_sample"].notna()
        & trials["target_sample"].notna()
        & trials["stimulation_to_target_ms"].notna()
    )


def robust_upper(values: np.ndarray, multiplier: float = ROBUST_MAD_MULTIPLIER) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    return center + multiplier * mad


def epoch_artifact_table(
    raw: mne.io.BaseRaw,
    trials: pd.DataFrame,
    good_channels: list[str],
) -> pd.DataFrame:
    """Return an in-memory artifact table with no behavioural measurements."""
    roi = [channel for channel in FRONTOCENTRAL_ROI if channel in good_channels]
    if len(roi) < MINIMUM_ROI_CHANNELS:
        raise RuntimeError(f"fewer than three usable frontocentral ROI channels: {roi}")
    channel_index = {channel: index for index, channel in enumerate(good_channels)}
    roi_indices = [channel_index[channel] for channel in roi]
    sfreq_raw = float(raw.info["sfreq"])
    rows = []
    for trial in trials.loc[eeg_sequence_mask(trials)].itertuples(index=False):
        onset = int(trial.stimulation_sample)
        start = onset - int(round(3.0 * sfreq_raw))
        stop = onset + int(round(4.0 * sfreq_raw))
        if start < 0 or stop > raw.n_times:
            rows.append(
                {
                    "cell_code_internal": int(trial.condition_code),
                    "median_channel_peak_to_peak_uv": np.nan,
                    "roi_peak_to_peak_uv": np.nan,
                    "within_bounds": False,
                }
            )
            continue
        data = raw.get_data(picks=good_channels, start=start, stop=stop)
        filtered, sfreq = preprocess_epoch(data, sfreq_raw)
        analysis_start = int(round((3.0 + ANALYSIS_WINDOW_START_SECONDS) * sfreq))
        analysis_stop = int(round((3.0 + ANALYSIS_WINDOW_END_SECONDS) * sfreq))
        analysis = filtered[:, analysis_start:analysis_stop]
        roi_signal = np.nanmean(analysis[roi_indices], axis=0)
        rows.append(
            {
                "cell_code_internal": int(trial.condition_code),
                "median_channel_peak_to_peak_uv": float(
                    np.nanmedian(np.ptp(analysis, axis=1))
                ),
                "roi_peak_to_peak_uv": float(np.ptp(roi_signal)),
                "within_bounds": True,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result.assign(usable=pd.Series(dtype=bool))
    within = result["within_bounds"].astype(bool)
    scalp_limit = robust_upper(
        result.loc[within, "median_channel_peak_to_peak_uv"].to_numpy()
    )
    roi_limit = robust_upper(result.loc[within, "roi_peak_to_peak_uv"].to_numpy())
    fixed_bad = (
        (result["median_channel_peak_to_peak_uv"] > SCALP_FIXED_PEAK_TO_PEAK_UV)
        | (result["roi_peak_to_peak_uv"] > ROI_FIXED_PEAK_TO_PEAK_UV)
    )
    robust_bad = (
        (result["median_channel_peak_to_peak_uv"] > scalp_limit)
        | (result["roi_peak_to_peak_uv"] > roi_limit)
    )
    result["usable"] = within & ~(fixed_bad | robust_bad)
    return result


def aggregate_technical_summary(
    trials: pd.DataFrame,
    epoch_table: pd.DataFrame,
    channel_table: pd.DataFrame,
) -> dict[str, object]:
    """Aggregate participant-level technical fields and apply candidate gates."""
    sequence_mask = eeg_sequence_mask(trials)
    eeg_trials = trials.loc[sequence_mask]
    n_eeg_sequences = int(sequence_mask.sum())
    n_bad = int(channel_table["r2b_bad_channel_development"].sum())
    n_good = int(len(channel_table) - n_bad)
    good_names = set(
        channel_table.loc[
            ~channel_table["r2b_bad_channel_development"], "channel"
        ].astype(str)
    )
    n_roi = len(good_names.intersection(FRONTOCENTRAL_ROI))
    n_usable = int(epoch_table["usable"].sum()) if len(epoch_table) else 0
    overall_usable = n_usable / n_eeg_sequences if n_eeg_sequences else 0.0
    if len(epoch_table):
        cell_usable = epoch_table.groupby("cell_code_internal")["usable"].mean()
        minimum_cell_usable = float(cell_usable.min())
        observed_cells = int(len(cell_usable))
    else:
        minimum_cell_usable = 0.0
        observed_cells = 0
    minimum_stimulation_to_target_ms = (
        float(eeg_trials["stimulation_to_target_ms"].min())
        if len(eeg_trials)
        else np.nan
    )
    pre_target_margin_ms = (
        minimum_stimulation_to_target_ms - ANALYSIS_WINDOW_END_SECONDS * 1000.0
        if np.isfinite(minimum_stimulation_to_target_ms)
        else np.nan
    )
    components = {
        "channel_count_and_quality_gate": (
            len(channel_table) == 64
            and n_bad <= MAXIMUM_BAD_CHANNELS
            and n_good >= MINIMUM_GOOD_CHANNELS
            and n_roi >= MINIMUM_ROI_CHANNELS
        ),
        "eeg_event_coverage_gate": n_eeg_sequences / EXPECTED_TRIALS
        >= MINIMUM_EVENT_COVERAGE,
        "four_event_cells_observed_gate": observed_cells == 4,
        "overall_epoch_usability_gate": overall_usable >= MINIMUM_OVERALL_USABLE,
        "minimum_event_cell_usability_gate": minimum_cell_usable
        >= MINIMUM_CELL_USABLE,
        "analysis_window_margin_gate": bool(
            np.isfinite(pre_target_margin_ms)
            and pre_target_margin_ms >= MINIMUM_PRE_TARGET_MARGIN_MS
        ),
    }
    return {
        "n_scalp_channels": int(len(channel_table)),
        "n_bad_channels_development": n_bad,
        "n_good_channels_development": n_good,
        "n_good_frontocentral_roi_channels": n_roi,
        "n_recorded_target_events": int(len(trials)),
        "n_eeg_usable_event_sequences": n_eeg_sequences,
        "expected_event_coverage_fraction": n_eeg_sequences / EXPECTED_TRIALS,
        "n_technically_usable_epochs": n_usable,
        "overall_epoch_usable_fraction": overall_usable,
        "minimum_event_cell_usable_fraction": minimum_cell_usable,
        "n_event_cells_observed": observed_cells,
        "minimum_stimulation_to_target_ms": minimum_stimulation_to_target_ms,
        "minimum_analysis_window_pre_target_margin_ms": pre_target_margin_ms,
        **components,
        "technical_pass_development_candidate": bool(all(components.values())),
        "input_scope": "continuous_signal_and_event_timing_only",
        "result_bearing_fields_emitted": False,
        "status": "development_only_not_confirmatory_freeze",
    }


def run_subject(
    bdf_path: Path,
    output_dir: Path,
    adjacency_path: Path = DEFAULT_ADJACENCY,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    edges = pd.read_csv(adjacency_path)
    validate_adjacency(edges)
    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="error")
    scalp_names = [
        name for name in raw.ch_names if name not in EXTERNAL_CHANNELS | {"Status"}
    ]
    if len(scalp_names) != 64:
        raise RuntimeError(f"expected 64 scalp channels, found {len(scalp_names)}")
    events, _, _ = find_status_events(raw)
    trials = reconstruct_trials(events, float(raw.info["sfreq"]))
    channels = r2b_channel_metrics(raw, scalp_names, edges)
    good_channels = channels.loc[
        ~channels["r2b_bad_channel_development"], "channel"
    ].tolist()
    epochs = epoch_artifact_table(raw, trials, good_channels)
    summary = aggregate_technical_summary(trials, epochs, channels)
    summary.update(
        {
            "source_bdf_name": bdf_path.name,
            "source_bdf_size_bytes": bdf_path.stat().st_size,
            "source_bdf_sha256": sha256(bdf_path),
            "adjacency_file": adjacency_path.name,
        }
    )
    require_technical_only_schema(channels, summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    channels.to_csv(output_dir / "channel_metrics_r2b_development.csv", index=False)
    (output_dir / "technical_summary_r2b_development.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_subject(args.bdf, args.output_dir, args.adjacency)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
