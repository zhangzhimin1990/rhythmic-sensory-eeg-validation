#!/usr/bin/env python3
"""Summarize the five-person, result-blind Dryad R2b development calibration."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/qc/dryad_raw_stream/R2b_development_v1"
DEFAULT_OUTPUT = ROOT / "outputs/qc/dryad_raw_stream/R2b_development_audit_v1"
EXPECTED_SUBJECTS = (2, 13, 23, 27, 37)
Z_LIMITS = (4.0, 5.0, 6.0)
NEIGHBOUR_FLOORS = (0.0, 0.1, 0.2, 0.3)
BRIDGE_CEILING = 0.9995


def technical_flags(table: pd.DataFrame, z_limit: float, floor: float) -> pd.Series:
    return (
        table["finite_fraction"].lt(0.999)
        | table["rms_uv"].lt(0.1)
        | table["z_log_rms"].abs().gt(z_limit)
        | (table["z_line_noise_db"].gt(z_limit) & table["line_noise_db"].gt(3.0))
        | (
            table["z_high_frequency_db"].gt(z_limit)
            & table["high_frequency_db"].gt(-3.0)
        )
        | (
            table["z_low_neighbor_correlation"].gt(z_limit)
            & table["median_neighbor_correlation"].lt(floor)
        )
        | table["maximum_neighbor_correlation"].gt(BRIDGE_CEILING)
    )


def load_subject(input_root: Path, subject: int) -> tuple[dict, pd.DataFrame]:
    directory = input_root / f"S{subject}"
    with (directory / "technical_summary_r2b_development.json").open(
        encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    channels = pd.read_csv(directory / "channel_metrics_r2b_development.csv")
    if len(channels) != 64 or channels["channel"].nunique() != 64:
        raise ValueError(f"S{subject}: expected 64 unique channel rows")
    channels.insert(0, "subject", subject)
    return summary, channels


def run(input_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    participant_rows = []
    channel_tables = []
    reason_counter: Counter[str] = Counter()
    for subject in EXPECTED_SUBJECTS:
        summary, channels = load_subject(input_root, subject)
        participant_rows.append(
            {
                "subject": subject,
                "n_bad_channels": int(summary["n_bad_channels_development"]),
                "n_good_channels": int(summary["n_good_channels_development"]),
                "n_good_frontocentral_roi_channels": int(
                    summary["n_good_frontocentral_roi_channels"]
                ),
                "n_recorded_target_events": int(summary["n_recorded_target_events"]),
                "n_eeg_usable_event_sequences": int(
                    summary["n_eeg_usable_event_sequences"]
                ),
                "event_coverage_fraction": float(
                    summary["expected_event_coverage_fraction"]
                ),
                "n_technically_usable_epochs": int(
                    summary["n_technically_usable_epochs"]
                ),
                "overall_epoch_usable_fraction": float(
                    summary["overall_epoch_usable_fraction"]
                ),
                "minimum_event_cell_usable_fraction": float(
                    summary["minimum_event_cell_usable_fraction"]
                ),
                "minimum_analysis_window_pre_target_margin_ms": float(
                    summary["minimum_analysis_window_pre_target_margin_ms"]
                ),
                "technical_pass_candidate": bool(
                    summary["technical_pass_development_candidate"]
                ),
            }
        )
        flagged = channels.loc[channels["r2b_bad_channel_development"]].copy()
        for value in flagged["r2b_bad_channel_reason_development"].fillna(""):
            reason_counter.update(reason for reason in value.split(";") if reason)
        channel_tables.append(channels)

    participants = pd.DataFrame(participant_rows).sort_values("subject")
    all_channels = pd.concat(channel_tables, ignore_index=True)
    flagged_columns = [
        "subject",
        "channel",
        "rms_uv",
        "line_noise_db",
        "high_frequency_db",
        "median_neighbor_correlation",
        "maximum_neighbor_correlation",
        "z_log_rms",
        "z_line_noise_db",
        "z_high_frequency_db",
        "z_low_neighbor_correlation",
        "r2b_bad_channel_reason_development",
    ]
    flagged = all_channels.loc[
        all_channels["r2b_bad_channel_development"], flagged_columns
    ].sort_values(["subject", "channel"])
    reasons = pd.DataFrame(
        sorted(reason_counter.items()), columns=["technical_flag_reason", "channel_count"]
    )

    sensitivity_rows = []
    for z_limit in Z_LIMITS:
        for floor in NEIGHBOUR_FLOORS:
            per_subject = []
            for subject, table in all_channels.groupby("subject", sort=True):
                n_bad = int(technical_flags(table, z_limit, floor).sum())
                per_subject.append(n_bad)
                sensitivity_rows.append(
                    {
                        "robust_z_limit": z_limit,
                        "neighbour_correlation_floor": floor,
                        "subject": int(subject),
                        "n_bad_channels": n_bad,
                        "channel_count_gate_pass": bool(n_bad <= 12),
                    }
                )
            sensitivity_rows.append(
                {
                    "robust_z_limit": z_limit,
                    "neighbour_correlation_floor": floor,
                    "subject": "all_five",
                    "n_bad_channels": int(max(per_subject)),
                    "channel_count_gate_pass": bool(max(per_subject) <= 12),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    all_grid_pass = bool(
        sensitivity.loc[sensitivity["subject"].eq("all_five"), "channel_count_gate_pass"].all()
    )
    summary = {
        "development_subjects": list(EXPECTED_SUBJECTS),
        "n_development_subjects": len(EXPECTED_SUBJECTS),
        "n_technical_pass_candidate": int(participants["technical_pass_candidate"].sum()),
        "candidate_bad_channel_range": [
            int(participants["n_bad_channels"].min()),
            int(participants["n_bad_channels"].max()),
        ],
        "event_coverage_fraction_range": [
            float(participants["event_coverage_fraction"].min()),
            float(participants["event_coverage_fraction"].max()),
        ],
        "overall_epoch_usable_fraction_range": [
            float(participants["overall_epoch_usable_fraction"].min()),
            float(participants["overall_epoch_usable_fraction"].max()),
        ],
        "all_threshold_grid_channel_count_gates_pass": all_grid_pass,
        "threshold_grid": {
            "robust_z_limits": list(Z_LIMITS),
            "neighbour_correlation_floors": list(NEIGHBOUR_FLOORS),
        },
        "result_bearing_fields_emitted": False,
        "status": "development_audit_complete_freeze_decision_pending",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    participants.to_csv(output_dir / "participant_technical_summary.csv", index=False)
    flagged.to_csv(output_dir / "flagged_channel_details.csv", index=False)
    reasons.to_csv(output_dir / "technical_flag_reason_counts.csv", index=False)
    sensitivity.to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    (output_dir / "development_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.input_root, args.output_dir), ensure_ascii=False, indent=2))
