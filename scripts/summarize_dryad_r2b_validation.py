#!/usr/bin/env python3
"""Summarize the completed frozen-protocol Dryad R2b technical validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = (
    ROOT
    / "outputs/qc/dryad_raw_stream/R2b_validation_gate_v1/r2b_validation_technical_gate.csv"
)
DEFAULT_OUTPUT = ROOT / "outputs/qc/dryad_raw_stream/R2b_validation_summary_v1"
GATE_COLUMNS = (
    "channel_count_and_quality_gate",
    "eeg_event_coverage_gate",
    "four_event_cells_observed_gate",
    "overall_epoch_usability_gate",
    "minimum_event_cell_usability_gate",
    "analysis_window_margin_gate",
)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("successes and total are inconsistent")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def run(gate_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    table = pd.read_csv(gate_path)
    if len(table) != 39 or table["subject"].nunique() != 39:
        raise ValueError("technical gate table must contain the exact 39-person validation set")
    passed = table["technical_pass_development_candidate"].astype(bool)
    n_pass = int(passed.sum())
    lower, upper = wilson_interval(n_pass, len(table))
    failure_counts = pd.DataFrame(
        [
            {
                "technical_gate": column,
                "n_failed": int((~table[column].astype(bool)).sum()),
            }
            for column in GATE_COLUMNS
        ]
    )
    failed_columns = ["subject", *GATE_COLUMNS]
    failed = table.loc[~passed, failed_columns].copy()
    failed["failed_technical_gates"] = failed.apply(
        lambda row: ";".join(column for column in GATE_COLUMNS if not bool(row[column])),
        axis=1,
    )
    failed = failed[["subject", "failed_technical_gates"]]
    participant = table.rename(
        columns={
            "technical_pass_development_candidate": "technical_pass_frozen_protocol"
        }
    )
    summary = {
        "n_validation_participants": 39,
        "n_technical_pass": n_pass,
        "n_technical_fail": int(len(table) - n_pass),
        "technical_pass_fraction": n_pass / len(table),
        "technical_pass_fraction_wilson_95_ci": [lower, upper],
        "required_technical_pass": 32,
        "technical_go": bool(n_pass >= 32),
        "failed_subjects": failed["subject"].astype(int).tolist(),
        "result_bearing_fields_emitted": False,
        "status": "frozen_protocol_technical_validation_complete",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    participant.to_csv(output_dir / "participant_technical_gate.csv", index=False)
    failure_counts.to_csv(output_dir / "failure_mechanism_counts.csv", index=False)
    failed.to_csv(output_dir / "failed_participants.csv", index=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.gate, args.output_dir), ensure_ascii=False, indent=2))
