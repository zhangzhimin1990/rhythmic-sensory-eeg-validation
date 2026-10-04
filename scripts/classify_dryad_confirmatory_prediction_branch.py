#!/usr/bin/env python3
"""Classify confirmatory prediction increment without overstating external validity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_ROOT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_v1"
DEFAULT_OUTPUT = ROOT / "outputs/status/dryad_confirmatory_prediction_branch_v1"
COMPARATORS = {"training_fold_mean", "baseline"}


def classify(increments: pd.DataFrame) -> dict[str, object]:
    if set(increments["comparator"]) != COMPARATORS or len(increments) != 2:
        raise ValueError("prediction increments must contain exactly the two frozen comparators")
    rows = increments.set_index("comparator")
    decisions = []
    for comparator in ("training_fold_mean", "baseline"):
        row = rows.loc[comparator]
        mae_supported = float(row["mae_improvement_bootstrap_ci95_low"]) > 0
        rmse_supported = float(row["rmse_improvement_comparator_minus_neural"]) > 0
        decisions.append({
            "comparator": comparator,
            "mae_improvement": float(row["mae_improvement_comparator_minus_neural"]),
            "mae_ci95_low": float(row["mae_improvement_bootstrap_ci95_low"]),
            "mae_ci95_high": float(row["mae_improvement_bootstrap_ci95_high"]),
            "rmse_improvement": float(row["rmse_improvement_comparator_minus_neural"]),
            "mae_interval_supports_improvement": mae_supported,
            "rmse_point_supports_improvement": rmse_supported,
            "comparator_gate_passed": bool(mae_supported and rmse_supported),
        })
    n_passed = sum(item["comparator_gate_passed"] for item in decisions)
    if n_passed == 2:
        branch = "consistent_internal_out_of_sample_increment"
        allowed = (
            "neural features showed exploratory incremental prediction across both "
            "prespecified internal comparators in nested leave-one-participant-out evaluation"
        )
    elif n_passed == 1:
        branch = "comparator_dependent_prediction_increment"
        allowed = (
            "prediction evidence depended on the comparator and did not satisfy the "
            "prespecified consistent-increment gate"
        )
    else:
        branch = "no_consistent_prediction_increment"
        allowed = "neural features did not show consistent incremental prediction in this sample"
    return {
        "status": "confirmatory_prediction_branch_classified",
        "branch": branch,
        "comparator_decisions": decisions,
        "prediction_increment_claim_unlocked": n_passed == 2,
        "evaluation_design": "nested_leave_one_participant_out_internal_validation",
        "independent_external_validation": False,
        "uncertainty_method": "paired_participant_bootstrap_of_fixed_out_of_fold_prediction_errors",
        "full_pipeline_retrained_within_each_bootstrap": False,
        "allowed_core_language": allowed,
        "forbidden_upgrades": [
            "external validation",
            "deployable treatment-selection model",
            "clinical utility",
            "individual benefit is intrinsically unpredictable",
        ],
    }


def run(audit_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    result = classify(pd.read_csv(audit_root / "prediction_increment_bootstrap.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction_branch_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.audit_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
