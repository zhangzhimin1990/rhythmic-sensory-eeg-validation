#!/usr/bin/env python3
"""Classify the frozen Dryad primary ITPC result into a pre-result manuscript branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_ROOT = ROOT / "outputs/models/dryad_confirmatory_group_v1"
DEFAULT_AUDIT_ROOT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_v1"
DEFAULT_OUTPUT = ROOT / "outputs/status/dryad_confirmatory_primary_branch_v1"
PRIMARY_METRIC = "mean_itpc_in_stimulation_window"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def classify(primary: pd.Series, sensitivity: pd.Series) -> dict[str, object]:
    estimate = float(primary["mean_difference_theta_plus_minus_non_rhythmic"])
    p_value = float(primary["p_value_t_two_sided"])
    equivalent = _as_bool(primary["equivalent_within_plus_minus_dz_0_3"])
    sensitivity_estimate = float(
        sensitivity["mean_difference_theta_plus_minus_non_rhythmic"]
    )
    significant = p_value < 0.05
    if significant and estimate > 0:
        branch = "theta_plus_higher"
        allowed_core_language = (
            "theta-plus showed a higher primary stimulus-locked ITPC estimate than "
            "the non-rhythmic active comparator"
        )
    elif significant and estimate < 0:
        branch = "theta_plus_lower_opposite_direction"
        allowed_core_language = (
            "the primary ITPC contrast was statistically different from zero in the "
            "opposite direction: theta-plus was lower than the non-rhythmic comparator"
        )
    elif equivalent:
        branch = "operationally_equivalent"
        allowed_core_language = (
            "the 90% interval fell within the prespecified measurement-oriented "
            "plus-or-minus 0.30 dz bounds"
        )
    else:
        branch = "inconclusive_not_equivalent"
        allowed_core_language = (
            "the primary ITPC contrast was not statistically different from zero and "
            "did not satisfy the prespecified equivalence criterion"
        )
    sensitivity_direction_consistent = bool(
        estimate == 0
        or sensitivity_estimate == 0
        or (estimate > 0) == (sensitivity_estimate > 0)
    )
    superiority_claim_unlocked = bool(
        branch == "theta_plus_higher" and sensitivity_direction_consistent
    )
    return {
        "status": "confirmatory_primary_branch_classified",
        "primary_metric": PRIMARY_METRIC,
        "branch": branch,
        "primary_estimate_theta_plus_minus_non_rhythmic": estimate,
        "primary_p_value_two_sided": p_value,
        "equivalent_within_plus_minus_dz_0_3": equivalent,
        "roi_sensitivity_estimate": sensitivity_estimate,
        "roi_sensitivity_direction_consistent": sensitivity_direction_consistent,
        "theta_plus_superiority_claim_unlocked": superiority_claim_unlocked,
        "allowed_core_language": allowed_core_language,
        "forbidden_upgrades": [
            "endogenous_entrainment_mechanism",
            "neural_mediation",
            "chronic_cognitive_efficacy",
            "disease_modification",
        ],
    }


def run(group_root: Path, audit_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    neural = pd.read_csv(group_root / "paired_neural_effects.csv")
    rows = neural.loc[neural["metric"].eq(PRIMARY_METRIC)]
    if len(rows) != 1:
        raise ValueError("expected exactly one frozen primary ITPC row")
    sensitivity = pd.read_csv(audit_root / "primary_roi_channel_sensitivity.csv")
    if len(sensitivity) != 1 or sensitivity.iloc[0]["metric"] != PRIMARY_METRIC:
        raise ValueError("expected exactly one primary ROI sensitivity row")
    result = classify(rows.iloc[0], sensitivity.iloc[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "primary_branch_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-root", type=Path, default=DEFAULT_GROUP_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.group_root, args.audit_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
