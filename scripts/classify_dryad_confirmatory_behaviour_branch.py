#!/usr/bin/env python3
"""Classify the frozen Dryad active-control behavioural result conservatively."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_ROOT = ROOT / "outputs/models/dryad_confirmatory_group_v1"
DEFAULT_OUTPUT = ROOT / "outputs/status/dryad_confirmatory_behaviour_branch_v1"
EXPECTED = {"accuracy", "median_correct_rt_ms", "inverse_efficiency_ms"}


def classify(effects: pd.DataFrame) -> dict[str, object]:
    if set(effects["metric"]) != EXPECTED or len(effects) != 3:
        raise ValueError("behaviour effects must contain exactly the three frozen endpoints")
    rows = effects.set_index("metric")
    accuracy = rows.loc["accuracy"]
    rt = rows.loc["median_correct_rt_ms"]
    ie = rows.loc["inverse_efficiency_ms"]

    def estimate(row: pd.Series) -> float:
        return float(row["mean_difference_theta_plus_minus_non_rhythmic"])

    def adjusted_significant(row: pd.Series) -> bool:
        return float(row["p_value_holm_behaviour"]) < 0.05

    accuracy_estimate = estimate(accuracy)
    ie_estimate = estimate(ie)
    ie_favourable = ie_estimate < 0 and adjusted_significant(ie)
    ie_unfavourable = ie_estimate > 0 and adjusted_significant(ie)
    accuracy_decrement_detected = accuracy_estimate < 0 and adjusted_significant(accuracy)
    accuracy_increment_detected = accuracy_estimate > 0 and adjusted_significant(accuracy)

    if ie_favourable and accuracy_decrement_detected:
        branch = "inverse_efficiency_advantage_with_accuracy_decrement"
        allowed = (
            "inverse efficiency favoured theta-plus, but a Holm-corrected accuracy "
            "decrement was also detected; a speed-accuracy trade-off cannot be dismissed"
        )
    elif ie_favourable:
        branch = "inverse_efficiency_advantage_no_detected_accuracy_decrement"
        allowed = (
            "inverse efficiency favoured theta-plus and no Holm-corrected accuracy "
            "decrement was detected; accuracy non-inferiority was not established"
        )
    elif ie_unfavourable:
        branch = "inverse_efficiency_disadvantage"
        allowed = "inverse efficiency was worse under theta-plus than under the active comparator"
    else:
        branch = "no_clear_inverse_efficiency_advantage"
        allowed = "there was no clear active-control advantage in inverse efficiency"

    return {
        "status": "confirmatory_behaviour_branch_classified",
        "branch": branch,
        "inverse_efficiency_estimate_theta_plus_minus_non_rhythmic": ie_estimate,
        "inverse_efficiency_holm_p": float(ie["p_value_holm_behaviour"]),
        "median_correct_rt_estimate_theta_plus_minus_non_rhythmic": estimate(rt),
        "median_correct_rt_holm_p": float(rt["p_value_holm_behaviour"]),
        "accuracy_estimate_theta_plus_minus_non_rhythmic": accuracy_estimate,
        "accuracy_holm_p": float(accuracy["p_value_holm_behaviour"]),
        "accuracy_decrement_detected": accuracy_decrement_detected,
        "accuracy_increment_detected": accuracy_increment_detected,
        "accuracy_noninferiority_established": False,
        "acute_task_performance_advantage_claim_unlocked": bool(
            ie_favourable and not accuracy_decrement_detected
        ),
        "allowed_core_language": allowed,
        "forbidden_language": [
            "accuracy was proven unaffected",
            "no speed-accuracy trade-off",
            "long-term cognitive improvement",
            "treatment efficacy",
        ],
    }


def run(group_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    effects = pd.read_csv(group_root / "paired_behaviour_effects.csv")
    result = classify(effects)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "behaviour_branch_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-root", type=Path, default=DEFAULT_GROUP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.group_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
