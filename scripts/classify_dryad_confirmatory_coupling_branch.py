#!/usr/bin/env python3
"""Separate within-condition coupling from theta-plus-specific coupling evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_ROOT = ROOT / "outputs/models/dryad_confirmatory_group_v1"
DEFAULT_AUDIT_ROOT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_v1"
DEFAULT_OUTPUT = ROOT / "outputs/status/dryad_confirmatory_coupling_branch_v1"
MODELS = {"correctness_binomial_gee": 1, "correct_rt_gaussian_gee": -1}
INTERACTION_TERM = "phase_alignment_within:C(condition)[T.f_theta_plus]"


def _favourable(value: float, direction: int) -> bool:
    return value * direction > 0


def classify(simple_slopes: pd.DataFrame, gee_terms: pd.DataFrame) -> dict[str, object]:
    expected_pairs = {(model, condition) for model in MODELS for condition in ("non_rhythmic", "f_theta_plus")}
    actual_pairs = set(zip(simple_slopes["model"], simple_slopes["condition"]))
    if actual_pairs != expected_pairs or len(simple_slopes) != 4:
        raise ValueError("simple slopes must contain the two frozen outcomes by two conditions")
    interaction = gee_terms.loc[gee_terms["term"].eq(INTERACTION_TERM)]
    if set(interaction["model"]) != set(MODELS) or len(interaction) != 2:
        raise ValueError("GEE terms must contain one frozen interaction per outcome")

    endpoint_rows = []
    for model, favourable_direction in MODELS.items():
        theta = simple_slopes.loc[
            simple_slopes["model"].eq(model) & simple_slopes["condition"].eq("f_theta_plus")
        ].iloc[0]
        nonrhythmic = simple_slopes.loc[
            simple_slopes["model"].eq(model) & simple_slopes["condition"].eq("non_rhythmic")
        ].iloc[0]
        contrast = interaction.loc[interaction["model"].eq(model)].iloc[0]
        theta_estimate = float(theta["simple_slope_phase_alignment_within"])
        contrast_estimate = float(contrast["estimate"])
        theta_supported = bool(
            _favourable(theta_estimate, favourable_direction)
            and float(theta["p_value_holm_simple_slope_family"]) < 0.05
        )
        interaction_supported = bool(
            _favourable(contrast_estimate, favourable_direction)
            and float(contrast["p_value_holm_within_family"]) < 0.05
        )
        endpoint_rows.append({
            "model": model,
            "favourable_direction": "positive" if favourable_direction > 0 else "negative",
            "theta_plus_simple_slope": theta_estimate,
            "theta_plus_simple_slope_holm_p": float(theta["p_value_holm_simple_slope_family"]),
            "non_rhythmic_simple_slope": float(nonrhythmic["simple_slope_phase_alignment_within"]),
            "non_rhythmic_simple_slope_holm_p": float(nonrhythmic["p_value_holm_simple_slope_family"]),
            "theta_plus_minus_non_rhythmic_interaction": contrast_estimate,
            "interaction_holm_p": float(contrast["p_value_holm_within_family"]),
            "favourable_theta_plus_within_condition_coupling": theta_supported,
            "favourable_theta_plus_specific_increment": interaction_supported,
        })

    matched_specific = any(
        row["favourable_theta_plus_within_condition_coupling"]
        and row["favourable_theta_plus_specific_increment"]
        for row in endpoint_rows
    )
    any_theta = any(row["favourable_theta_plus_within_condition_coupling"] for row in endpoint_rows)
    any_interaction = any(row["favourable_theta_plus_specific_increment"] for row in endpoint_rows)
    if matched_specific:
        branch = "theta_plus_coupling_with_condition_specific_increment"
        allowed = (
            "at least one prespecified endpoint showed favourable within-theta-plus coupling "
            "and a favourable theta-plus versus non-rhythmic slope difference"
        )
    elif any_theta:
        branch = "theta_plus_within_condition_coupling_not_condition_specific"
        allowed = (
            "at least one endpoint showed favourable concurrent coupling within theta-plus "
            "trials, but no matched condition-specific slope increment was established"
        )
    elif any_interaction:
        branch = "condition_difference_without_favourable_theta_plus_slope"
        allowed = (
            "a favourable condition interaction was detected without a corresponding "
            "Holm-supported favourable theta-plus simple slope"
        )
    else:
        branch = "no_favourable_theta_plus_coupling"
        allowed = "no prespecified endpoint supported favourable concurrent coupling within theta-plus trials"

    return {
        "status": "confirmatory_coupling_branch_classified",
        "branch": branch,
        "endpoint_decisions": endpoint_rows,
        "theta_plus_within_condition_coupling_supported": any_theta,
        "theta_plus_specific_coupling_supported": matched_specific,
        "allowed_core_language": allowed,
        "forbidden_upgrades": [
            "neural mediation",
            "causal ordering",
            "theta-plus specificity from a simple slope alone",
            "treatment mechanism",
        ],
    }


def run(group_root: Path, audit_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    result = classify(
        pd.read_csv(audit_root / "within_condition_coupling_simple_slopes.csv"),
        pd.read_csv(group_root / "within_subject_coupling.csv"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coupling_branch_decision.json").write_text(
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
