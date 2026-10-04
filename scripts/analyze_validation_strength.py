#!/usr/bin/env python3
"""Quantify how strongly key null/limited effects constrain scientific claims.

The bounds in this script are sensitivity grids, not clinical MCIDs.  They are
reported on interpretable scales so the manuscript can distinguish a
non-significant result from evidence that excludes an effect of a stated size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


ROOT = Path(__file__).resolve().parents[1]


def tost_normal(
    estimate: float, standard_error: float, lower: float, upper: float
) -> dict[str, float | bool]:
    """Two one-sided normal-approximation equivalence test at alpha=.05."""
    if not lower < upper or standard_error <= 0:
        raise ValueError("Bounds must be ordered and standard_error must be positive")
    z_lower = (estimate - lower) / standard_error
    z_upper = (estimate - upper) / standard_error
    p_lower = float(norm.sf(z_lower))
    p_upper = float(norm.cdf(z_upper))
    ci90_low = estimate - norm.ppf(0.95) * standard_error
    ci90_high = estimate + norm.ppf(0.95) * standard_error
    return {
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
        "ci90_low": float(ci90_low),
        "ci90_high": float(ci90_high),
        "equivalent_alpha_0_05": bool(max(p_lower, p_upper) < 0.05),
    }


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, *, iterations: int = 20_000, seed: int = 20260921
) -> dict[str, float | int]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[keep], np.asarray(y)[keep]
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        indices = rng.integers(0, len(x), len(x))
        if np.unique(x[indices]).size < 2 or np.unique(y[indices]).size < 2:
            continue
        value = spearmanr(x[indices], y[indices]).statistic
        if np.isfinite(value):
            values.append(float(value))
    values_array = np.asarray(values)
    return {
        "n": int(len(x)),
        "estimate": observed,
        "ci90_low": float(np.quantile(values_array, 0.05)),
        "ci90_high": float(np.quantile(values_array, 0.95)),
        "ci95_low": float(np.quantile(values_array, 0.025)),
        "ci95_high": float(np.quantile(values_array, 0.975)),
        "bootstrap_iterations_valid": int(len(values_array)),
    }


def build_equivalence_grid() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    ds7 = pd.read_csv(
        ROOT / "outputs/models/ds007648_behavioral_validity/clustered_models.csv"
    )
    for outcome, term, bounds in (
        ("accuracy", "hz40_average_contrast", [("OR_1.10", np.log(1.10)), ("OR_1.12", np.log(1.12))]),
        ("rt", "hz40_average_contrast", [("ratio_1.01", np.log(1.01)), ("ratio_1.02", np.log(1.02))]),
    ):
        selected = ds7.loc[
            ds7.specification.eq("main")
            & ds7.feature_family.eq("phase_projected")
            & ds7.analysis.eq(outcome)
            & ds7.term.eq(term)
        ].iloc[0]
        for label, bound in bounds:
            result = tost_normal(
                float(selected.estimate), float(selected.std_error_cluster), -bound, bound
            )
            rows.append(
                {
                    "dataset": "ds007648",
                    "outcome": outcome,
                    "predictor": "40Hz_trial_phase_projected_score",
                    "effect_scale": selected.scale,
                    "estimate": selected.estimate,
                    "standard_error": selected.std_error_cluster,
                    "bound_label": label,
                    "lower_bound": -bound,
                    "upper_bound": bound,
                    "bound_status": "sensitivity_not_clinical_mcid",
                    **result,
                }
            )

    ds6 = pd.read_csv(
        ROOT / "outputs/models/ds006780_behavioral_validity/primary_tests.csv"
    )
    subjects = pd.read_csv(
        ROOT / "outputs/models/ds006780_behavioral_validity/subject_level_features.csv"
    )
    outcome_sd = float(
        subjects.loc[subjects.technical_usable, "dprime_40"].std(ddof=0)
    )
    ds6_sensitivity = pd.read_csv(
        ROOT / "outputs/models/ds006780_behavioral_validity/sensitivity_tests.csv"
    )
    ds6_models = {
        "local_log_snr": ds6.loc[ds6.outcome.eq("dprime_40")].iloc[0],
        "itpc": ds6_sensitivity.loc[
            ds6_sensitivity.outcome.eq("dprime_40")
            & ds6_sensitivity.feature_family.eq("itpc")
            & ~ds6_sensitivity.quadratic_age
            & ~ds6_sensitivity.group_interaction
        ].iloc[0],
    }
    for predictor, selected in ds6_models.items():
        estimate = float(selected.estimate) / outcome_sd
        standard_error = float(selected.std_error_hc3) / outcome_sd
        for bound in (0.20, 0.25, 0.30):
            result = tost_normal(estimate, standard_error, -bound, bound)
            rows.append(
                {
                    "dataset": "ds006780",
                    "outcome": "dprime_40",
                    "predictor": predictor,
                    "effect_scale": "outcome_sd_per_neural_predictor_sd",
                    "estimate": estimate,
                    "standard_error": standard_error,
                    "bound_label": f"standardized_{bound:.2f}",
                    "lower_bound": -bound,
                    "upper_bound": bound,
                    "bound_status": "sensitivity_not_clinical_mcid",
                    **result,
                }
            )
    return pd.DataFrame(rows)


def build_correlation_precision() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    data = pd.read_csv(ROOT / "outputs/qc/ds005048_full_foundation/subject_summary.csv")
    used = data[["snr_narrow_db_stim_minus_rest", "mmse"]].dropna()
    result = bootstrap_spearman(
        used.snr_narrow_db_stim_minus_rest.to_numpy(), used.mmse.to_numpy()
    )
    rows.append(
        {
            "dataset": "ds005048",
            "population": "older_adults_MCI_AD",
            "outcome": "MMSE",
            "predictor": "40Hz_stim_minus_rest_local_snr",
            "effect_scale": "spearman_rho",
            **result,
            "interpretation": "imprecise_no_task_specific_equivalence_bound",
        }
    )

    visual = pd.read_csv(
        ROOT / "outputs/competition/ds006036_exact_preprint/features_ica.csv"
    )
    group_map = {"A": "AD", "F": "FTD"}
    for offset, (code, label) in enumerate(group_map.items(), start=1):
        used = visual.loc[
            visual["group"].eq(code), ["post_10Hz_di", "mmse"]
        ].dropna()
        result = bootstrap_spearman(
            used.post_10Hz_di.to_numpy(),
            used.mmse.to_numpy(),
            seed=20260921 + offset,
        )
        rows.append(
            {
                "dataset": "ds006036",
                "population": label,
                "outcome": "MMSE",
                "predictor": "closed_eye_10Hz_driving_index",
                "effect_scale": "spearman_rho",
                **result,
                "interpretation": "imprecise_within_diagnosis_association",
            }
        )
    return pd.DataFrame(rows)


def build_prediction_precision() -> pd.DataFrame:
    ds7 = pd.read_csv(ROOT / "outputs/models/ds007648_behavioral_validity/loso_summary.csv")
    ds7 = ds7.assign(dataset="ds007648", validation="leave_one_subject_out")
    ds7 = ds7.rename(
        columns={
            "mean_delta_neural_minus_base": "delta_neural_minus_base",
            "n_subjects": "n_units",
        }
    )
    ds6 = pd.read_csv(
        ROOT / "outputs/models/ds006780_behavioral_validity/cross_validation_summary.csv"
    )
    ds6 = ds6.assign(dataset="ds006780", validation="repeated_10x10_fold")
    ds6 = ds6.rename(
        columns={
            "mean_delta_neural_minus_base": "delta_neural_minus_base",
            "n_folds": "n_units",
        }
    )
    columns = [
        "dataset", "outcome", "metric", "validation", "n_units", "mean_base",
        "mean_neural", "delta_neural_minus_base", "ci_low", "ci_high",
    ]
    result = pd.concat([ds7[columns], ds6[columns]], ignore_index=True)

    dryad = pd.read_csv(
        ROOT / "outputs/models/dryad_theta_validation_gate/pretreatment_prediction_summary.csv"
    ).iloc[0]
    result = pd.concat(
        [
            result,
            pd.DataFrame(
                [
                    {
                        "dataset": "dryad_theta",
                        "outcome": "personalized_advantage_ms",
                        "metric": "mae_ms",
                        "validation": dryad.validation,
                        "n_units": int(dryad.n_subjects),
                        "mean_base": dryad.base_mae_ms,
                        "mean_neural": dryad.model_mae_ms,
                        "delta_neural_minus_base": dryad.delta_mae_model_minus_base_ms,
                        "ci_low": dryad.delta_mae_ci95_low,
                        "ci_high": dryad.delta_mae_ci95_high,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    result["relative_delta_percent"] = (
        100 * result["delta_neural_minus_base"] / result["mean_base"]
    )
    result["relative_ci_low_percent"] = 100 * result["ci_low"] / result["mean_base"]
    result["relative_ci_high_percent"] = 100 * result["ci_high"] / result["mean_base"]
    result["direction"] = np.select(
        [result["ci_high"] < 0, result["ci_low"] > 0],
        ["clear_improvement", "clear_worsening"],
        default="no_clear_increment",
    )
    return result


def build_claim_strength_summary(
    equivalence: pd.DataFrame,
    correlations: pd.DataFrame,
    prediction: pd.DataFrame,
) -> pd.DataFrame:
    """Create a manuscript-facing classification without inventing clinical MCIDs."""
    rows: list[dict[str, object]] = []
    for _, row in correlations.iterrows():
        rows.append(
            {
                "dataset": row.dataset,
                "claim": f"{row.predictor}_to_{row.outcome}_{row.population}",
                "evidence_type": "association_precision",
                "estimate": row.estimate,
                "interval_low": row.ci95_low,
                "interval_high": row.ci95_high,
                "strength": "imprecise",
                "permitted_wording": "no stable association detected; interval remains compatible with non-trivial effects",
                "prohibited_wording": "equivalent to zero; clinically irrelevant",
            }
        )

    dryad_assoc = pd.read_csv(
        ROOT / "outputs/models/dryad_theta_validation_gate/association_models.csv"
    )
    dryad_assoc = dryad_assoc.loc[
        dryad_assoc["model"].eq("continuous_post_rt_ms")
        & dryad_assoc["term"].eq("entrainment_within_z")
    ].iloc[0]
    rows.append(
        {
            "dataset": "dryad_theta",
            "claim": "within_person_entrainment_to_post_rt",
            "evidence_type": "association_precision",
            "estimate": dryad_assoc.estimate,
            "interval_low": dryad_assoc.ci95_low,
            "interval_high": dryad_assoc.ci95_high,
            "strength": "imprecise",
            "permitted_wording": "within-person condition-level association was imprecise and did not support mediation",
            "prohibited_wording": "entrainment is not a mechanism",
        }
    )

    chosen_bounds = {
        ("ds007648", "accuracy", "40Hz_trial_phase_projected_score"): "OR_1.12",
        ("ds007648", "rt", "40Hz_trial_phase_projected_score"): "ratio_1.02",
        ("ds006780", "dprime_40", "local_log_snr"): "standardized_0.25",
        ("ds006780", "dprime_40", "itpc"): "standardized_0.30",
    }
    for (dataset, outcome, predictor), bound_label in chosen_bounds.items():
        row = equivalence.loc[
            equivalence["dataset"].eq(dataset)
            & equivalence["outcome"].eq(outcome)
            & equivalence["predictor"].eq(predictor)
            & equivalence["bound_label"].eq(bound_label)
        ].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "claim": f"{predictor}_to_{outcome}_sensitivity_bound_{bound_label}",
                "evidence_type": "nonclinical_sensitivity_equivalence",
                "estimate": row.estimate,
                "interval_low": row.ci90_low,
                "interval_high": row.ci90_high,
                "strength": "bound_excluded_not_clinical_equivalence",
                "permitted_wording": f"effects at or beyond the {bound_label} sensitivity bound were excluded",
                "prohibited_wording": "clinically equivalent; no effect",
            }
        )

    for _, row in prediction.iterrows():
        rows.append(
            {
                "dataset": row.dataset,
                "claim": f"{row.outcome}_{row.metric}_prediction_increment",
                "evidence_type": row.validation,
                "estimate": row.relative_delta_percent,
                "interval_low": row.relative_ci_low_percent,
                "interval_high": row.relative_ci_high_percent,
                "strength": row.direction,
                "permitted_wording": (
                    "neural features worsened held-out prediction"
                    if row.direction == "clear_worsening"
                    else "no clear out-of-sample prediction gain"
                ),
                "prohibited_wording": "individual differences do not exist",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/models/validation_strength",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    equivalence = build_equivalence_grid()
    correlations = build_correlation_precision()
    prediction = build_prediction_precision()
    claim_strength = build_claim_strength_summary(equivalence, correlations, prediction)
    equivalence.to_csv(args.output / "association_equivalence_sensitivity.csv", index=False)
    correlations.to_csv(args.output / "correlation_precision.csv", index=False)
    prediction.to_csv(args.output / "prediction_increment_precision.csv", index=False)
    claim_strength.to_csv(args.output / "claim_strength_summary.csv", index=False)

    manifest = {
        "equivalence_rows": len(equivalence),
        "correlation_rows": len(correlations),
        "prediction_rows": len(prediction),
        "claim_strength_rows": len(claim_strength),
        "alpha": 0.05,
        "equivalence_ci": "90% normal approximation, equivalent to TOST alpha=.05",
        "boundary_policy": "sensitivity grids only; none are claimed as clinical MCIDs",
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
