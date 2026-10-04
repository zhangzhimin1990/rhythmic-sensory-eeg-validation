#!/usr/bin/env python3
"""Robustness and influence audit of the ds006780 signal--FSIQ pattern."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor


P40 = "local_log_snr_db_mean_40"
P27 = "local_log_snr_db_mean_27"


def add_standardized(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (P40, P27, "age", "n_standard_valid_40", "n_standard_valid_27"):
        result[f"z_{column}"] = (result[column] - result[column].mean()) / result[column].std(ddof=0)
    return result


def fit_variant(frame: pd.DataFrame, label: str, rhs: str) -> tuple[dict[str, object], object]:
    fitted = smf.ols(f"fsiq ~ {rhs}", data=frame).fit(cov_type="HC3")
    term = f"z_{P40}"
    confidence = fitted.conf_int().loc[term]
    return (
        {
            "variant": label,
            "estimate_40_per_sd": float(fitted.params[term]),
            "std_error_hc3": float(fitted.bse[term]),
            "ci_low": float(confidence.iloc[0]),
            "ci_high": float(confidence.iloc[1]),
            "p_value": float(fitted.pvalues[term]),
            "n_subjects": int(fitted.nobs),
            "adjusted_r_squared": float(fitted.rsquared_adj),
            "formula": fitted.model.formula,
        },
        fitted,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/models/ds006780_behavioral_validity/subject_level_features.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/models/ds006780_behavioral_validity/fsiq_audit"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    all_subjects = pd.read_csv(args.input)
    usable = all_subjects.loc[all_subjects.technical_usable].dropna(
        subset=["fsiq", P40, P27, "age", "sex", "group"]
    ).copy()
    usable = add_standardized(usable)
    common = f"z_{P40} + z_{P27} + z_age + C(sex) + C(group)"
    rows = []
    main_row, main_fit = fit_variant(usable, "main", common)
    rows.append(main_row)
    rows.append(fit_variant(usable, "without_27hz_control", f"z_{P40} + z_age + C(sex) + C(group)")[0])
    rows.append(fit_variant(usable, "quadratic_age", common + " + I(z_age ** 2)")[0])
    rows.append(
        fit_variant(
            usable,
            "signal_quality_adjusted",
            common
            + " + z_n_standard_valid_40 + z_n_standard_valid_27"
            + " + max_bad_channels_40",
        )[0]
    )

    winsorized = usable.copy()
    for column in (P40, P27):
        low, high = winsorized[column].quantile([0.025, 0.975])
        winsorized[column] = winsorized[column].clip(low, high)
    winsorized = add_standardized(winsorized)
    rows.append(fit_variant(winsorized, "winsorized_2.5_percent", common)[0])

    eligibility_subsets = {
        "published_age_8_to_12": usable.loc[usable.age.ge(8) & usable.age.lt(13)],
        "published_fsiq_above_80": usable.loc[usable.fsiq.gt(80)],
        "published_age_and_fsiq": usable.loc[
            usable.age.ge(8) & usable.age.lt(13) & usable.fsiq.gt(80)
        ],
    }
    hit_rate = (usable.hits_40 + usable.hits_27) / (
        usable.hits_40 + usable.hits_27 + usable.misses_40 + usable.misses_27
    )
    false_alarm_rate = (usable.false_alarms_40 + usable.false_alarms_27) / (
        usable.false_alarms_40
        + usable.false_alarms_27
        + usable.correct_rejections_40
        + usable.correct_rejections_27
    )
    eligibility_subsets["published_behavior_threshold"] = usable.loc[
        hit_rate.ge(0.5) & false_alarm_rate.le(0.5)
    ]
    for label, subset in eligibility_subsets.items():
        subset = add_standardized(subset)
        rows.append(fit_variant(subset, label, common)[0])
    pd.DataFrame(rows).to_csv(args.output / "robustness_models.csv", index=False)

    influence = main_fit.get_influence().summary_frame()
    influence.insert(0, "subject", usable.subject.to_numpy())
    influence.to_csv(args.output / "influence_diagnostics.csv", index=False)

    loo_rows = []
    for subject in usable.subject:
        subset = usable.loc[usable.subject.ne(subject)]
        fitted = smf.ols(f"fsiq ~ {common}", data=subset).fit(cov_type="HC3")
        term = f"z_{P40}"
        loo_rows.append(
            {
                "excluded_subject": subject,
                "estimate_40_per_sd": float(fitted.params[term]),
                "p_value": float(fitted.pvalues[term]),
            }
        )
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(args.output / "leave_one_out_coefficients.csv", index=False)

    rng = np.random.default_rng(20260921)
    bootstrap_estimates = []
    grouped = {name: group for name, group in usable.groupby("group")}
    for _ in range(10000):
        sample = pd.concat(
            [group.iloc[rng.integers(0, len(group), len(group))] for group in grouped.values()],
            ignore_index=True,
        )
        try:
            fitted = smf.ols(f"fsiq ~ {common}", data=sample).fit()
            bootstrap_estimates.append(float(fitted.params[f"z_{P40}"]))
        except np.linalg.LinAlgError:
            continue
    pd.DataFrame({"estimate_40_per_sd": bootstrap_estimates}).to_csv(
        args.output / "stratified_bootstrap_coefficients.csv", index=False
    )

    covariate_formula = f"z_{P27} + z_age + C(sex) + C(group)"
    fsiq_residual = smf.ols(f"fsiq ~ {covariate_formula}", data=usable).fit().resid
    snr_residual = smf.ols(f"z_{P40} ~ {covariate_formula}", data=usable).fit().resid
    partial_rho, partial_p = spearmanr(fsiq_residual, snr_residual)

    design = main_fit.model.exog
    names = main_fit.model.exog_names
    vif = {
        name: float(variance_inflation_factor(design, index))
        for index, name in enumerate(names)
        if name != "Intercept"
    }
    excluded = all_subjects.loc[~all_subjects.technical_usable]
    summary = {
        "n_fsiq_complete_usable": len(usable),
        "partial_spearman_rho": float(partial_rho),
        "partial_spearman_p_value": float(partial_p),
        "bootstrap_n": len(bootstrap_estimates),
        "bootstrap_median": float(np.median(bootstrap_estimates)),
        "bootstrap_ci_low": float(np.quantile(bootstrap_estimates, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap_estimates, 0.975)),
        "bootstrap_fraction_negative": float(np.mean(np.array(bootstrap_estimates) < 0)),
        "loo_estimate_min": float(loo.estimate_40_per_sd.min()),
        "loo_estimate_max": float(loo.estimate_40_per_sd.max()),
        "loo_all_negative": bool(loo.estimate_40_per_sd.lt(0).all()),
        "max_cooks_distance": float(influence.cooks_d.max()),
        "max_cooks_subject": str(influence.loc[influence.cooks_d.idxmax(), "subject"]),
        "variance_inflation_factors": vif,
        "technical_excluded_n": len(excluded),
        "technical_excluded_group_counts": excluded.group.value_counts().to_dict(),
        "technical_usable_group_counts": all_subjects.loc[
            all_subjects.technical_usable, "group"
        ].value_counts().to_dict(),
    }
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
