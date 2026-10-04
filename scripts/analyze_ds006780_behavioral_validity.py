#!/usr/bin/env python3
"""Age- and diagnosis-adjusted validity analysis for OpenNeuro ds006780."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NEURAL_FAMILIES = {
    "local_log_snr": "local_log_snr_db_mean",
    "morlet_power": "morlet_power_percent_change_mean",
    "itpc": "itpc",
}


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    result = np.full_like(pvalues, np.nan)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return result
    used = pvalues[finite]
    order = np.argsort(used)
    ranked = used[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[finite] = restored
    return result


def corrected_dprime(hits: int, misses: int, false_alarms: int, correct_rejections: int) -> float:
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    false_alarm_rate = (false_alarms + 0.5) / (
        false_alarms + correct_rejections + 1
    )
    return float(norm.ppf(hit_rate) - norm.ppf(false_alarm_rate))


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    valid = group[column].notna() & group.n_standard_valid.gt(0)
    if not valid.any():
        return np.nan
    return float(
        np.average(group.loc[valid, column], weights=group.loc[valid, "n_standard_valid"])
    )


def aggregate_subjects(runs: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    neural_columns = sorted(set(NEURAL_FAMILIES.values()) | {
        "morlet_power_percent_change_median",
        "local_log_snr_db_median",
        "itpc_odd_trials",
        "itpc_even_trials",
    })
    for (subject, frequency), group in runs.groupby(["subject", "frequency_hz"]):
        counts = {
            column: int(group[column].fillna(0).sum())
            for column in ("hits", "misses", "false_alarms", "correct_rejections")
        }
        row: dict[str, object] = {
            "subject": subject,
            "frequency_hz": int(frequency),
            "n_runs": int(group.run.nunique()),
            "n_standard_expected": int(group.n_standard_expected.sum()),
            "n_standard_valid": int(group.n_standard_valid.sum()),
            **counts,
            "dprime": corrected_dprime(**counts),
            "median_hit_rt_run_weighted": weighted_mean(group, "median_hit_rt"),
            "unassigned_responses_total": int(group.unassigned_responses_total.fillna(0).sum()),
            "max_bad_channels": int(group.n_bad_channels.max()),
        }
        for column in neural_columns:
            row[column] = weighted_mean(group, column)
        rows.append(row)
    long = pd.DataFrame(rows)
    wide_parts = []
    value_columns = [column for column in long.columns if column not in {"subject", "frequency_hz"}]
    for frequency in (27, 40):
        part = long.loc[long.frequency_hz.eq(frequency), ["subject"] + value_columns].copy()
        part = part.rename(columns={column: f"{column}_{frequency}" for column in value_columns})
        wide_parts.append(part)
    wide = wide_parts[0].merge(wide_parts[1], on="subject", how="outer", validate="one_to_one")
    metadata = participants.rename(columns={"participant_id": "subject"}).copy()
    result = metadata.merge(wide, on="subject", how="inner", validate="one_to_one")
    result["technical_usable"] = (
        result.n_standard_valid_27.ge(100)
        & result.n_standard_valid_40.ge(100)
        & result.local_log_snr_db_mean_27.notna()
        & result.local_log_snr_db_mean_40.notna()
    )
    result["technical_exclusion_reason"] = np.where(
        result.technical_usable,
        "",
        "fewer_than_100_valid_standard_trials_in_at_least_one_frequency_or_missing_neural_metric",
    )
    return result


def standardized(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        scale = result[column].std(ddof=0)
        result[f"z_{column}"] = (result[column] - result[column].mean()) / scale
    result["age_z"] = (result.age - result.age.mean()) / result.age.std(ddof=0)
    return result


def fit_model(
    frame: pd.DataFrame,
    outcome: str,
    family: str,
    quadratic_age: bool = False,
    interaction: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = NEURAL_FAMILIES[family]
    p40 = f"z_{source}_40"
    p27 = f"z_{source}_27"
    needed = [outcome, "age", "sex", "group", f"{source}_40", f"{source}_27"]
    used = frame.dropna(subset=needed).copy()
    used = standardized(used, [f"{source}_40", f"{source}_27"])
    age_term = "age_z + I(age_z ** 2)" if quadratic_age else "age_z"
    if interaction:
        rhs = f"{p40} * C(group) + {p27} + {age_term} + C(sex)"
    else:
        rhs = f"{p40} + {p27} + {age_term} + C(sex) + C(group)"
    fitted = smf.ols(f"{outcome} ~ {rhs}", data=used).fit(cov_type="HC3")
    confidence = fitted.conf_int()
    rows = []
    for term in fitted.params.index:
        if term == "Intercept" or term.startswith("C("):
            continue
        rows.append(
            {
                "outcome": outcome,
                "feature_family": family,
                "quadratic_age": quadratic_age,
                "group_interaction": interaction,
                "term": term,
                "estimate": float(fitted.params[term]),
                "std_error_hc3": float(fitted.bse[term]),
                "ci_low": float(confidence.loc[term, 0]),
                "ci_high": float(confidence.loc[term, 1]),
                "p_value": float(fitted.pvalues[term]),
                "n_subjects": int(fitted.nobs),
                "r_squared": float(fitted.rsquared),
                "adjusted_r_squared": float(fitted.rsquared_adj),
            }
        )
    metadata = {
        "outcome": outcome,
        "feature_family": family,
        "quadratic_age": quadratic_age,
        "group_interaction": interaction,
        "formula": fitted.model.formula,
        "n_subjects": int(fitted.nobs),
    }
    return pd.DataFrame(rows), metadata


def split_half_reliability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frequency in (27, 40):
        odd = frame[f"itpc_odd_trials_{frequency}"]
        even = frame[f"itpc_even_trials_{frequency}"]
        valid = odd.notna() & even.notna()
        rho, pvalue = spearmanr(odd[valid], even[valid])
        rows.append(
            {
                "frequency_hz": frequency,
                "n_subjects": int(valid.sum()),
                "odd_even_spearman_rho": float(rho),
                "p_value": float(pvalue),
                "spearman_brown_full_length": float(2 * rho / (1 + rho)),
            }
        )
    return pd.DataFrame(rows)


def replication_models(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, source in NEURAL_FAMILIES.items():
        for frequency in (27, 40):
            outcome = f"{source}_{frequency}"
            used = frame.dropna(subset=[outcome, "age", "sex", "group"]).copy()
            used["age_z"] = (used.age - used.age.mean()) / used.age.std(ddof=0)
            fitted = smf.ols(f"{outcome} ~ age_z + C(sex) + C(group)", data=used).fit(
                cov_type="HC3"
            )
            confidence = fitted.conf_int()
            for term in fitted.params.index:
                if term == "Intercept" or term.startswith("C(sex)"):
                    continue
                rows.append(
                    {
                        "feature_family": family,
                        "frequency_hz": frequency,
                        "term": term,
                        "estimate": float(fitted.params[term]),
                        "ci_low": float(confidence.loc[term, 0]),
                        "ci_high": float(confidence.loc[term, 1]),
                        "p_value": float(fitted.pvalues[term]),
                        "n_subjects": int(fitted.nobs),
                    }
                )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = benjamini_hochberg(result.p_value.to_numpy())
    return result


def group_slope_contrasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    slope_rows = []
    joint_rows = []
    for family, source in NEURAL_FAMILIES.items():
        needed = ["dprime_40", "age", "sex", "group", f"{source}_40", f"{source}_27"]
        used = standardized(
            frame.dropna(subset=needed), [f"{source}_40", f"{source}_27"]
        )
        p40, p27 = f"z_{source}_40", f"z_{source}_27"
        fitted = smf.ols(
            f"dprime_40 ~ {p40} * C(group) + {p27} + age_z + C(sex)",
            data=used,
        ).fit(cov_type="HC3")
        names = fitted.params.index.tolist()
        covariance = fitted.cov_params()
        interaction_terms = [
            f"{p40}:C(group)[T.ASD SIBLING]", f"{p40}:C(group)[T.TD]"
        ]
        for group, interaction_term in (
            ("ASD", None),
            ("ASD SIBLING", interaction_terms[0]),
            ("TD", interaction_terms[1]),
        ):
            weights = pd.Series(0.0, index=names)
            weights[p40] = 1.0
            if interaction_term in names:
                weights[interaction_term] = 1.0
            estimate = float(weights @ fitted.params)
            std_error = float(np.sqrt(weights @ covariance @ weights))
            z_value = estimate / std_error
            slope_rows.append(
                {
                    "feature_family": family,
                    "group": group,
                    "estimate_40_per_sd": estimate,
                    "std_error_hc3": std_error,
                    "ci_low": estimate - 1.96 * std_error,
                    "ci_high": estimate + 1.96 * std_error,
                    "p_value": float(2 * norm.sf(abs(z_value))),
                    "n_subjects_total": int(fitted.nobs),
                    "n_group": int(used.group.eq(group).sum()),
                }
            )
        restriction = np.zeros((2, len(names)))
        for row_index, term in enumerate(interaction_terms):
            restriction[row_index, names.index(term)] = 1.0
        test = fitted.wald_test(restriction, scalar=True)
        joint_rows.append(
            {
                "feature_family": family,
                "wald_statistic": float(test.statistic),
                "df": 2,
                "p_value": float(test.pvalue),
                "n_subjects": int(fitted.nobs),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    slopes["q_value_bh_nine_slopes"] = benjamini_hochberg(slopes.p_value.to_numpy())
    joint = pd.DataFrame(joint_rows)
    joint["q_value_bh_three_families"] = benjamini_hochberg(joint.p_value.to_numpy())
    return slopes, joint


def original_claim_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce unadjusted within-group associations before covariate adjustment.

    These estimates are deliberately labeled descriptive.  They mirror the logic of
    the source article's simple clinical correlations, but do not replace the
    prespecified age-, sex-, group-, and 27-Hz-adjusted models.
    """
    source = frame.copy()
    denominator = source.false_alarms_40 + source.correct_rejections_40
    source["false_alarm_rate_40"] = source.false_alarms_40 / denominator.replace(0, np.nan)
    rows = []
    for subset_name, subset in (
        ("all_technical_usable", source),
        ("ASD", source.loc[source.group.eq("ASD")]),
    ):
        for family, feature in NEURAL_FAMILIES.items():
            x_column = f"{feature}_40"
            for outcome in ("dprime_40", "false_alarm_rate_40"):
                used = subset[[x_column, outcome]].dropna()
                pearson = pearsonr(used[x_column], used[outcome])
                spearman = spearmanr(used[x_column], used[outcome])
                rows.append(
                    {
                        "subset": subset_name,
                        "feature_family": family,
                        "outcome": outcome,
                        "n_subjects": len(used),
                        "pearson_r": float(pearson.statistic),
                        "pearson_p": float(pearson.pvalue),
                        "spearman_rho": float(spearman.statistic),
                        "spearman_p": float(spearman.pvalue),
                    }
                )
    result = pd.DataFrame(rows)
    result["pearson_q_bh_twelve_tests"] = benjamini_hochberg(
        result.pearson_p.to_numpy()
    )
    result["spearman_q_bh_twelve_tests"] = benjamini_hochberg(
        result.spearman_p.to_numpy()
    )
    return result


def repeated_cv(frame: pd.DataFrame, outcome: str, seed: int = 20260921) -> pd.DataFrame:
    used = frame.dropna(
        subset=[outcome, "age", "sex", "group", "local_log_snr_db_mean_27", "local_log_snr_db_mean_40"]
    ).reset_index(drop=True)
    base_numeric = ["age"]
    neural_numeric = base_numeric + ["local_log_snr_db_mean_27", "local_log_snr_db_mean_40"]
    categorical = ["sex", "group"]
    folds = RepeatedKFold(n_splits=10, n_repeats=10, random_state=seed)
    rows = []
    for fold_index, (train_index, test_index) in enumerate(folds.split(used)):
        train, test = used.iloc[train_index], used.iloc[test_index]
        for model_name, numeric in (("base", base_numeric), ("neural", neural_numeric)):
            transform = ColumnTransformer(
                [
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                    ("numeric", make_pipeline(SimpleImputer(), StandardScaler()), numeric),
                ]
            )
            estimator = make_pipeline(transform, Ridge(alpha=1.0))
            estimator.fit(train[categorical + numeric], train[outcome])
            prediction = estimator.predict(test[categorical + numeric])
            rows.append(
                {
                    "outcome": outcome,
                    "fold": fold_index,
                    "repeat": fold_index // 10,
                    "model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "mae": float(mean_absolute_error(test[outcome], prediction)),
                    "rmse": float(mean_squared_error(test[outcome], prediction) ** 0.5),
                }
            )
    return pd.DataFrame(rows)


def summarize_cv(folds: pd.DataFrame, seed: int = 20260921) -> pd.DataFrame:
    wide = folds.pivot(index=["outcome", "fold", "repeat"], columns="model", values=["mae", "rmse"])
    rows = []
    rng = np.random.default_rng(seed)
    for outcome in folds.outcome.unique():
        selected = wide.loc[outcome]
        for metric in ("mae", "rmse"):
            delta = (selected[(metric, "neural")] - selected[(metric, "base")]).to_numpy()
            boot = np.mean(rng.choice(delta, size=(10000, len(delta)), replace=True), axis=1)
            rows.append(
                {
                    "outcome": outcome,
                    "metric": metric,
                    "n_folds": len(delta),
                    "mean_base": float(selected[(metric, "base")].mean()),
                    "mean_neural": float(selected[(metric, "neural")].mean()),
                    "mean_delta_neural_minus_base": float(delta.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "folds_improved": int((delta < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("outputs/derived/ds006780_assr_features/all_run_features.csv"),
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=Path("data/public/ds006780/v1.0.0/participants.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/models/ds006780_behavioral_validity"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.runs)
    participants = pd.read_csv(args.participants, sep="\t")
    subjects = aggregate_subjects(runs, participants)
    subjects.to_csv(args.output / "subject_level_features.csv", index=False)
    subjects.loc[~subjects.technical_usable].to_csv(
        args.output / "technical_exclusions.csv", index=False
    )
    analysis = subjects.loc[subjects.technical_usable].copy()

    coefficient_frames = []
    model_metadata = []
    for family in NEURAL_FAMILIES:
        for outcome in ("dprime_40", "fsiq"):
            for quadratic_age, interaction in ((False, False), (True, False), (False, True)):
                result, metadata = fit_model(
                    analysis, outcome, family,
                    quadratic_age=quadratic_age, interaction=interaction,
                )
                coefficient_frames.append(result)
                model_metadata.append(metadata)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    coefficients.to_csv(args.output / "model_coefficients.csv", index=False)

    primary = coefficients.loc[
        coefficients.feature_family.eq("local_log_snr")
        & ~coefficients.quadratic_age
        & ~coefficients.group_interaction
        & coefficients.term.eq("z_local_log_snr_db_mean_40")
    ].copy()
    primary["q_value_bh_two_outcomes"] = benjamini_hochberg(primary.p_value.to_numpy())
    primary.to_csv(args.output / "primary_tests.csv", index=False)

    sensitivity = coefficients.loc[
        coefficients.feature_family.isin(["morlet_power", "itpc"])
        & ~coefficients.quadratic_age
        & ~coefficients.group_interaction
        & coefficients.term.str.endswith("_40")
    ].copy()
    sensitivity["q_value_bh_four_tests"] = benjamini_hochberg(sensitivity.p_value.to_numpy())
    sensitivity.to_csv(args.output / "sensitivity_tests.csv", index=False)

    reliability = split_half_reliability(analysis)
    reliability.to_csv(args.output / "split_half_reliability.csv", index=False)
    replication_models(analysis).to_csv(args.output / "replication_models.csv", index=False)
    group_slopes, group_interactions = group_slope_contrasts(analysis)
    group_slopes.to_csv(args.output / "group_specific_behavior_slopes.csv", index=False)
    group_interactions.to_csv(args.output / "group_interaction_tests.csv", index=False)
    original_claim_correlations(analysis).to_csv(
        args.output / "original_claim_correlations.csv", index=False
    )

    folds = pd.concat(
        [repeated_cv(analysis, outcome) for outcome in ("dprime_40", "fsiq")],
        ignore_index=True,
    )
    folds.to_csv(args.output / "cross_validation_folds.csv", index=False)
    summarize_cv(folds).to_csv(args.output / "cross_validation_summary.csv", index=False)

    manifest = {
        "declared_subjects_with_assr": int(participants.completed_ASSR.eq("yes").sum()),
        "derived_subjects": int(len(subjects)),
        "technical_usable_subjects": int(analysis.subject.nunique()),
        "technical_exclusions": int((~subjects.technical_usable).sum()),
        "groups_usable": analysis.group.value_counts().to_dict(),
        "fsiq_complete_usable": int(analysis.fsiq.notna().sum()),
        "models": model_metadata,
        "feature_algorithm_version": sorted(runs.feature_algorithm_version.dropna().astype(str).unique()),
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
