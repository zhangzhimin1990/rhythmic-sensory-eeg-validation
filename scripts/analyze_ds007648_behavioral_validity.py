#!/usr/bin/env python3
"""Frozen full-cohort behavioral-validity models for OpenNeuro ds007648."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, spearmanr, ttest_1samp, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NEURAL_TERMS = (
    "hz40_within_z",
    "hz36_within_z",
    "hz40_within_z:target_auditory",
    "hz36_within_z:target_auditory",
)


def within_subject_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("subject")[column]
    centered = frame[column] - grouped.transform("mean")
    scale = grouped.transform("std").replace(0, np.nan)
    return centered / scale


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def prepare_trials(path: Path, strict_eog: bool = False, exclude_sub18: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    usable = frame.eeg_valid.fillna(False).astype(bool)
    if strict_eog:
        usable &= frame.eeg_valid_strict_eog.fillna(False).astype(bool)
    frame = frame.loc[usable].copy()
    if exclude_sub18:
        frame = frame.loc[frame.subject.ne("sub-18")].copy()
    frame["target_auditory"] = frame.trial_type.isin(["aud_dis", "aud_only", "nonsp_aud"]).astype(int)
    frame["trial_order_z"] = frame.groupby("subject").trial_index.transform(
        lambda x: (x - x.mean()) / x.std()
    )
    frame["log_rt"] = np.log(frame.response_time)
    return frame


def add_features(frame: pd.DataFrame, feature_family: str) -> pd.DataFrame:
    result = frame.copy()
    if feature_family == "phase_projected":
        source = {40: "hz40_phase_projected_uv", 36: "hz36_phase_projected_uv"}
    elif feature_family == "envelope":
        source = {40: "hz40_log_amp_ratio", 36: "hz36_log_amp_ratio"}
    else:
        raise ValueError(feature_family)
    for frequency, column in source.items():
        result[f"hz{frequency}_within_z"] = within_subject_zscore(result, column)
    return result.dropna(subset=["hz40_within_z", "hz36_within_z"])


def fit_cluster_models(frame: pd.DataFrame, analysis: str) -> pd.DataFrame:
    formula_rhs = (
        "C(subject) + C(trial_type) + trial_order_z + hz40_within_z + hz36_within_z + "
        "hz40_within_z:target_auditory + hz36_within_z:target_auditory"
    )
    if analysis == "accuracy":
        robust = smf.logit(f"Correct ~ {formula_rhs}", data=frame).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": frame.subject, "use_correction": True},
        )
        scale = "log_odds_per_within_subject_sd"
    elif analysis == "rt":
        rt_frame = frame.loc[frame.Correct.eq(1) & frame.response_time.gt(0)].copy()
        robust = smf.ols(f"log_rt ~ {formula_rhs}", data=rt_frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": rt_frame.subject, "use_correction": True},
        )
        frame = rt_frame
        scale = "log_rt_per_within_subject_sd"
    else:
        raise ValueError(analysis)
    names = robust.params.index.tolist()
    confidence = robust.conf_int()
    rows = []
    for term in NEURAL_TERMS:
        index = names.index(term)
        rows.append(
            {
                "analysis": analysis,
                "term": term,
                "estimate": float(robust.params.iloc[index]),
                "std_error_cluster": float(robust.bse.iloc[index]),
                "ci_low": float(confidence.iloc[index, 0]),
                "ci_high": float(confidence.iloc[index, 1]),
                "p_value": float(robust.pvalues.iloc[index]),
                "scale": scale,
                "n_trials": int(len(frame)),
                "n_subjects": int(frame.subject.nunique()),
            }
        )
    covariance = robust.cov_params()
    for frequency in (40, 36):
        main_term = f"hz{frequency}_within_z"
        interaction_term = f"hz{frequency}_within_z:target_auditory"
        for label, main_weight, interaction_weight in [
            ("visual", 1.0, 0.0),
            ("auditory", 1.0, 1.0),
            ("average", 1.0, 0.5),
        ]:
            weights = pd.Series(0.0, index=names)
            weights[main_term] = main_weight
            weights[interaction_term] = interaction_weight
            estimate = float(weights @ robust.params)
            std_error = float(np.sqrt(weights @ covariance @ weights))
            z_value = estimate / std_error
            rows.append(
                {
                    "analysis": analysis,
                    "term": f"hz{frequency}_{label}_contrast",
                    "estimate": estimate,
                    "std_error_cluster": std_error,
                    "ci_low": estimate - 1.96 * std_error,
                    "ci_high": estimate + 1.96 * std_error,
                    "p_value": float(2 * norm.sf(abs(z_value))),
                    "scale": scale,
                    "n_trials": int(len(frame)),
                    "n_subjects": int(frame.subject.nunique()),
                }
            )
    return pd.DataFrame(rows)


def two_stage_subject_slopes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    individual = []
    for subject, group in frame.groupby("subject"):
        for analysis in ("accuracy", "rt"):
            if analysis == "accuracy":
                fitted = smf.glm(
                    "Correct ~ C(trial_type) + trial_order_z + hz40_within_z + hz36_within_z",
                    data=group,
                    family=sm.families.Binomial(),
                ).fit()
            else:
                used = group.loc[group.Correct.eq(1) & group.response_time.gt(0)]
                fitted = smf.ols(
                    "log_rt ~ C(trial_type) + trial_order_z + hz40_within_z + hz36_within_z",
                    data=used,
                ).fit()
            for frequency in (40, 36):
                individual.append(
                    {
                        "subject": subject,
                        "analysis": analysis,
                        "frequency_hz": frequency,
                        "slope": float(fitted.params[f"hz{frequency}_within_z"]),
                    }
                )
    individual_frame = pd.DataFrame(individual)
    group_rows = []
    for (analysis, frequency), group in individual_frame.groupby(["analysis", "frequency_hz"]):
        t_result = ttest_1samp(group.slope, 0)
        w_result = wilcoxon(group.slope)
        group_rows.append(
            {
                "analysis": analysis,
                "frequency_hz": frequency,
                "n_subjects": int(len(group)),
                "mean_slope": float(group.slope.mean()),
                "median_slope": float(group.slope.median()),
                "positive_subjects": int(group.slope.gt(0).sum()),
                "t_statistic": float(t_result.statistic),
                "t_p_value": float(t_result.pvalue),
                "wilcoxon_statistic": float(w_result.statistic),
                "wilcoxon_p_value": float(w_result.pvalue),
            }
        )
    return individual_frame, pd.DataFrame(group_rows)


def loso_incremental_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    categorical = ["trial_type"]
    base_numeric = ["trial_order_z"]
    neural = [
        "hz40_within_z",
        "hz36_within_z",
        "hz40_auditory_interaction",
        "hz36_auditory_interaction",
    ]
    frame = frame.copy()
    frame["hz40_auditory_interaction"] = frame.hz40_within_z * frame.target_auditory
    frame["hz36_auditory_interaction"] = frame.hz36_within_z * frame.target_auditory
    rows = []
    for held_out in sorted(frame.subject.unique()):
        train = frame.loc[frame.subject.ne(held_out)]
        test = frame.loc[frame.subject.eq(held_out)]
        for outcome in ("accuracy", "rt"):
            if outcome == "rt":
                train = train.loc[train.Correct.eq(1) & train.response_time.gt(0)]
                test = test.loc[test.Correct.eq(1) & test.response_time.gt(0)]
            for model_name, numeric in [("base", base_numeric), ("neural", base_numeric + neural)]:
                transform = ColumnTransformer(
                    [
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("numeric", make_pipeline(SimpleImputer(), StandardScaler()), numeric),
                    ]
                )
                if outcome == "accuracy":
                    estimator = make_pipeline(transform, LogisticRegression(max_iter=2000, C=1.0))
                    estimator.fit(train[categorical + numeric], train.Correct)
                    prediction = estimator.predict_proba(test[categorical + numeric])[:, 1]
                    score = log_loss(test.Correct, prediction, labels=[0, 1])
                    metric = "log_loss"
                else:
                    estimator = make_pipeline(transform, Ridge(alpha=1.0))
                    estimator.fit(train[categorical + numeric], train.log_rt)
                    prediction = estimator.predict(test[categorical + numeric])
                    score = mean_absolute_error(test.log_rt, prediction)
                    metric = "log_rt_mae"
                rows.append(
                    {
                        "subject": held_out,
                        "outcome": outcome,
                        "model": model_name,
                        "metric": metric,
                        "score": float(score),
                        "n_test": int(len(test)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_cv(folds: pd.DataFrame, seed: int = 20260920) -> pd.DataFrame:
    wide = folds.pivot(index=["subject", "outcome", "metric"], columns="model", values="score").reset_index()
    wide["delta_neural_minus_base"] = wide.neural - wide.base
    rng = np.random.default_rng(seed)
    rows = []
    for (outcome, metric), group in wide.groupby(["outcome", "metric"]):
        values = group.delta_neural_minus_base.to_numpy()
        boot = np.mean(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
        rows.append(
            {
                "outcome": outcome,
                "metric": metric,
                "n_subjects": len(values),
                "mean_base": float(group.base.mean()),
                "mean_neural": float(group.neural.mean()),
                "mean_delta_neural_minus_base": float(values.mean()),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "subjects_improved": int((values < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def subject_level_associations(summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    frame = frame.loc[frame.n_trials_valid.gt(0)].copy()
    rows = []
    for neural in ("hz40_itpc", "hz36_itpc"):
        for outcome in ("accuracy_all_trials", "median_correct_rt"):
            rho, pvalue = spearmanr(frame[neural], frame[outcome])
            rows.append(
                {
                    "neural_measure": neural,
                    "outcome": outcome,
                    "spearman_rho": float(rho),
                    "p_value": float(pvalue),
                    "n_subjects": int(len(frame)),
                }
            )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = benjamini_hochberg(result.p_value.to_numpy())
    return result


def split_half_reliability(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (subject, half), group in frame.groupby(["subject", frame.trial_index.mod(2)]):
        row: dict[str, object] = {"subject": subject, "half": int(half), "n_trials": len(group)}
        for frequency in (36, 40):
            coefficient = (
                group[f"hz{frequency}_coefficient_real"].to_numpy()
                + 1j * group[f"hz{frequency}_coefficient_imag"].to_numpy()
            )
            unit_phase = coefficient / np.maximum(np.abs(coefficient), np.finfo(float).tiny)
            row[f"hz{frequency}_itpc"] = float(np.abs(unit_phase.mean()))
            row[f"hz{frequency}_median_log_amp_ratio"] = float(
                group[f"hz{frequency}_log_amp_ratio"].median()
            )
        rows.append(row)
    halves = pd.DataFrame(rows)
    reliability_rows = []
    for measure in [
        "hz36_itpc",
        "hz40_itpc",
        "hz36_median_log_amp_ratio",
        "hz40_median_log_amp_ratio",
    ]:
        wide = halves.pivot(index="subject", columns="half", values=measure).dropna()
        rho, pvalue = spearmanr(wide[0], wide[1])
        reliability_rows.append(
            {
                "measure": measure,
                "n_subjects": len(wide),
                "odd_even_spearman_rho": float(rho),
                "p_value": float(pvalue),
                "spearman_brown_full_length": float(2 * rho / (1 + rho)),
            }
        )
    return halves, pd.DataFrame(reliability_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("outputs/derived/ds007648_trial_features/all_trial_features.csv")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("outputs/derived/ds007648_trial_features/subject_summary.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/models/ds007648_behavioral_validity")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model_rows = []
    cv_outputs = []
    specifications = [
        ("main", False, False, "phase_projected"),
        ("strict_eog", True, False, "phase_projected"),
        ("exclude_sub18", False, True, "phase_projected"),
        ("envelope", False, False, "envelope"),
    ]
    for specification, strict_eog, exclude_sub18, feature_family in specifications:
        frame = add_features(
            prepare_trials(args.input, strict_eog=strict_eog, exclude_sub18=exclude_sub18),
            feature_family,
        )
        for analysis in ("accuracy", "rt"):
            result = fit_cluster_models(frame, analysis)
            result.insert(0, "specification", specification)
            result.insert(1, "feature_family", feature_family)
            model_rows.append(result)
        if specification == "main":
            folds = loso_incremental_prediction(frame)
            folds.to_csv(args.output / "loso_fold_metrics.csv", index=False)
            summarize_cv(folds).to_csv(args.output / "loso_summary.csv", index=False)

    models = pd.concat(model_rows, ignore_index=True)
    multiplicity_terms = [
        "hz40_average_contrast",
        "hz36_average_contrast",
        "hz40_within_z:target_auditory",
        "hz36_within_z:target_auditory",
    ]
    models["q_value_within_outcome_four_tests"] = np.nan
    for (specification, analysis), index in models.loc[
        models.term.isin(multiplicity_terms)
    ].groupby(["specification", "analysis"]).groups.items():
        models.loc[index, "q_value_within_outcome_four_tests"] = benjamini_hochberg(
            models.loc[index, "p_value"].to_numpy()
        )
    primary_mask = models.specification.eq("main") & models.term.eq("hz40_average_contrast")
    models["q_value_primary_two_outcomes"] = np.nan
    models.loc[primary_mask, "q_value_primary_two_outcomes"] = benjamini_hochberg(
        models.loc[primary_mask, "p_value"].to_numpy()
    )
    models.to_csv(args.output / "clustered_models.csv", index=False)
    main_frame = add_features(prepare_trials(args.input), "phase_projected")
    individual_slopes, slope_summary = two_stage_subject_slopes(main_frame)
    individual_slopes.to_csv(args.output / "two_stage_subject_slopes.csv", index=False)
    slope_summary.to_csv(args.output / "two_stage_slope_summary.csv", index=False)
    subject_level_associations(args.summary).to_csv(
        args.output / "subject_level_associations.csv", index=False
    )
    main_raw = prepare_trials(args.input)
    half_values, half_reliability = split_half_reliability(main_raw)
    half_values.to_csv(args.output / "split_half_values.csv", index=False)
    half_reliability.to_csv(args.output / "split_half_reliability.csv", index=False)
    manifest = {
        "input_trials": str(args.input),
        "input_summary": str(args.summary),
        "main_feature": "leave-one-trial-out phase-projected target-frequency coefficient",
        "main_sample_rule": "eeg_valid; sub-18 retained after manual audit",
        "sensitivity": ["strict EOG", "exclude sub-18", "Hilbert envelope log ratio"],
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(models.loc[primary_mask].to_string(index=False))
    print((args.output / "loso_summary.csv").read_text())


if __name__ == "__main__":
    main()
