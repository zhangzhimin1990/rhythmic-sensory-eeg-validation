#!/usr/bin/env python3
"""Validation-gate analysis for the Dryad personalised-theta ageing dataset.

This analysis deliberately separates three questions that the source article
partly combines: (1) do personalised conditions outperform the active 2-Hz
control, (2) is entrainment associated with outcome between people or within
the same person across conditions, and (3) can pretreatment variables predict
who benefits from personalisation out of sample?

The workbook contains subject-level condition summaries, not trial-level data.
Accordingly, this script is a fast scientific gate; trial-level confirmation
requires the public raw BDF files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/public/dryad_t76hdr8dm/v5/metadata/Dataset.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs/models/dryad_theta_validation_gate"
CONDITIONS = {
    "2Hz": {
        "rt": "doshz_rt",
        "acc_con": "doshz_acc_con",
        "acc_incon": "doshz_acc_incon",
        "improvement": "base_doshz_rt_percent",
        "entrainment": "entrain_doshz",
    },
    "f_theta": {
        "rt": "fθ_rt",
        "acc_con": "fθ_acc_con",
        "acc_incon": "fθ_acc_incon",
        "improvement": "base_fθ_rt_percent",
        "entrainment": "entrain_fθ",
    },
    "f_theta_plus": {
        "rt": "fθ+_rt",
        "acc_con": "fθ+_acc_con",
        "acc_incon": "fθ+_acc_incon",
        "improvement": "base_fθ+_rt_percent",
        "entrainment": "entrain_fθ+",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"Cannot standardize {series.name}: zero/non-finite SD")
    return (series - series.mean()) / sd


def make_long(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, columns in CONDITIONS.items():
        for _, values in data.iterrows():
            accuracy = (values[columns["acc_con"]] + values[columns["acc_incon"]]) / 2
            rt = values[columns["rt"]]
            rows.append(
                {
                    "subject": int(values["subject"]),
                    "condition": condition,
                    "post_rt_ms": rt,
                    "accuracy": accuracy,
                    "inverse_efficiency_ms": rt / accuracy,
                    "rt_improvement_percent": values[columns["improvement"]],
                    "entrainment_uv2": values[columns["entrainment"]],
                    "baseline_rt_ms": values["bl_rt"],
                    "individual_freq_hz": values["individual_freq"],
                    "age_years": values["age"],
                    "gender": values["gender"],
                    "baseline_category": values["cat_rt"],
                    "entrainment_category": values["cat_entrain"],
                }
            )
    long = pd.DataFrame(rows).sort_values(["subject", "condition"]).reset_index(drop=True)
    long["entrainment_between"] = long.groupby("subject")["entrainment_uv2"].transform("mean")
    long["entrainment_within"] = long["entrainment_uv2"] - long["entrainment_between"]
    for column in (
        "entrainment_between",
        "entrainment_within",
        "baseline_rt_ms",
        "individual_freq_hz",
        "age_years",
    ):
        long[f"{column}_z"] = zscore(long[column])
    return long


def coefficient_table(model, model_name: str, terms: list[str]) -> pd.DataFrame:
    ci = model.conf_int(alpha=0.05)
    rows = []
    for term in terms:
        rows.append(
            {
                "model": model_name,
                "term": term,
                "estimate": model.params[term],
                "standard_error_cluster": model.bse[term],
                "ci95_low": ci.loc[term, 0],
                "ci95_high": ci.loc[term, 1],
                "p_value": model.pvalues[term],
                "n_observations": int(model.nobs),
                "n_subjects": int(model.model.data.frame["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def fit_association_models(long: pd.DataFrame) -> pd.DataFrame:
    common = (
        "C(condition) + entrainment_between_z + entrainment_within_z + "
        "baseline_rt_ms_z + age_years_z + C(gender)"
    )
    results = []
    for outcome in ("post_rt_ms", "rt_improvement_percent", "inverse_efficiency_ms"):
        model = smf.ols(f"{outcome} ~ {common}", data=long).fit(
            cov_type="cluster", cov_kwds={"groups": long["subject"]}
        )
        results.append(
            coefficient_table(
                model,
                f"continuous_{outcome}",
                [
                    "C(condition)[T.f_theta]",
                    "C(condition)[T.f_theta_plus]",
                    "entrainment_between_z",
                    "entrainment_within_z",
                    "baseline_rt_ms_z",
                ],
            )
        )

    fixed = smf.ols(
        "post_rt_ms ~ C(subject) + C(condition) + entrainment_within_z", data=long
    ).fit(cov_type="cluster", cov_kwds={"groups": long["subject"]})
    results.append(
        coefficient_table(
            fixed,
            "subject_fixed_effect_post_rt_ms",
            ["entrainment_within_z"],
        )
    )

    original_style = smf.ols(
        "rt_improvement_percent ~ C(condition) + C(baseline_category) + "
        "C(entrainment_category)",
        data=long,
    ).fit(cov_type="cluster", cov_kwds={"groups": long["subject"]})
    results.append(
        coefficient_table(
            original_style,
            "median_split_change_score_reproduction",
            ["C(baseline_category)[T.Poor]", "C(entrainment_category)[T.Low]"],
        )
    )
    return pd.concat(results, ignore_index=True)


def paired_summary(name: str, difference: pd.Series, scale: str) -> dict[str, object]:
    values = difference.to_numpy(dtype=float)
    n = len(values)
    estimate = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    critical = float(stats.t.ppf(0.975, n - 1))
    test = stats.ttest_1samp(values, 0)
    return {
        "contrast": name,
        "scale": scale,
        "n": n,
        "estimate": estimate,
        "standard_error": se,
        "ci95_low": estimate - critical * se,
        "ci95_high": estimate + critical * se,
        "p_value": float(test.pvalue),
        "paired_standardized_dz": estimate / float(values.std(ddof=1)),
    }


def build_paired_effects(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    for condition, columns in CONDITIONS.items():
        work[f"{condition}_accuracy"] = (
            work[columns["acc_con"]] + work[columns["acc_incon"]]
        ) / 2
        work[f"{condition}_ies"] = work[columns["rt"]] / work[f"{condition}_accuracy"]

    personalized_rt = (work["fθ_rt"] + work["fθ+_rt"]) / 2
    personalized_acc = (work["f_theta_accuracy"] + work["f_theta_plus_accuracy"]) / 2
    personalized_ies = (work["f_theta_ies"] + work["f_theta_plus_ies"]) / 2
    rows = [
        paired_summary("f_theta_minus_2Hz", work["fθ_rt"] - work["doshz_rt"], "ms"),
        paired_summary("f_theta_plus_minus_2Hz", work["fθ+_rt"] - work["doshz_rt"], "ms"),
        paired_summary("personalized_mean_minus_2Hz", personalized_rt - work["doshz_rt"], "ms"),
        paired_summary("f_theta_plus_minus_f_theta", work["fθ+_rt"] - work["fθ_rt"], "ms"),
        paired_summary(
            "personalized_mean_minus_2Hz_accuracy",
            personalized_acc - work["2Hz_accuracy"],
            "proportion_correct",
        ),
        paired_summary(
            "personalized_mean_minus_2Hz_inverse_efficiency",
            personalized_ies - work["2Hz_ies"],
            "ms_per_proportion_correct",
        ),
    ]
    return pd.DataFrame(rows)


def residualize(vector: np.ndarray, design: np.ndarray) -> np.ndarray:
    fitted = design @ np.linalg.lstsq(design, vector, rcond=None)[0]
    return vector - fitted


def within_subject_permutation(
    long: pd.DataFrame, *, iterations: int = 20_000, seed: int = 20260921
) -> dict[str, float | int]:
    """Permutation test for the subject- and condition-adjusted within effect."""
    subject_dummies = pd.get_dummies(long["subject"].astype(str), drop_first=False, dtype=float)
    condition_dummies = pd.get_dummies(long["condition"], drop_first=True, dtype=float)
    design = np.column_stack([subject_dummies.to_numpy(), condition_dummies.to_numpy()])
    y_resid = residualize(long["post_rt_ms"].to_numpy(float), design)
    x = long["entrainment_within_z"].to_numpy(float)
    x_resid = residualize(x, design)
    observed = float(np.dot(x_resid, y_resid) / np.dot(x_resid, x_resid))

    groups = [idx.to_numpy() for _, idx in long.groupby("subject").groups.items()]
    rng = np.random.default_rng(seed)
    permuted = np.empty(iterations)
    for iteration in range(iterations):
        x_perm = x.copy()
        for indices in groups:
            x_perm[indices] = rng.permutation(x_perm[indices])
        xr = residualize(x_perm, design)
        permuted[iteration] = np.dot(xr, y_resid) / np.dot(xr, xr)
    p_value = (np.sum(np.abs(permuted) >= abs(observed)) + 1) / (iterations + 1)
    return {
        "estimate_ms_per_within_sd": observed,
        "permutation_p_two_sided": float(p_value),
        "iterations": iterations,
        "seed": seed,
    }


def nested_ridge_predictions(features: pd.DataFrame, outcome: pd.Series) -> np.ndarray:
    numeric = [column for column in features.columns if column != "gender"]
    categorical = ["gender"] if "gender" in features.columns else []
    predictions = np.empty(len(outcome), dtype=float)
    alphas = (0.1, 1.0, 10.0, 100.0)
    for held_out in range(len(outcome)):
        train = np.arange(len(outcome)) != held_out
        x_train, y_train = features.loc[train], outcome.loc[train]
        transformer = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), categorical),
            ],
            remainder="drop",
        )
        inner = KFold(n_splits=5, shuffle=True, random_state=20260921 + held_out)
        best_alpha, best_score = None, np.inf
        for alpha in alphas:
            fold_errors = []
            for fit_idx, val_idx in inner.split(x_train):
                model = make_pipeline(transformer, Ridge(alpha=alpha))
                model.fit(x_train.iloc[fit_idx], y_train.iloc[fit_idx])
                predicted = model.predict(x_train.iloc[val_idx])
                fold_errors.append(mean_squared_error(y_train.iloc[val_idx], predicted))
            score = float(np.mean(fold_errors))
            if score < best_score:
                best_alpha, best_score = alpha, score
        final = make_pipeline(transformer, Ridge(alpha=best_alpha))
        final.fit(x_train, y_train)
        predictions[held_out] = final.predict(features.iloc[[held_out]])[0]
    return predictions


def prediction_gate(data: pd.DataFrame, *, bootstrap_iterations: int = 20_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome = ((data["fθ_rt"] + data["fθ+_rt"]) / 2 - data["doshz_rt"]).rename(
        "personalized_advantage_ms"
    )
    features = data[["individual_freq", "bl_rt", "age", "gender"]].rename(
        columns={
            "individual_freq": "individual_freq_hz",
            "bl_rt": "baseline_rt_ms",
            "age": "age_years",
        }
    )
    predictions = nested_ridge_predictions(features.reset_index(drop=True), outcome.reset_index(drop=True))
    observed = outcome.to_numpy(float)
    base = np.array(
        [(observed.sum() - observed[i]) / (len(observed) - 1) for i in range(len(observed))]
    )
    errors_base = np.abs(observed - base)
    errors_model = np.abs(observed - predictions)
    rng = np.random.default_rng(20260921)
    boot = np.empty(bootstrap_iterations)
    for iteration in range(bootstrap_iterations):
        indices = rng.integers(0, len(observed), len(observed))
        boot[iteration] = np.mean(errors_model[indices] - errors_base[indices])
    summary = pd.DataFrame(
        [
            {
                "outcome": outcome.name,
                "validation": "nested_leave_one_subject_out",
                "n_subjects": len(observed),
                "base_mae_ms": mean_absolute_error(observed, base),
                "model_mae_ms": mean_absolute_error(observed, predictions),
                "delta_mae_model_minus_base_ms": float(np.mean(errors_model - errors_base)),
                "delta_mae_ci95_low": float(np.quantile(boot, 0.025)),
                "delta_mae_ci95_high": float(np.quantile(boot, 0.975)),
                "base_rmse_ms": mean_squared_error(observed, base) ** 0.5,
                "model_rmse_ms": mean_squared_error(observed, predictions) ** 0.5,
                "prediction_outcome_r": stats.pearsonr(observed, predictions).statistic,
                "prediction_outcome_p": stats.pearsonr(observed, predictions).pvalue,
                "feature_timing": "pretreatment",
            }
        ]
    )
    subject_predictions = pd.DataFrame(
        {
            "subject": data["subject"].astype(int),
            "observed_personalized_advantage_ms": observed,
            "base_prediction_ms": base,
            "pretreatment_ridge_prediction_ms": predictions,
            "absolute_error_base_ms": errors_base,
            "absolute_error_model_ms": errors_model,
        }
    )
    return summary, subject_predictions


def moderator_checks(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    work["personalized_improvement_percent"] = (
        work["base_fθ_rt_percent"] + work["base_fθ+_rt_percent"]
    ) / 2
    work["personalized_advantage_vs_2Hz_ms"] = (
        work["fθ_rt"] + work["fθ+_rt"]
    ) / 2 - work["doshz_rt"]
    work["personalized_advantage_vs_NR_ms"] = (
        work["fθ_rt"] + work["fθ+_rt"]
    ) / 2 - work["nr_rt"]
    rows = []
    for outcome in (
        "personalized_improvement_percent",
        "personalized_advantage_vs_2Hz_ms",
        "personalized_advantage_vs_NR_ms",
    ):
        for predictor in ("bl_rt", "individual_freq", "mean_entrain"):
            result = stats.pearsonr(work[predictor], work[outcome])
            fisher = np.arctanh(np.clip(result.statistic, -0.999999, 0.999999))
            fisher_se = 1 / np.sqrt(len(work) - 3)
            rows.append(
                {
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": len(work),
                    "pearson_r": result.statistic,
                    "ci95_low": np.tanh(fisher - 1.96 * fisher_se),
                    "ci95_high": np.tanh(fisher + 1.96 * fisher_se),
                    "p_value": result.pvalue,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=20_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(args.input, sheet_name="Dataset")
    if data.shape != (44, 110) or data.isna().any().any():
        raise ValueError(f"Unexpected source shape/missingness: {data.shape}")
    long = make_long(data)
    associations = fit_association_models(long)
    paired = build_paired_effects(data)
    permutation = within_subject_permutation(long, iterations=args.permutations)
    prediction, subject_predictions = prediction_gate(data)
    moderators = moderator_checks(data)

    long.to_csv(args.output / "condition_level_long.csv", index=False)
    associations.to_csv(args.output / "association_models.csv", index=False)
    paired.to_csv(args.output / "paired_condition_effects.csv", index=False)
    prediction.to_csv(args.output / "pretreatment_prediction_summary.csv", index=False)
    subject_predictions.to_csv(args.output / "pretreatment_subject_predictions.csv", index=False)
    moderators.to_csv(args.output / "moderator_checks.csv", index=False)
    (args.output / "within_subject_permutation.json").write_text(
        json.dumps(permutation, indent=2), encoding="utf-8"
    )

    manifest = {
        "source": str(args.input.relative_to(ROOT)),
        "source_sha256": sha256(args.input),
        "n_subjects": int(data.subject.nunique()),
        "n_condition_rows": len(long),
        "conditions": list(CONDITIONS),
        "source_grain": "subject-by-condition summaries; not trials",
        "permutations": args.permutations,
        "claim_boundary": (
            "Association and prediction gate only. Raw BDF/event reanalysis is required "
            "for trial-level, reliability, order, and causal-mechanism claims."
        ),
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({**manifest, "within_subject": permutation}, indent=2))


if __name__ == "__main__":
    main()
