#!/usr/bin/env python3
"""Group analysis for the locked 35-person Dryad confirmatory cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.genmod.cov_struct import Exchangeable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
DEFAULT_INPUT = ROOT / "outputs/models/dryad_confirmatory_subjects_v1"
DEFAULT_METADATA = ROOT / "data/public/dryad_t76hdr8dm/v5/metadata/Dataset.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs/models/dryad_confirmatory_group_v1"
NEURAL_METRICS = (
    "mean_itpc_in_stimulation_window",
    "evoked_local_log_snr_db",
    "total_power_local_log_snr_db",
    "induced_local_log_snr_db",
)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted


def bootstrap_mean_ci(
    values: np.ndarray, iterations: int = 20_000, seed: int = 20260924
) -> tuple[float, float]:
    vector = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(vector, size=(iterations, len(vector)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def paired_effect(values: pd.DataFrame, metric: str, seed: int) -> dict[str, object]:
    wide = values.pivot(index="subject", columns="condition", values=metric)
    if wide.isna().any().any() or set(wide.columns) != {"f_theta_plus", "non_rhythmic"}:
        raise ValueError(f"incomplete paired condition values for {metric}")
    difference = (wide["f_theta_plus"] - wide["non_rhythmic"]).to_numpy(float)
    n = len(difference)
    estimate = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    se = sd / np.sqrt(n)
    critical = float(stats.t.ppf(0.975, n - 1))
    boot_low, boot_high = bootstrap_mean_ci(difference, seed=seed)
    signed = stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
    equivalence_bound_dz = 0.3
    equivalence_bound_raw = equivalence_bound_dz * sd
    critical90 = float(stats.t.ppf(0.95, n - 1))
    ci90_low, ci90_high = estimate - critical90 * se, estimate + critical90 * se
    return {
        "metric": metric,
        "n": n,
        "mean_difference_theta_plus_minus_non_rhythmic": estimate,
        "ci95_low": estimate - critical * se,
        "ci95_high": estimate + critical * se,
        "paired_dz": estimate / sd if sd else np.nan,
        "direction_fraction_positive": float(np.mean(difference > 0)),
        "p_value_t_two_sided": float(stats.ttest_1samp(difference, 0).pvalue),
        "p_value_wilcoxon_two_sided": float(signed.pvalue),
        "ci90_low": ci90_low,
        "ci90_high": ci90_high,
        "equivalence_bound_raw_plus_minus": equivalence_bound_raw,
        "equivalent_within_plus_minus_dz_0_3": bool(
            ci90_low > -equivalence_bound_raw and ci90_high < equivalence_bound_raw
        ),
        "bootstrap_ci95_low": boot_low,
        "bootstrap_ci95_high": boot_high,
    }


def spearman_brown(correlation: float) -> float:
    return 2.0 * correlation / (1.0 + correlation) if correlation > -1 else np.nan


def reliability_rows(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = {
        "odd_even": ("odd_trial", "even_trial"),
        "time_half": ("first_half", "second_half"),
    }
    for condition, frame in values.groupby("condition"):
        for metric_root in ("itpc", "evoked_local_log_snr_db"):
            for split, (left, right) in pairs.items():
                left_column = f"{left}_{metric_root}" if metric_root == "itpc" else f"{left}_{metric_root}"
                right_column = f"{right}_{metric_root}" if metric_root == "itpc" else f"{right}_{metric_root}"
                rho, p_value = stats.spearmanr(frame[left_column], frame[right_column])
                rows.append({
                    "condition": condition,
                    "metric": metric_root,
                    "split": split,
                    "n": len(frame),
                    "spearman_rho": float(rho),
                    "spearman_brown": float(spearman_brown(float(rho))),
                    "p_value_two_sided": float(p_value),
                    "interpretation": "internal_consistency_not_test_retest",
                })
    return pd.DataFrame(rows)


def participant_behaviour(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject, condition), frame in trials.groupby(["subject", "condition"]):
        accuracy = float(frame["correct"].astype(bool).mean())
        correct_rt = frame.loc[frame["correct"].astype(bool), "rt_ms"].dropna().astype(float)
        median_rt = float(correct_rt.median()) if len(correct_rt) else np.nan
        rows.append({
            "subject": int(subject),
            "condition": condition,
            "accuracy": accuracy,
            "median_correct_rt_ms": median_rt,
            "inverse_efficiency_ms": median_rt / accuracy if accuracy > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def gee_coupling(trials: pd.DataFrame) -> pd.DataFrame:
    data = trials.loc[
        trials["neural_usable"].astype(bool) & trials["phase_alignment_loo"].notna()
    ].copy()
    data["condition"] = pd.Categorical(
        data["condition"], categories=["non_rhythmic", "f_theta_plus"]
    )
    data["phase_alignment_within"] = data["phase_alignment_loo"] - data.groupby(
        ["subject", "condition"], observed=True
    )["phase_alignment_loo"].transform("mean")
    data["correct_binary"] = data["correct"].astype(int)
    formula = "{outcome} ~ phase_alignment_within * C(condition) + C(congruency)"
    models = [
        (
            "correctness_binomial_gee",
            smf.gee(
                formula.format(outcome="correct_binary"), "subject", data,
                cov_struct=Exchangeable(), family=sm.families.Binomial()
            ).fit(),
        )
    ]
    correct = data.loc[data["correct_binary"].eq(1) & data["rt_ms"].notna()].copy()
    correct["log_rt_ms"] = np.log(correct["rt_ms"].astype(float))
    models.append(
        (
            "correct_rt_gaussian_gee",
            smf.gee(
                formula.format(outcome="log_rt_ms"), "subject", correct,
                cov_struct=Exchangeable(), family=sm.families.Gaussian()
            ).fit(),
        )
    )
    terms = [
        "phase_alignment_within",
        "phase_alignment_within:C(condition)[T.f_theta_plus]",
    ]
    rows = []
    for name, model in models:
        intervals = model.conf_int()
        for term in terms:
            rows.append({
                "model": name,
                "term": term,
                "estimate": float(model.params[term]),
                "standard_error": float(model.bse[term]),
                "ci95_low": float(intervals.loc[term, 0]),
                "ci95_high": float(intervals.loc[term, 1]),
                "p_value_two_sided": float(model.pvalues[term]),
                "n_trials": int(model.nobs),
                "n_subjects": int(model.model.data.frame["subject"].nunique()),
            })
    result = pd.DataFrame(rows)
    result["p_value_holm_within_family"] = holm_adjust(result["p_value_two_sided"].to_numpy())
    return result


def nested_predictions(features: pd.DataFrame, outcome: np.ndarray) -> pd.DataFrame:
    alphas = (0.1, 1.0, 10.0, 100.0)
    rows = []
    feature_sets = {
        "baseline": ["individual_freq", "bl_rt", "age", "gender"],
        "baseline_plus_neural": [
            "individual_freq", "bl_rt", "age", "gender",
            "primary_itpc_contrast", "evoked_snr_contrast", "internal_consistency",
        ],
    }
    for model_name, columns in feature_sets.items():
        x = features[columns].reset_index(drop=True)
        numeric = [column for column in columns if column != "gender"]
        predictions = np.empty(len(outcome), dtype=float)
        for held_out in range(len(outcome)):
            train = np.arange(len(outcome)) != held_out
            transformer = ColumnTransformer(
                [
                    ("numeric", StandardScaler(), numeric),
                    ("gender", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), ["gender"]),
                ]
            )
            inner = KFold(n_splits=5, shuffle=True, random_state=20260924 + held_out)
            best_alpha, best_error = None, np.inf
            for alpha in alphas:
                errors = []
                for fit_index, validation_index in inner.split(x.loc[train]):
                    x_train = x.loc[train].reset_index(drop=True)
                    y_train = outcome[train]
                    model = make_pipeline(transformer, Ridge(alpha=alpha))
                    model.fit(x_train.iloc[fit_index], y_train[fit_index])
                    errors.append(mean_absolute_error(y_train[validation_index], model.predict(x_train.iloc[validation_index])))
                if np.mean(errors) < best_error:
                    best_alpha, best_error = alpha, float(np.mean(errors))
            final = make_pipeline(transformer, Ridge(alpha=best_alpha))
            final.fit(x.loc[train], outcome[train])
            predictions[held_out] = final.predict(x.iloc[[held_out]])[0]
        for subject, observed, predicted in zip(features["subject"], outcome, predictions):
            rows.append({
                "subject": int(subject), "model": model_name,
                "observed": float(observed), "predicted": float(predicted),
            })
    return pd.DataFrame(rows)


def run(input_root: Path, metadata_path: Path, protocol_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = set(map(int, protocol["confirmatory_subjects"]))
    condition_parts, trial_parts = [], []
    for directory in sorted(input_root.glob("S*")):
        condition_parts.append(pd.read_csv(directory / "condition_neural_metrics.csv"))
        trial_parts.append(pd.read_csv(directory / "trial_neural_behaviour.csv"))
    conditions = pd.concat(condition_parts, ignore_index=True)
    trials = pd.concat(trial_parts, ignore_index=True)
    if set(conditions["subject"].astype(int)) != expected or set(trials["subject"].astype(int)) != expected:
        raise ValueError("group inputs are not the exact locked confirmatory cohort")
    if conditions.groupby("subject")["condition"].nunique().ne(2).any():
        raise ValueError("each confirmatory participant must have both active-contrast conditions")

    neural_effects = pd.DataFrame([
        paired_effect(conditions, metric, 20260924 + index)
        for index, metric in enumerate(NEURAL_METRICS)
    ])
    secondary = neural_effects["metric"].ne("mean_itpc_in_stimulation_window")
    neural_effects.loc[secondary, "p_value_holm_key_secondary"] = holm_adjust(
        neural_effects.loc[secondary, "p_value_t_two_sided"].to_numpy()
    )
    behaviour = participant_behaviour(trials)
    behaviour_effects = pd.DataFrame([
        paired_effect(behaviour, metric, 20261000 + index)
        for index, metric in enumerate(("accuracy", "median_correct_rt_ms", "inverse_efficiency_ms"))
    ])
    behaviour_effects["p_value_holm_behaviour"] = holm_adjust(
        behaviour_effects["p_value_t_two_sided"].to_numpy()
    )
    reliability = reliability_rows(conditions)
    coupling = gee_coupling(trials)

    metadata = pd.read_excel(metadata_path, sheet_name="Dataset")
    metadata = metadata.loc[metadata["subject"].isin(expected)].copy()
    neural_wide = conditions.pivot(index="subject", columns="condition")
    prediction = metadata[["subject", "individual_freq", "bl_rt", "age", "gender"]].copy()
    prediction["primary_itpc_contrast"] = prediction["subject"].map(
        neural_wide["mean_itpc_in_stimulation_window"]["f_theta_plus"]
        - neural_wide["mean_itpc_in_stimulation_window"]["non_rhythmic"]
    )
    prediction["evoked_snr_contrast"] = prediction["subject"].map(
        neural_wide["evoked_local_log_snr_db"]["f_theta_plus"]
        - neural_wide["evoked_local_log_snr_db"]["non_rhythmic"]
    )
    split_values = conditions.set_index(["subject", "condition"])[
        ["odd_trial_itpc", "even_trial_itpc"]
    ]
    # A higher value means closer odd/even agreement; unlike their mean, this
    # feature does not duplicate the participant's primary ITPC level.
    reliability_index = (
        -(split_values["odd_trial_itpc"] - split_values["even_trial_itpc"]).abs()
    ).groupby("subject").mean()
    prediction["internal_consistency"] = prediction["subject"].map(reliability_index)
    behaviour_wide = behaviour.pivot(index="subject", columns="condition")
    outcome = prediction["subject"].map(
        behaviour_wide["inverse_efficiency_ms"]["f_theta_plus"]
        - behaviour_wide["inverse_efficiency_ms"]["non_rhythmic"]
    ).to_numpy(float)
    predictions = nested_predictions(prediction, outcome)
    performance_rows = []
    baseline_loo = np.asarray([
        np.mean(np.delete(outcome, index)) for index in range(len(outcome))
    ])
    performance_rows.append({
        "model": "training_fold_mean", "mae": mean_absolute_error(outcome, baseline_loo),
        "rmse": mean_squared_error(outcome, baseline_loo) ** 0.5,
        "spearman_rho": stats.spearmanr(outcome, baseline_loo).statistic,
    })
    for model, frame in predictions.groupby("model"):
        performance_rows.append({
            "model": model,
            "mae": mean_absolute_error(frame["observed"], frame["predicted"]),
            "rmse": mean_squared_error(frame["observed"], frame["predicted"]) ** 0.5,
            "spearman_rho": stats.spearmanr(frame["observed"], frame["predicted"]).statistic,
        })
    performance = pd.DataFrame(performance_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    neural_effects.to_csv(output_dir / "paired_neural_effects.csv", index=False)
    reliability.to_csv(output_dir / "internal_consistency.csv", index=False)
    behaviour.to_csv(output_dir / "participant_behaviour.csv", index=False)
    behaviour_effects.to_csv(output_dir / "paired_behaviour_effects.csv", index=False)
    coupling.to_csv(output_dir / "within_subject_coupling.csv", index=False)
    predictions.to_csv(output_dir / "nested_prediction_rows.csv", index=False)
    performance.to_csv(output_dir / "prediction_performance.csv", index=False)
    primary = neural_effects.loc[
        neural_effects["metric"].eq("mean_itpc_in_stimulation_window")
    ].iloc[0]
    summary = {
        "protocol_id": protocol["protocol_id"],
        "n_confirmatory_subjects": len(expected),
        "primary_metric": primary["metric"],
        "primary_estimate": float(primary["mean_difference_theta_plus_minus_non_rhythmic"]),
        "primary_ci95": [float(primary["ci95_low"]), float(primary["ci95_high"])],
        "primary_p_value_two_sided": float(primary["p_value_t_two_sided"]),
        "claim_scope": "acute_target_engagement_and_concurrent_validity_not_treatment_efficacy",
        "status": "confirmatory_group_analysis_complete",
    }
    (output_dir / "confirmatory_group_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.input_root, args.metadata, args.protocol, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
