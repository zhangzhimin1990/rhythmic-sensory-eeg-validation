#!/usr/bin/env python3
"""Result-blind inferential completion audit for the Dryad confirmatory analysis.

This extension does not alter frozen participant-level derivation or the primary
group estimator.  It supplies uncertainty and sensitivity outputs promised by
the protocol but absent from the original group implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable

from scripts.analyze_dryad_confirmatory_group import (
    holm_adjust,
    paired_effect,
    participant_behaviour,
    spearman_brown,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBJECT_ROOT = ROOT / "outputs/models/dryad_confirmatory_subjects_v1"
DEFAULT_GROUP_ROOT = ROOT / "outputs/models/dryad_confirmatory_group_v1"
DEFAULT_OUTPUT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_v1"
DEFAULT_PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 20260927


def _finite_quantiles(values: list[float], probabilities=(0.025, 0.975)) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan
    return tuple(np.quantile(array, probabilities).tolist())


def bootstrap_split_reliability(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    values = frame[[left, right]].dropna().to_numpy(float)
    n = len(values)
    if n < 3:
        raise ValueError("at least three paired participants are required")
    rho = float(stats.spearmanr(values[:, 0], values[:, 1]).statistic)
    rng = np.random.default_rng(seed)
    boot_rho: list[float] = []
    for _ in range(iterations):
        sample = values[rng.integers(0, n, size=n)]
        boot_rho.append(float(stats.spearmanr(sample[:, 0], sample[:, 1]).statistic))
    rho_low, rho_high = _finite_quantiles(boot_rho)
    boot_sb = [spearman_brown(value) for value in boot_rho]
    sb_low, sb_high = _finite_quantiles(boot_sb)
    return {
        "n": n,
        "spearman_rho": rho,
        "spearman_rho_bootstrap_ci95_low": rho_low,
        "spearman_rho_bootstrap_ci95_high": rho_high,
        "spearman_brown": float(spearman_brown(rho)),
        "spearman_brown_bootstrap_ci95_low": sb_low,
        "spearman_brown_bootstrap_ci95_high": sb_high,
        "bootstrap_iterations": iterations,
    }


def reliability_uncertainty(conditions: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        "odd_even": ("odd_trial", "even_trial"),
        "time_half": ("first_half", "second_half"),
    }
    rows = []
    offset = 0
    for condition, frame in conditions.groupby("condition", sort=True):
        for metric in ("itpc", "evoked_local_log_snr_db"):
            for split, (left_root, right_root) in pairs.items():
                result = bootstrap_split_reliability(
                    frame,
                    f"{left_root}_{metric}",
                    f"{right_root}_{metric}",
                    seed=BOOTSTRAP_SEED + offset,
                )
                rows.append({"condition": condition, "metric": metric, "split": split, **result})
                offset += 1
    return pd.DataFrame(rows)


def prediction_increment_uncertainty(
    prediction_rows: pd.DataFrame,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED + 100,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required_models = {"baseline", "baseline_plus_neural"}
    if set(prediction_rows["model"]) != required_models:
        raise ValueError("prediction rows must contain exactly the two fitted models")
    wide = prediction_rows.pivot(index="subject", columns="model", values="predicted")
    observed_by_subject = prediction_rows.groupby("subject")["observed"].agg(["min", "max"])
    if not np.allclose(observed_by_subject["min"], observed_by_subject["max"]):
        raise ValueError("observed outcomes differ across prediction models")
    observed = observed_by_subject["min"].reindex(wide.index).to_numpy(float)
    baseline = wide["baseline"].to_numpy(float)
    neural = wide["baseline_plus_neural"].to_numpy(float)
    n = len(observed)
    training_mean = np.asarray([
        np.mean(np.delete(observed, index)) for index in range(n)
    ])
    comparators = {"training_fold_mean": training_mean, "baseline": baseline}
    rng = np.random.default_rng(seed)
    rows = []
    for name, comparator in comparators.items():
        mae_subject_delta = np.abs(comparator - observed) - np.abs(neural - observed)
        mae_improvement = float(mae_subject_delta.mean())
        rmse_improvement = float(
            np.sqrt(np.mean((comparator - observed) ** 2))
            - np.sqrt(np.mean((neural - observed) ** 2))
        )
        mae_boot, rmse_boot, rho_boot = [], [], []
        for _ in range(iterations):
            indices = rng.integers(0, n, size=n)
            y = observed[indices]
            c = comparator[indices]
            m = neural[indices]
            mae_boot.append(float(np.mean(np.abs(c - y) - np.abs(m - y))))
            rmse_boot.append(float(
                np.sqrt(np.mean((c - y) ** 2)) - np.sqrt(np.mean((m - y) ** 2))
            ))
            rho_c = stats.spearmanr(y, c).statistic
            rho_m = stats.spearmanr(y, m).statistic
            rho_boot.append(float(rho_m - rho_c))
        mae_low, mae_high = _finite_quantiles(mae_boot)
        rmse_low, rmse_high = _finite_quantiles(rmse_boot)
        rho_low, rho_high = _finite_quantiles(rho_boot)
        rows.append({
            "comparator": name,
            "n": n,
            "mae_improvement_comparator_minus_neural": mae_improvement,
            "mae_improvement_bootstrap_ci95_low": mae_low,
            "mae_improvement_bootstrap_ci95_high": mae_high,
            "rmse_improvement_comparator_minus_neural": rmse_improvement,
            "rmse_improvement_bootstrap_ci95_low": rmse_low,
            "rmse_improvement_bootstrap_ci95_high": rmse_high,
            "spearman_increment_neural_minus_comparator": float(
                stats.spearmanr(observed, neural).statistic
                - stats.spearmanr(observed, comparator).statistic
            ),
            "spearman_increment_bootstrap_ci95_low": rho_low,
            "spearman_increment_bootstrap_ci95_high": rho_high,
            "bootstrap_iterations": iterations,
        })
    result = pd.DataFrame(rows)
    unlocked = bool(
        result["mae_improvement_bootstrap_ci95_low"].gt(0).all()
        and result["rmse_improvement_comparator_minus_neural"].gt(0).all()
    )
    decision = {
        "prediction_increment_claim_unlocked": unlocked,
        "locked_rule": (
            "baseline-plus-neural must reduce MAE versus both comparators with both "
            "participant-bootstrap 95% lower bounds above zero, and reduce point RMSE "
            "versus both; Spearman increments are supporting estimates"
        ),
        "if_not_unlocked": "no consistent out-of-sample predictive increment",
    }
    return result, decision


def roi_channel_sensitivity(conditions: pd.DataFrame) -> pd.DataFrame:
    data = conditions.copy()
    data["n_roi_channels"] = data["frontocentral_roi"].fillna("").map(
        lambda value: len([item for item in str(value).split(";") if item])
    )
    keep = data.groupby("subject")["n_roi_channels"].min()
    keep = keep.index[keep.ge(4)]
    subset = data.loc[data["subject"].isin(keep)]
    result = paired_effect(subset, "mean_itpc_in_stimulation_window", BOOTSTRAP_SEED + 200)
    result["sensitivity_population"] = "participants_with_at_least_four_good_frontocentral_roi_channels"
    return pd.DataFrame([result])


def behaviour_by_congruency(trials: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for congruency, frame in trials.groupby("congruency", sort=True):
        participant = participant_behaviour(frame)
        for index, metric in enumerate(("accuracy", "median_correct_rt_ms", "inverse_efficiency_ms")):
            effect = paired_effect(participant, metric, BOOTSTRAP_SEED + 300 + len(parts) + index)
            parts.append({"congruency": congruency, **effect})
    result = pd.DataFrame(parts)
    result["p_value_holm_sensitivity_family"] = holm_adjust(
        result["p_value_t_two_sided"].to_numpy()
    )
    result["interpretation"] = "prespecified_sensitivity_not_a_replacement_primary_endpoint"
    return result


def _linear_combination(
    estimates: pd.Series,
    covariance: pd.DataFrame,
    weights: dict[str, float],
) -> tuple[float, float, float, float, float]:
    names = list(weights)
    vector = np.asarray([weights[name] for name in names], dtype=float)
    estimate = float(sum(weights[name] * estimates[name] for name in names))
    subcovariance = covariance.loc[names, names].to_numpy(float)
    standard_error = float(np.sqrt(vector @ subcovariance @ vector))
    z_value = estimate / standard_error if standard_error else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return (
        estimate,
        standard_error,
        estimate - 1.96 * standard_error,
        estimate + 1.96 * standard_error,
        p_value,
    )


def coupling_simple_slopes(trials: pd.DataFrame) -> pd.DataFrame:
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
    fitted = [
        (
            "correctness_binomial_gee",
            smf.gee(
                formula.format(outcome="correct_binary"), "subject", data,
                cov_struct=Exchangeable(), family=sm.families.Binomial(),
            ).fit(),
        )
    ]
    correct = data.loc[data["correct_binary"].eq(1) & data["rt_ms"].notna()].copy()
    correct["log_rt_ms"] = np.log(correct["rt_ms"].astype(float))
    fitted.append((
        "correct_rt_gaussian_gee",
        smf.gee(
            formula.format(outcome="log_rt_ms"), "subject", correct,
            cov_struct=Exchangeable(), family=sm.families.Gaussian(),
        ).fit(),
    ))
    main = "phase_alignment_within"
    interaction = "phase_alignment_within:C(condition)[T.f_theta_plus]"
    rows = []
    for model_name, model in fitted:
        for condition, weights in (
            ("non_rhythmic", {main: 1.0}),
            ("f_theta_plus", {main: 1.0, interaction: 1.0}),
        ):
            estimate, se, low, high, p_value = _linear_combination(
                model.params, model.cov_params(), weights
            )
            rows.append({
                "model": model_name,
                "condition": condition,
                "simple_slope_phase_alignment_within": estimate,
                "standard_error": se,
                "ci95_low": low,
                "ci95_high": high,
                "p_value_two_sided": p_value,
                "n_trials": int(model.nobs),
                "n_subjects": int(model.model.data.frame["subject"].nunique()),
                "interpretation": "concurrent_within_subject_condition_association_not_mediation",
            })
    result = pd.DataFrame(rows)
    result["p_value_holm_simple_slope_family"] = holm_adjust(
        result["p_value_two_sided"].to_numpy()
    )
    return result


def load_subject_inputs(subject_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_parts, trial_parts = [], []
    for directory in sorted(subject_root.glob("S*")):
        condition_parts.append(pd.read_csv(directory / "condition_neural_metrics.csv"))
        trial_parts.append(pd.read_csv(directory / "trial_neural_behaviour.csv"))
    if not condition_parts:
        raise ValueError("no participant-level confirmatory inputs found")
    return pd.concat(condition_parts, ignore_index=True), pd.concat(trial_parts, ignore_index=True)


def run(
    subject_root: Path,
    group_root: Path,
    protocol_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    conditions, trials = load_subject_inputs(subject_root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = set(map(int, protocol["confirmatory_subjects"]))
    if set(conditions["subject"].astype(int)) != expected or set(trials["subject"].astype(int)) != expected:
        raise ValueError("inferential audit requires the exact completed 35-person cohort")
    condition_counts = conditions.groupby(["subject", "condition"]).size()
    if len(condition_counts) != 2 * len(expected) or not condition_counts.eq(1).all():
        raise ValueError("condition metrics must contain one row per subject and active-contrast condition")
    prediction_rows = pd.read_csv(group_root / "nested_prediction_rows.csv")
    if set(prediction_rows["subject"].astype(int)) != expected:
        raise ValueError("prediction rows do not match the exact locked cohort")

    reliability = reliability_uncertainty(conditions)
    prediction, prediction_decision = prediction_increment_uncertainty(prediction_rows)
    roi_sensitivity = roi_channel_sensitivity(conditions)
    behaviour_sensitivity = behaviour_by_congruency(trials)
    coupling_slopes = coupling_simple_slopes(trials)

    output_dir.mkdir(parents=True, exist_ok=True)
    reliability.to_csv(output_dir / "internal_consistency_bootstrap.csv", index=False)
    prediction.to_csv(output_dir / "prediction_increment_bootstrap.csv", index=False)
    roi_sensitivity.to_csv(output_dir / "primary_roi_channel_sensitivity.csv", index=False)
    behaviour_sensitivity.to_csv(output_dir / "behaviour_congruency_sensitivity.csv", index=False)
    coupling_slopes.to_csv(output_dir / "within_condition_coupling_simple_slopes.csv", index=False)
    summary = {
        "status": "confirmatory_inferential_completion_audit_complete",
        "n_subjects": 35,
        **prediction_decision,
        "primary_estimator_changed": False,
        "participant_level_derivation_changed": False,
    }
    (output_dir / "inference_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, default=DEFAULT_SUBJECT_ROOT)
    parser.add_argument("--group-root", type=Path, default=DEFAULT_GROUP_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.subject_root, args.group_root, args.protocol, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
