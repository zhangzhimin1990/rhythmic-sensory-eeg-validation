#!/usr/bin/env python3
"""Build matched reliability--validity evidence for the behavioral cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def bootstrap_spearman_ci(
    x: pd.Series, y: pd.Series, seed: int, n_boot: int = 10000
) -> tuple[float, float, float, int]:
    used = pd.DataFrame({"x": x, "y": y}).dropna().to_numpy()
    estimate = float(spearmanr(used[:, 0], used[:, 1]).statistic)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample = used[rng.integers(0, len(used), len(used))]
        candidate = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        if np.isfinite(candidate):
            boot.append(candidate)
    return estimate, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)), len(used)


def single_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    used = frame.copy()
    for column, value in filters.items():
        if isinstance(value, bool):
            used = used.loc[used[column].astype(str).str.lower().eq(str(value).lower())]
        else:
            used = used.loc[used[column].eq(value)]
    if len(used) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(used)}")
    return used.iloc[0]


def build_alignment(
    ds7_summary: pd.DataFrame,
    ds7_neural_reliability: pd.DataFrame,
    ds7_clustered: pd.DataFrame,
    ds6_neural_reliability: pd.DataFrame,
    ds6_models: pd.DataFrame,
    behavior_reliability: pd.DataFrame,
    n_boot: int = 10000,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ds7_summary = ds7_summary.loc[ds7_summary.n_trials_valid.gt(0)].copy()
    itpc40 = single_row(ds7_neural_reliability, measure="hz40_itpc")
    for offset, (outcome, label) in enumerate(
        (("accuracy_all_trials", "aggregate_accuracy"), ("median_correct_rt", "aggregate_correct_rt"))
    ):
        estimate, low, high, n = bootstrap_spearman_ci(
            ds7_summary.hz40_itpc, ds7_summary[outcome], 20260923 + offset, n_boot
        )
        outcome_rel = single_row(
            behavior_reliability,
            cohort="ds007648",
            measure="accuracy" if outcome == "accuracy_all_trials" else "median_correct_rt",
        )
        rows.append(
            {
                "dataset": "ds007648",
                "evidence_pair": f"aggregate_itpc_to_{label}",
                "neural_metric": "40-Hz aggregate ITPC",
                "neural_level": "between-participant aggregate",
                "neural_reliability": float(itpc40.spearman_brown_full_length),
                "outcome_metric": label,
                "outcome_level": "between-participant aggregate",
                "outcome_reliability": float(outcome_rel.spearman_brown_full_length),
                "validity_estimand": "Spearman rho",
                "validity_estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "p_value": float(spearmanr(ds7_summary.hz40_itpc, ds7_summary[outcome]).pvalue),
                "n_subjects": n,
                "alignment_status": "matched metric and level",
                "interpretation": "aggregate reliability and aggregate concurrent validity are directly comparable",
            }
        )

    for analysis, outcome, term in (
        ("accuracy", "trial_accuracy", "hz40_average_contrast"),
        ("rt", "correct_trial_log_rt", "hz40_average_contrast"),
    ):
        model = single_row(
            ds7_clustered,
            specification="main",
            feature_family="phase_projected",
            analysis=analysis,
            term=term,
        )
        rows.append(
            {
                "dataset": "ds007648",
                "evidence_pair": f"trial_phase_state_to_{outcome}",
                "neural_metric": "40-Hz trial phase-projected score",
                "neural_level": "within-participant trial",
                "neural_reliability": np.nan,
                "outcome_metric": outcome,
                "outcome_level": "within-participant trial",
                "outcome_reliability": np.nan,
                "validity_estimand": str(model.scale),
                "validity_estimate": float(model.estimate),
                "ci_low": float(model.ci_low),
                "ci_high": float(model.ci_high),
                "p_value": float(model.p_value),
                "n_subjects": int(model.n_subjects),
                "alignment_status": "validity level matched; reliability not established for trial-state predictor",
                "interpretation": "aggregate ITPC or aggregate behavior reliability must not be attached to this coefficient",
            }
        )

    ds6_itpc_rel = single_row(ds6_neural_reliability, frequency_hz=40)
    ds6_dprime_rel = single_row(
        behavior_reliability, cohort="ds006780", measure="dprime", frequency_hz=40.0
    )
    for family, term, neural_metric, status in (
        ("itpc", "z_itpc_40", "40-Hz aggregate ITPC", "matched metric and level"),
        (
            "local_log_snr",
            "z_local_log_snr_db_mean_40",
            "40-Hz aggregate local log-SNR",
            "level matched; neural reliability not estimated",
        ),
    ):
        model = single_row(
            ds6_models,
            outcome="dprime_40",
            feature_family=family,
            quadratic_age=False,
            group_interaction=False,
            term=term,
        )
        rows.append(
            {
                "dataset": "ds006780",
                "evidence_pair": f"{family}_to_dprime_40",
                "neural_metric": neural_metric,
                "neural_level": "between-participant aggregate",
                "neural_reliability": (
                    float(ds6_itpc_rel.spearman_brown_full_length) if family == "itpc" else np.nan
                ),
                "outcome_metric": "40-Hz d-prime",
                "outcome_level": "between-participant aggregate",
                "outcome_reliability": float(ds6_dprime_rel.spearman_brown_full_length),
                "validity_estimand": "adjusted coefficient per neural SD",
                "validity_estimate": float(model.estimate),
                "ci_low": float(model.ci_low),
                "ci_high": float(model.ci_high),
                "p_value": float(model.p_value),
                "n_subjects": int(model.n_subjects),
                "alignment_status": status,
                "interpretation": (
                    "neural and outcome reliability align with the adjusted validity model"
                    if family == "itpc"
                    else "outcome reliability aligns, but ITPC reliability cannot be transferred to local log-SNR"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/models/estimand_alignment"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    result = build_alignment(
        pd.read_csv("outputs/derived/ds007648_trial_features/subject_summary.csv"),
        pd.read_csv("outputs/models/ds007648_behavioral_validity/split_half_reliability.csv"),
        pd.read_csv("outputs/models/ds007648_behavioral_validity/clustered_models.csv"),
        pd.read_csv("outputs/models/ds006780_behavioral_validity/split_half_reliability.csv"),
        pd.read_csv("outputs/models/ds006780_behavioral_validity/model_coefficients.csv"),
        pd.read_csv("outputs/models/behavior_measurement_reliability/behavior_reliability.csv"),
        n_boot=args.bootstrap,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "matched_reliability_validity.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
