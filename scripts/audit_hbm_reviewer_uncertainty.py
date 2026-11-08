#!/usr/bin/env python3
"""Quantify uncertainty requested by the blinded HBM mock review.

This script does not refit or select any prediction model. It resamples the
already frozen participant-level outputs to provide effect-uncertainty
intervals for prediction loss increments and split-half reliability. The
analysis is explicitly post-review robustness work, not confirmatory work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, skew


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTION = ROOT / "outputs/models/hbm_prediction_audit_v1/participant_loss_increments.csv"
DEFAULT_DS7_HALVES = ROOT / "outputs/models/ds007648_behavioral_validity/split_half_values.csv"
DEFAULT_DS6_FEATURES = ROOT / "outputs/models/ds006780_behavioral_validity/subject_level_features.csv"
DEFAULT_OUTPUT = ROOT / "outputs/models/hbm_reviewer_uncertainty_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_mean_interval(values: np.ndarray, iterations: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("values must be a finite vector with at least three observations")
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 5000):
        stop = min(start + 5000, iterations)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean_delta_neural_minus_base": float(values.mean()),
        "bootstrap_ci95_low": float(np.quantile(means, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(means, 0.975)),
        "median_delta_neural_minus_base": float(np.median(values)),
        "skewness": float(skew(values, bias=False)) if np.ptp(values) > 0 else 0.0,
        "fraction_below_zero": float(np.mean(values < 0)),
    }


def spearman_brown(rho: float) -> float:
    return float(2 * rho / (1 + rho)) if np.isfinite(rho) and rho > -1 else np.nan


def bootstrap_split_half(
    first: np.ndarray, second: np.ndarray, iterations: int, seed: int
) -> dict[str, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    keep = np.isfinite(first) & np.isfinite(second)
    first, second = first[keep], second[keep]
    if len(first) < 5:
        raise ValueError("at least five complete participant pairs are required")
    rho = float(spearmanr(first, second).statistic)
    point = spearman_brown(rho)
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=float)
    for item in range(iterations):
        indices = rng.integers(0, len(first), size=len(first))
        sampled_rho = float(spearmanr(first[indices], second[indices]).statistic)
        boot[item] = spearman_brown(sampled_rho)
    boot = boot[np.isfinite(boot)]
    if len(boot) < 0.99 * iterations:
        raise RuntimeError("too many undefined reliability bootstrap samples")
    return {
        "n_subjects": int(len(first)),
        "split_half_spearman_rho": rho,
        "spearman_brown_full_length": point,
        "sb_bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
        "sb_bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
    }


def prediction_intervals(frame: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rows = []
    for index, ((dataset, outcome, metric), group) in enumerate(
        frame.groupby(["dataset", "outcome", "metric"], sort=True)
    ):
        result = percentile_mean_interval(
            group["delta_neural_minus_base"].to_numpy(float), iterations, seed + index
        )
        rows.append({
            "dataset": dataset,
            "outcome": outcome,
            "metric": metric,
            "n_subjects": int(len(group)),
            **result,
            "bootstrap_iterations": iterations,
            "interpretation": "negative values favor the neural-feature model",
        })
    return pd.DataFrame(rows)


def reliability_intervals(
    ds7_halves: pd.DataFrame, ds6_features: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    rows = []
    wide = ds7_halves.pivot(index="subject", columns="half")
    for offset, (frequency, column) in enumerate(((36, "hz36_itpc"), (40, "hz40_itpc"))):
        result = bootstrap_split_half(
            wide[column][0].to_numpy(float), wide[column][1].to_numpy(float),
            iterations, seed + offset,
        )
        rows.append({"dataset": "ds007648", "frequency_hz": frequency,
                     "measure": "aggregate_itpc", **result,
                     "bootstrap_iterations": iterations})
    usable = ds6_features.loc[ds6_features["technical_usable"].astype(bool)].copy()
    for offset, frequency in enumerate((27, 40), start=2):
        result = bootstrap_split_half(
            usable[f"itpc_odd_trials_{frequency}"].to_numpy(float),
            usable[f"itpc_even_trials_{frequency}"].to_numpy(float),
            iterations, seed + offset,
        )
        rows.append({"dataset": "ds006780", "frequency_hz": frequency,
                     "measure": "aggregate_itpc", **result,
                     "bootstrap_iterations": iterations})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--ds7-halves", type=Path, default=DEFAULT_DS7_HALVES)
    parser.add_argument("--ds6-features", type=Path, default=DEFAULT_DS6_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    if args.iterations < 1000:
        raise ValueError("at least 1000 bootstrap iterations are required")
    args.output.mkdir(parents=True, exist_ok=True)
    prediction = pd.read_csv(args.prediction)
    ds7 = pd.read_csv(args.ds7_halves)
    ds6 = pd.read_csv(args.ds6_features)
    prediction_output = prediction_intervals(prediction, args.iterations, args.seed)
    reliability_output = reliability_intervals(ds7, ds6, args.iterations, args.seed + 100)
    prediction_path = args.output / "prediction_increment_bootstrap.csv"
    reliability_path = args.output / "matched_itpc_reliability_bootstrap.csv"
    prediction_output.to_csv(prediction_path, index=False)
    reliability_output.to_csv(reliability_path, index=False)
    manifest = {
        "analysis_id": "hbm_reviewer_uncertainty_audit_v1",
        "classification": "post-review robustness analysis, not confirmatory",
        "purpose": "quantify effect uncertainty without refitting or selecting prediction models",
        "bootstrap": {"method": "participant percentile bootstrap", "iterations": args.iterations,
                      "seed": args.seed, "interval": 0.95},
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (args.prediction, args.ds7_halves, args.ds6_features)
        },
        "outputs": {
            str(prediction_path.relative_to(ROOT)): sha256_file(prediction_path),
            str(reliability_path.relative_to(ROOT)): sha256_file(reliability_path),
        },
        "claim_boundary": (
            "Intervals quantify uncertainty in frozen participant-level summaries. "
            "They do not convert absence of support into proof of no effect."
        ),
    }
    manifest_path = args.output / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prediction_rows": len(prediction_output),
                      "reliability_rows": len(reliability_output),
                      "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
