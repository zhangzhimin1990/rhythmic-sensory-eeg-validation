import numpy as np
import pandas as pd

from scripts.audit_hbm_reviewer_uncertainty import (
    bootstrap_split_half,
    percentile_mean_interval,
    prediction_intervals,
    spearman_brown,
)


def test_percentile_mean_interval_is_seed_reproducible_and_contains_point():
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    first = percentile_mean_interval(values, iterations=2000, seed=7)
    second = percentile_mean_interval(values, iterations=2000, seed=7)
    assert first == second
    assert first["bootstrap_ci95_low"] <= first["mean_delta_neural_minus_base"]
    assert first["bootstrap_ci95_high"] >= first["mean_delta_neural_minus_base"]


def test_split_half_reliability_recovers_perfect_rank_order():
    first = np.arange(1.0, 11.0)
    second = first * 3.0
    result = bootstrap_split_half(first, second, iterations=1000, seed=9)
    assert np.isclose(result["split_half_spearman_rho"], 1.0)
    assert np.isclose(result["spearman_brown_full_length"], 1.0)
    assert np.isclose(result["sb_bootstrap_ci95_low"], 1.0)


def test_prediction_interval_direction_is_neural_minus_base():
    frame = pd.DataFrame({
        "dataset": ["x"] * 4,
        "outcome": ["y"] * 4,
        "subject": ["a", "b", "c", "d"],
        "metric": ["absolute_error"] * 4,
        "base": [2.0] * 4,
        "neural": [1.0] * 4,
        "delta_neural_minus_base": [-1.0] * 4,
    })
    result = prediction_intervals(frame, iterations=1000, seed=11).iloc[0]
    assert result.mean_delta_neural_minus_base == -1.0
    assert result.fraction_below_zero == 1.0


def test_spearman_brown_known_value():
    assert np.isclose(spearman_brown(0.5), 2 / 3)
