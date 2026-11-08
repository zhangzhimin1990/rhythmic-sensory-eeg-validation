import numpy as np
import pandas as pd

from scripts.analyze_hbm_prediction_audit import (
    loss_audit,
    participant_losses,
    sign_flip_test,
)


def test_participant_losses_and_increment_direction():
    rows = []
    for subject, observed in (("s1", 1.0), ("s2", 2.0), ("s3", 3.0)):
        rows.extend(
            [
                {"dataset": "x", "outcome": "continuous", "subject": subject,
                 "model": "base", "observed": observed, "predicted": observed + 1.0},
                {"dataset": "x", "outcome": "continuous", "subject": subject,
                 "model": "neural", "observed": observed, "predicted": observed + 0.25},
            ]
        )
    losses = participant_losses(pd.DataFrame(rows))
    increments, summary = loss_audit(losses, permutations=1000, seed=17)
    assert np.allclose(increments.delta_neural_minus_base, -0.75)
    assert summary.loc[0, "fraction_subjects_improved"] == 1.0


def test_binary_loss_is_computed_within_held_out_participant():
    predictions = pd.DataFrame(
        {
            "dataset": ["x"] * 8,
            "outcome": ["accuracy"] * 8,
            "subject": ["s1"] * 8,
            "model": ["base"] * 4 + ["neural"] * 4,
            "observed": [0, 0, 1, 1] * 2,
            "predicted": [0.4, 0.4, 0.6, 0.6, 0.2, 0.2, 0.8, 0.8],
        }
    )
    losses = participant_losses(predictions).set_index("model")
    assert losses.loc["neural", "loss"] < losses.loc["base", "loss"]


def test_sign_flip_is_seed_reproducible_and_two_sided():
    values = np.array([-1.0, -0.5, -0.25, -0.125])
    first = sign_flip_test(values, permutations=2000, seed=91)
    second = sign_flip_test(values, permutations=2000, seed=91)
    assert first == second
    assert first["observed_mean_delta"] < 0
    assert 0 < first["p_value_two_sided"] <= 1
