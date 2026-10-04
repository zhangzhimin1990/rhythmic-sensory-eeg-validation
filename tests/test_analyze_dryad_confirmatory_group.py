import unittest

import numpy as np
import pandas as pd

from scripts.analyze_dryad_confirmatory_group import (
    bootstrap_mean_ci,
    holm_adjust,
    paired_effect,
    participant_behaviour,
    spearman_brown,
)


class DryadConfirmatoryGroupTests(unittest.TestCase):
    def test_holm_adjust_is_monotone_in_rank_and_bounded(self):
        raw = np.asarray([0.04, 0.001, 0.02])
        adjusted = holm_adjust(raw)
        order = np.argsort(raw)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= 0))
        self.assertTrue(np.all((adjusted >= raw) & (adjusted <= 1)))

    def test_paired_effect_uses_theta_plus_minus_non_rhythmic(self):
        rows = []
        for subject in range(1, 7):
            rows.extend([
                {"subject": subject, "condition": "non_rhythmic", "metric": subject},
                {"subject": subject, "condition": "f_theta_plus", "metric": subject + 2},
            ])
        result = paired_effect(pd.DataFrame(rows), "metric", 7)
        self.assertAlmostEqual(result["mean_difference_theta_plus_minus_non_rhythmic"], 2.0)
        self.assertEqual(result["direction_fraction_positive"], 1.0)

    def test_bootstrap_interval_is_deterministic(self):
        values = np.asarray([-1.0, 0.0, 1.0, 2.0])
        self.assertEqual(bootstrap_mean_ci(values, iterations=100, seed=9), bootstrap_mean_ci(values, iterations=100, seed=9))

    def test_participant_behaviour_keeps_omissions_in_accuracy(self):
        trials = pd.DataFrame({
            "subject": [1, 1, 1], "condition": ["f_theta_plus"] * 3,
            "correct": [True, False, False], "rt_ms": [400.0, 500.0, np.nan],
        })
        result = participant_behaviour(trials).iloc[0]
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["median_correct_rt_ms"], 400.0)
        self.assertEqual(result["inverse_efficiency_ms"], 1200.0)

    def test_participant_behaviour_does_not_filter_on_neural_usability(self):
        trials = pd.DataFrame({
            "subject": [1, 1], "condition": ["f_theta_plus", "f_theta_plus"],
            "correct": [True, False], "rt_ms": [400.0, np.nan],
            "neural_usable": [True, False],
        })
        result = participant_behaviour(trials).iloc[0]
        self.assertEqual(result["accuracy"], 0.5)

    def test_spearman_brown_identity(self):
        self.assertAlmostEqual(spearman_brown(0.5), 2 / 3)


if __name__ == "__main__":
    unittest.main()
