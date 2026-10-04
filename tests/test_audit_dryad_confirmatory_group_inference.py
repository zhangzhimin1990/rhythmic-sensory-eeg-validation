import unittest

import numpy as np
import pandas as pd

from scripts.audit_dryad_confirmatory_group_inference import (
    _linear_combination,
    behaviour_by_congruency,
    bootstrap_split_reliability,
    prediction_increment_uncertainty,
    roi_channel_sensitivity,
)


class DryadConfirmatoryInferenceAuditTests(unittest.TestCase):
    def test_linear_combination_uses_covariance_between_terms(self):
        estimates = pd.Series({"main": 1.0, "interaction": 2.0})
        covariance = pd.DataFrame(
            [[1.0, 0.25], [0.25, 4.0]],
            index=["main", "interaction"], columns=["main", "interaction"],
        )
        estimate, standard_error, *_ = _linear_combination(
            estimates, covariance, {"main": 1.0, "interaction": 1.0}
        )
        self.assertAlmostEqual(estimate, 3.0)
        self.assertAlmostEqual(standard_error, np.sqrt(5.5))

    def test_reliability_bootstrap_is_deterministic(self):
        frame = pd.DataFrame({"left": range(8), "right": range(8)})
        first = bootstrap_split_reliability(frame, "left", "right", iterations=50, seed=3)
        second = bootstrap_split_reliability(frame, "left", "right", iterations=50, seed=3)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["spearman_rho"], 1.0)

    def test_prediction_unlock_requires_both_comparators(self):
        rows = []
        observed = np.arange(10, dtype=float)
        for subject, value in enumerate(observed, start=1):
            rows.extend([
                {"subject": subject, "model": "baseline", "observed": value, "predicted": value + 2.0},
                {"subject": subject, "model": "baseline_plus_neural", "observed": value, "predicted": value + 0.1},
            ])
        result, decision = prediction_increment_uncertainty(
            pd.DataFrame(rows), iterations=100, seed=4
        )
        self.assertEqual(set(result["comparator"]), {"training_fold_mean", "baseline"})
        self.assertTrue(decision["prediction_increment_claim_unlocked"])

    def test_prediction_unlock_fails_when_neural_model_does_not_beat_baseline(self):
        rows = []
        observed = np.arange(10, dtype=float)
        for subject, value in enumerate(observed, start=1):
            rows.extend([
                {"subject": subject, "model": "baseline", "observed": value, "predicted": value},
                {"subject": subject, "model": "baseline_plus_neural", "observed": value, "predicted": value + 0.2},
            ])
        _, decision = prediction_increment_uncertainty(
            pd.DataFrame(rows), iterations=100, seed=5
        )
        self.assertFalse(decision["prediction_increment_claim_unlocked"])

    def test_roi_sensitivity_excludes_three_channel_subject(self):
        rows = []
        for subject, roi, difference in [
            (1, "FC1;FCz;FC2", 0.1),
            (2, "FC3;FC1;FCz;FC2", 0.2),
            (3, "FC3;FC1;FCz;FC2;FC4", 0.3),
        ]:
            rows.extend([
                {"subject": subject, "condition": "non_rhythmic", "frontocentral_roi": roi,
                 "mean_itpc_in_stimulation_window": 0.2},
                {"subject": subject, "condition": "f_theta_plus", "frontocentral_roi": roi,
                 "mean_itpc_in_stimulation_window": 0.2 + difference},
            ])
        result = roi_channel_sensitivity(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result["n"], 2)

    def test_behaviour_sensitivity_preserves_both_strata(self):
        rows = []
        for subject in range(1, 5):
            for congruency in ("congruent", "incongruent"):
                for condition in ("non_rhythmic", "f_theta_plus"):
                    rows.extend([
                        {"subject": subject, "condition": condition, "congruency": congruency,
                         "correct": True, "rt_ms": 500.0 - (10.0 * subject if condition == "f_theta_plus" else 0.0)},
                        {"subject": subject, "condition": condition, "congruency": congruency,
                         "correct": condition == "f_theta_plus" and subject % 2 == 0,
                         "rt_ms": 520.0 if condition == "f_theta_plus" and subject % 2 == 0 else np.nan},
                    ])
        result = behaviour_by_congruency(pd.DataFrame(rows))
        self.assertEqual(set(result["congruency"]), {"congruent", "incongruent"})
        self.assertEqual(len(result), 6)


if __name__ == "__main__":
    unittest.main()
