import unittest

import pandas as pd

from scripts.classify_dryad_confirmatory_prediction_branch import classify


def increments(training=(1.0, 0.2, 1.1), baseline=(0.8, 0.1, 0.7)):
    rows = []
    for comparator, values in (("training_fold_mean", training), ("baseline", baseline)):
        rows.append({
            "comparator": comparator,
            "mae_improvement_comparator_minus_neural": values[0],
            "mae_improvement_bootstrap_ci95_low": values[1],
            "mae_improvement_bootstrap_ci95_high": values[0] + 0.8,
            "rmse_improvement_comparator_minus_neural": values[2],
        })
    return pd.DataFrame(rows)


class DryadConfirmatoryPredictionBranchTests(unittest.TestCase):
    def test_both_comparators_unlock_internal_increment_only(self):
        result = classify(increments())
        self.assertEqual(result["branch"], "consistent_internal_out_of_sample_increment")
        self.assertTrue(result["prediction_increment_claim_unlocked"])
        self.assertFalse(result["independent_external_validation"])

    def test_one_comparator_is_mixed_not_consistent(self):
        result = classify(increments(baseline=(0.2, -0.3, 0.1)))
        self.assertEqual(result["branch"], "comparator_dependent_prediction_increment")
        self.assertFalse(result["prediction_increment_claim_unlocked"])

    def test_no_comparator_support(self):
        result = classify(increments(training=(-0.1, -0.5, -0.2), baseline=(0.2, -0.3, 0.1)))
        self.assertEqual(result["branch"], "no_consistent_prediction_increment")


if __name__ == "__main__":
    unittest.main()
