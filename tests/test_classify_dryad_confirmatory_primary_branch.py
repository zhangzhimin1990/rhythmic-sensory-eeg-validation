import unittest

import pandas as pd

from scripts.classify_dryad_confirmatory_primary_branch import classify


def primary(estimate, p_value, equivalent=False):
    return pd.Series({
        "mean_difference_theta_plus_minus_non_rhythmic": estimate,
        "p_value_t_two_sided": p_value,
        "equivalent_within_plus_minus_dz_0_3": equivalent,
    })


def sensitivity(estimate):
    return pd.Series({"mean_difference_theta_plus_minus_non_rhythmic": estimate})


class DryadConfirmatoryPrimaryBranchTests(unittest.TestCase):
    def test_positive_significant_branch_requires_consistent_sensitivity_for_claim(self):
        result = classify(primary(0.05, 0.01), sensitivity(0.03))
        self.assertEqual(result["branch"], "theta_plus_higher")
        self.assertTrue(result["theta_plus_superiority_claim_unlocked"])

    def test_opposite_direction_is_not_misclassified_as_inconclusive(self):
        result = classify(primary(-0.05, 0.01), sensitivity(-0.02))
        self.assertEqual(result["branch"], "theta_plus_lower_opposite_direction")
        self.assertFalse(result["theta_plus_superiority_claim_unlocked"])

    def test_equivalent_branch(self):
        result = classify(primary(0.001, 0.80, True), sensitivity(0.002))
        self.assertEqual(result["branch"], "operationally_equivalent")

    def test_inconclusive_branch(self):
        result = classify(primary(0.02, 0.18, False), sensitivity(-0.01))
        self.assertEqual(result["branch"], "inconclusive_not_equivalent")
        self.assertFalse(result["roi_sensitivity_direction_consistent"])


if __name__ == "__main__":
    unittest.main()
