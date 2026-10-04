import unittest

import pandas as pd

from scripts.classify_dryad_confirmatory_behaviour_branch import classify


def effects(accuracy=(0.0, 0.8), rt=(-10.0, 0.03), ie=(-12.0, 0.02)):
    return pd.DataFrame([
        {"metric": "accuracy", "mean_difference_theta_plus_minus_non_rhythmic": accuracy[0], "p_value_holm_behaviour": accuracy[1]},
        {"metric": "median_correct_rt_ms", "mean_difference_theta_plus_minus_non_rhythmic": rt[0], "p_value_holm_behaviour": rt[1]},
        {"metric": "inverse_efficiency_ms", "mean_difference_theta_plus_minus_non_rhythmic": ie[0], "p_value_holm_behaviour": ie[1]},
    ])


class DryadConfirmatoryBehaviourBranchTests(unittest.TestCase):
    def test_advantage_does_not_claim_accuracy_noninferiority(self):
        result = classify(effects())
        self.assertEqual(result["branch"], "inverse_efficiency_advantage_no_detected_accuracy_decrement")
        self.assertTrue(result["acute_task_performance_advantage_claim_unlocked"])
        self.assertFalse(result["accuracy_noninferiority_established"])

    def test_accuracy_decrement_keeps_tradeoff_visible(self):
        result = classify(effects(accuracy=(-0.03, 0.01)))
        self.assertEqual(result["branch"], "inverse_efficiency_advantage_with_accuracy_decrement")
        self.assertFalse(result["acute_task_performance_advantage_claim_unlocked"])

    def test_no_clear_inverse_efficiency_advantage(self):
        result = classify(effects(ie=(-4.0, 0.30)))
        self.assertEqual(result["branch"], "no_clear_inverse_efficiency_advantage")

    def test_inverse_efficiency_disadvantage(self):
        result = classify(effects(ie=(12.0, 0.01)))
        self.assertEqual(result["branch"], "inverse_efficiency_disadvantage")


if __name__ == "__main__":
    unittest.main()
