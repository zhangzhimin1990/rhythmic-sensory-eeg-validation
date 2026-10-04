import unittest

import pandas as pd

from scripts.classify_dryad_confirmatory_coupling_branch import classify


def frames(theta_correct=(0.2, 0.01), theta_rt=(-0.1, 0.20), interaction_correct=(0.15, 0.01), interaction_rt=(-0.05, 0.30)):
    slopes = pd.DataFrame([
        {"model": "correctness_binomial_gee", "condition": "non_rhythmic", "simple_slope_phase_alignment_within": 0.05, "p_value_holm_simple_slope_family": 0.8},
        {"model": "correctness_binomial_gee", "condition": "f_theta_plus", "simple_slope_phase_alignment_within": theta_correct[0], "p_value_holm_simple_slope_family": theta_correct[1]},
        {"model": "correct_rt_gaussian_gee", "condition": "non_rhythmic", "simple_slope_phase_alignment_within": 0.02, "p_value_holm_simple_slope_family": 0.8},
        {"model": "correct_rt_gaussian_gee", "condition": "f_theta_plus", "simple_slope_phase_alignment_within": theta_rt[0], "p_value_holm_simple_slope_family": theta_rt[1]},
    ])
    terms = pd.DataFrame([
        {"model": "correctness_binomial_gee", "term": "phase_alignment_within:C(condition)[T.f_theta_plus]", "estimate": interaction_correct[0], "p_value_holm_within_family": interaction_correct[1]},
        {"model": "correct_rt_gaussian_gee", "term": "phase_alignment_within:C(condition)[T.f_theta_plus]", "estimate": interaction_rt[0], "p_value_holm_within_family": interaction_rt[1]},
    ])
    return slopes, terms


class DryadConfirmatoryCouplingBranchTests(unittest.TestCase):
    def test_matched_slope_and_interaction_support_specific_increment(self):
        result = classify(*frames())
        self.assertEqual(result["branch"], "theta_plus_coupling_with_condition_specific_increment")
        self.assertTrue(result["theta_plus_specific_coupling_supported"])

    def test_simple_slope_alone_does_not_establish_specificity(self):
        result = classify(*frames(interaction_correct=(0.15, 0.40)))
        self.assertEqual(result["branch"], "theta_plus_within_condition_coupling_not_condition_specific")
        self.assertFalse(result["theta_plus_specific_coupling_supported"])

    def test_interaction_without_theta_slope_is_separate_branch(self):
        result = classify(*frames(theta_correct=(0.2, 0.40)))
        self.assertEqual(result["branch"], "condition_difference_without_favourable_theta_plus_slope")

    def test_no_favourable_coupling(self):
        result = classify(*frames(theta_correct=(-0.2, 0.01), interaction_correct=(-0.15, 0.01)))
        self.assertEqual(result["branch"], "no_favourable_theta_plus_coupling")


if __name__ == "__main__":
    unittest.main()
