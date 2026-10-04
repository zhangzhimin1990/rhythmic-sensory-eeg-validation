import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_validation_strength import (
    build_claim_strength_summary,
    build_correlation_precision,
    build_equivalence_grid,
    build_prediction_precision,
    tost_normal,
)


class ValidationStrengthTests(unittest.TestCase):
    def test_equivalence_when_90_percent_ci_is_inside_bounds(self):
        result = tost_normal(0.0, 0.05, -0.10, 0.10)
        self.assertTrue(result["equivalent_alpha_0_05"])
        self.assertGreater(result["ci90_low"], -0.10)
        self.assertLess(result["ci90_high"], 0.10)

    def test_not_equivalent_when_interval_crosses_bound(self):
        result = tost_normal(0.0, 0.10, -0.10, 0.10)
        self.assertFalse(result["equivalent_alpha_0_05"])

    def test_invalid_bounds_fail(self):
        with self.assertRaises(ValueError):
            tost_normal(0.0, 0.10, 0.10, -0.10)

    def test_correlation_precision_covers_auditory_and_visual_severity(self):
        result = build_correlation_precision()
        self.assertEqual(len(result), 3)
        self.assertEqual(set(result["dataset"]), {"ds005048", "ds006036"})
        self.assertEqual(set(result.loc[result.dataset.eq("ds006036"), "population"]), {"AD", "FTD"})

    def test_prediction_precision_includes_dryad_and_relative_scale(self):
        result = build_prediction_precision()
        self.assertEqual(len(result), 7)
        self.assertIn("dryad_theta", set(result["dataset"]))
        self.assertTrue(result["relative_delta_percent"].notna().all())
        dprime = result[(result.dataset == "ds006780") & (result.outcome == "dprime_40")]
        self.assertTrue((dprime["direction"] == "clear_worsening").all())

    def test_equivalence_grid_separates_local_snr_and_itpc(self):
        result = build_equivalence_grid()
        ds6 = result[result.dataset.eq("ds006780")]
        self.assertEqual(set(ds6["predictor"]), {"local_log_snr", "itpc"})
        self.assertEqual(len(ds6), 6)
        itpc_025 = ds6[
            ds6.predictor.eq("itpc") & ds6.bound_label.eq("standardized_0.25")
        ].iloc[0]
        self.assertFalse(bool(itpc_025.equivalent_alpha_0_05))

    def test_claim_strength_keeps_imprecision_and_prediction_distinct(self):
        result = build_claim_strength_summary(
            build_equivalence_grid(),
            build_correlation_precision(),
            build_prediction_precision(),
        )
        self.assertIn("imprecise", set(result["strength"]))
        self.assertIn("clear_worsening", set(result["strength"]))
        self.assertTrue(result["prohibited_wording"].str.len().gt(0).all())


if __name__ == "__main__":
    unittest.main()
