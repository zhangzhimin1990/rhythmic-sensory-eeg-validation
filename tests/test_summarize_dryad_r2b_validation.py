import unittest

from scripts.summarize_dryad_r2b_validation import wilson_interval


class DryadR2bValidationSummaryTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_fraction(self):
        lower, upper = wilson_interval(35, 39)
        self.assertLess(lower, 35 / 39)
        self.assertGreater(upper, 35 / 39)
        self.assertAlmostEqual(lower, 0.7642, places=4)
        self.assertAlmostEqual(upper, 0.9594, places=4)

    def test_wilson_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            wilson_interval(40, 39)


if __name__ == "__main__":
    unittest.main()
