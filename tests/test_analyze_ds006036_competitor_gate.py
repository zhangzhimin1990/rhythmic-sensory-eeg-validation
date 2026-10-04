import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_ds006036_competitor_gate import driving_index, spearman_brown


class Ds006036CompetitorGateTests(unittest.TestCase):
    def test_driving_index_recovers_tenfold_target_to_background(self):
        frequencies = np.arange(0.0, 20.5, 0.5)
        psd = np.ones_like(frequencies)
        psd[np.abs(frequencies - 10.0) <= 0.26] = 10.0
        self.assertAlmostEqual(driving_index(frequencies, psd, 10.0), 10.0)

    def test_spearman_brown(self):
        self.assertAlmostEqual(spearman_brown(0.5), 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
