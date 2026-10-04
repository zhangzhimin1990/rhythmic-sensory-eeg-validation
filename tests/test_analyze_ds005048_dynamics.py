import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_ds005048_dynamics_network import linear_slope


class Ds005048DynamicsTests(unittest.TestCase):
    def test_linear_slope_recovers_known_quartile_change(self):
        self.assertAlmostEqual(linear_slope([1.0, 1.5, 2.0, 2.5]), 0.5)


if __name__ == "__main__":
    unittest.main()
