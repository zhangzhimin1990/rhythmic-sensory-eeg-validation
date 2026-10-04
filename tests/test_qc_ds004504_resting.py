import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qc_ds004504_resting import band_power


class Ds004504RestingQcTests(unittest.TestCase):
    def test_band_power_integrates_constant_spectrum(self):
        freqs = np.arange(0.0, 50.25, 0.25)
        psd = np.ones_like(freqs)
        self.assertAlmostEqual(band_power(freqs, psd, 8.0, 13.0), 4.75)


if __name__ == "__main__":
    unittest.main()
