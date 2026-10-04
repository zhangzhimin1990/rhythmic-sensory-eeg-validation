import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qc_ds006036 import local_snr_db


class Ds006036QcTests(unittest.TestCase):
    def test_local_snr_is_ten_db_for_tenfold_ratio(self):
        freqs = np.arange(0.0, 31.0, 0.5)
        psd = np.ones_like(freqs)
        psd[np.abs(freqs - 10.0) <= 0.26] = 10.0
        self.assertAlmostEqual(local_snr_db(freqs, psd, 10.0), 10.0)


if __name__ == "__main__":
    unittest.main()
