import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_ds006036_network import (
    debiased_squared_wpli,
    pair_connectivity,
    target_fourier_observations,
)


class Ds006036NetworkTests(unittest.TestCase):
    def test_dwpli_is_one_for_consistent_nonzero_lag(self):
        self.assertAlmostEqual(debiased_squared_wpli(np.array([1.0, 2.0, 3.0, 4.0])), 1.0)

    def test_zero_lag_sinusoids_have_near_zero_imaginary_coherence(self):
        sfreq = 500.0
        time = np.arange(1500) / sfreq
        signal = np.sin(2 * np.pi * 10.0 * time)
        observations = target_fourier_observations(np.vstack([signal, signal]), sfreq, 10.0)
        _, imaginary, _ = pair_connectivity(observations, 0, 1)
        self.assertLess(imaginary, 1e-10)

    def test_quarter_cycle_lag_has_large_imaginary_coherence(self):
        sfreq = 500.0
        time = np.arange(1500) / sfreq
        first = np.sin(2 * np.pi * 10.0 * time)
        second = np.sin(2 * np.pi * 10.0 * time + np.pi / 2)
        observations = target_fourier_observations(np.vstack([first, second]), sfreq, 10.0)
        _, imaginary, dwpli = pair_connectivity(observations, 0, 1)
        self.assertGreater(imaginary, 0.99)
        self.assertGreater(dwpli, 0.99)


if __name__ == "__main__":
    unittest.main()
