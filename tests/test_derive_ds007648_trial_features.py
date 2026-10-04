import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from derive_ds007648_trial_features import narrowband_log_ratio, robust_upper


class Ds007648FeatureTests(unittest.TestCase):
    def test_robust_upper_is_not_driven_by_single_outlier(self):
        threshold = robust_upper(np.array([1.0, 1.0, 1.1, 0.9, 100.0]))
        self.assertLess(threshold, 2.0)

    def test_narrowband_ratio_recovers_late_amplitude_increase(self):
        sfreq = 500.0
        time = np.arange(2500) / sfreq - 1.5
        amplitude = np.where(time >= 0, 2.0, 1.0)
        signal = amplitude * np.sin(2 * np.pi * 40 * time)
        epoch = np.vstack([signal, signal])
        zero = int(1.5 * sfreq)
        baseline = slice(zero - int(0.7 * sfreq), zero - int(0.2 * sfreq))
        late = slice(zero + int(2.5 * sfreq), zero + int(3.0 * sfreq))
        ratio, baseline_amp, late_amp, coefficient = narrowband_log_ratio(
            epoch, sfreq, 40.0, baseline, late
        )
        self.assertGreater(ratio, 0.4)
        self.assertGreater(late_amp, baseline_amp)
        self.assertGreater(abs(coefficient), 0.1)


if __name__ == "__main__":
    unittest.main()
