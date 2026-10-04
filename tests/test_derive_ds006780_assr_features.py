import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from derive_ds006780_assr_features import behavior_by_frequency, morlet_metrics


class Ds006780AssrFeatureTests(unittest.TestCase):
    def test_behavior_is_reconstructed_by_standard_block(self):
        events = pd.DataFrame(
            {
                "onset": [0, 0.3, 2, 4, 4.4, 6],
                "trial_type": [
                    "27_Hz_Oddball",
                    "Response_button",
                    "40_Hz_Standard",
                    "40_Hz_Oddball",
                    "Response_button",
                    "27_Hz_Standard",
                ],
            }
        )
        result = {x["frequency_hz"]: x for x in behavior_by_frequency(events)}
        self.assertEqual(result[40]["hits"], 1)
        self.assertEqual(result[40]["correct_rejections"], 1)
        self.assertEqual(result[27]["hits"], 1)

    def test_morlet_metrics_detect_phase_locked_target(self):
        sfreq = 512.0
        times = np.arange(int(1.4 * sfreq)) / sfreq - 0.6
        rng = np.random.default_rng(4)
        epochs = []
        for _ in range(30):
            signal = rng.normal(scale=0.2, size=(3, len(times)))
            signal += 2 * np.sin(2 * np.pi * 40 * times)[None, :] * (times >= 0)
            epochs.append(signal)
        metrics = morlet_metrics(np.stack(epochs), sfreq, times, 40)
        self.assertGreater(metrics["itpc"], 0.5)
        self.assertGreater(metrics["morlet_power_percent_change_mean"], 0)
        self.assertTrue(np.isfinite(metrics["local_log_snr_db_mean"]))
        self.assertTrue(np.isfinite(metrics["local_log_snr_db_median"]))
        self.assertGreater(metrics["local_log_snr_db_mean"], 3)


if __name__ == "__main__":
    unittest.main()
