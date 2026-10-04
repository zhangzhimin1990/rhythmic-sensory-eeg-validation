import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qc_behavioral_validation_candidates import (
    local_snr_db,
    reconstruct_oddball,
    reconstruct_oddball_by_frequency,
)


class BehavioralValidationQcTests(unittest.TestCase):
    def test_local_snr_is_positive_for_target_peak(self):
        frequency = np.arange(0, 60.5, 0.5)
        power = np.ones_like(frequency)
        power[np.argmin(np.abs(frequency - 40))] = 10
        self.assertAlmostEqual(local_snr_db(frequency, power, 40), 10.0)

    def test_oddball_behavior_reconstruction(self):
        events = pd.DataFrame(
            {
                "onset": [0.0, 0.4, 2.0, 2.3, 4.0, 6.0],
                "trial_type": [
                    "40_Hz_Oddball",
                    "Response_button",
                    "40_Hz_Standard",
                    "Response_button",
                    "40_Hz_Oddball",
                    "40_Hz_Standard",
                ],
            }
        )
        result = reconstruct_oddball(events)
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["misses"], 1)
        self.assertEqual(result["false_alarms"], 1)
        self.assertEqual(result["correct_rejections"], 1)
        self.assertAlmostEqual(result["median_hit_rt"], 0.4)

    def test_oddball_behavior_is_stratified_by_standard_frequency(self):
        events = pd.DataFrame(
            {
                "onset": [0.0, 0.3, 2.0, 4.0, 4.4, 6.0],
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
        result = {row["standard_frequency_hz"]: row for row in reconstruct_oddball_by_frequency(events)}
        self.assertEqual(result[40]["hits"], 1)
        self.assertEqual(result[40]["correct_rejections"], 1)
        self.assertEqual(result[27]["hits"], 1)
        self.assertEqual(result[27]["correct_rejections"], 1)


if __name__ == "__main__":
    unittest.main()
