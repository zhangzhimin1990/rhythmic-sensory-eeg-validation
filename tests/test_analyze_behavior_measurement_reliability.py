import unittest

import numpy as np
import pandas as pd

from scripts.analyze_behavior_measurement_reliability import (
    corrected_dprime,
    ds007648_halves,
    spearman_brown,
    split_ds006780_events,
)


class BehaviorMeasurementReliabilityTests(unittest.TestCase):
    def test_spearman_brown(self):
        self.assertAlmostEqual(spearman_brown(0.5), 2 / 3)

    def test_ds007648_split_is_stratified_by_trial_type(self):
        frame = pd.DataFrame(
            {
                "subject": ["s1"] * 8,
                "trial_type": ["a"] * 4 + ["b"] * 4,
                "trial_index": range(8),
                "eeg_valid": [True] * 8,
                "Correct": [1, 0, 1, 0, 1, 1, 0, 0],
                "response_time": np.arange(1, 9, dtype=float),
            }
        )
        halves = ds007648_halves(frame)
        self.assertEqual(halves.set_index("half").n_trials.to_dict(), {0: 4, 1: 4})

    def test_ds006780_split_preserves_counts(self):
        events = pd.DataFrame(
            {
                "onset": [0.0, 0.2, 2.0, 4.0, 4.3, 6.0, 8.0, 8.2],
                "trial_type": [
                    "40_Hz_Oddball", "Response_button", "40_Hz_Oddball",
                    "40_Hz_Standard", "Response_button", "40_Hz_Standard",
                    "27_Hz_Oddball", "Response_button",
                ],
            }
        )
        split = split_ds006780_events(events)
        totals = split.groupby("frequency_hz")[list(COUNT_COLUMNS_FOR_TEST)].sum()
        self.assertEqual(totals.loc[40].to_dict(), {"hits": 1, "misses": 0, "false_alarms": 1, "correct_rejections": 1})
        self.assertEqual(totals.loc[27].to_dict(), {"hits": 1, "misses": 1, "false_alarms": 0, "correct_rejections": 0})

    def test_corrected_dprime_is_finite_at_ceiling(self):
        self.assertTrue(np.isfinite(corrected_dprime(30, 0, 0, 170)))


COUNT_COLUMNS_FOR_TEST = ("hits", "misses", "false_alarms", "correct_rejections")


if __name__ == "__main__":
    unittest.main()
