import unittest

import numpy as np
import pandas as pd

from scripts.qc_dryad_signal_pilot import (
    flag_channels,
    local_log_snr_db,
    local_snr_interpretation,
    robust_z,
    select_epochs_by_trial_index,
)


class DryadSignalPilotTests(unittest.TestCase):
    def test_robust_z_is_not_driven_by_one_outlier(self):
        values = np.array([1.0, 1.1, 0.9, 1.05, 20.0])
        scores = robust_z(values)
        self.assertLess(np.max(np.abs(scores[:4])), 2.1)
        self.assertGreater(scores[-1], 5.0)

    def test_local_snr_recovers_target_peak(self):
        freqs = np.arange(0.0, 20.25, 0.25)
        power = np.ones_like(freqs)
        power[(freqs >= 4.5) & (freqs <= 5.5)] = 10.0
        self.assertAlmostEqual(local_log_snr_db(freqs, power, 5.0), 10.0, places=6)

    def test_channel_flags_preserve_normal_channels_and_find_outlier(self):
        metrics = pd.DataFrame(
            {
                "channel": [f"C{i}" for i in range(10)],
                "finite_fraction": [1.0] * 10,
                "rms_uv": [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 1000.0],
                "line_noise_db": [0.0] * 10,
                "high_frequency_db": [-10.0] * 10,
                "median_channel_correlation": [0.7] * 10,
            }
        )
        flagged = flag_channels(metrics)
        self.assertFalse(flagged.iloc[:9]["bad_channel"].any())
        self.assertTrue(bool(flagged.iloc[9]["bad_channel"]))
        self.assertIn("rms_outlier", flagged.iloc[9]["bad_channel_reason"])

    def test_tiny_absolute_line_noise_difference_is_not_flagged(self):
        metrics = pd.DataFrame(
            {
                "channel": [f"C{i}" for i in range(10)],
                "finite_fraction": [1.0] * 10,
                "rms_uv": [10.0] * 10,
                "line_noise_db": [0.0] * 9 + [0.3],
                "high_frequency_db": [-10.0] * 10,
                "median_channel_correlation": [0.7] * 10,
            }
        )
        flagged = flag_channels(metrics)
        self.assertFalse(flagged["bad_channel"].any())

    def test_non_rhythmic_snr_is_labeled_as_analysis_frequency_readout(self):
        self.assertEqual(
            local_snr_interpretation("non_rhythmic", 6.66),
            "same_analysis_frequency_readout_not_stimulus_frequency_specificity",
        )
        self.assertEqual(
            local_snr_interpretation("2_hz", 2.0),
            "descriptive_only_low_frequency_1f_boundary",
        )

    def test_epoch_selection_uses_trial_identity_after_out_of_bounds_rejection(self):
        indexed = [
            (0, np.full(3, 10.0)),
            (2, np.full(3, 20.0)),
            (3, np.full(3, 30.0)),
        ]
        selected = select_epochs_by_trial_index(indexed, [2, 3])
        np.testing.assert_array_equal(
            selected,
            np.asarray([[20.0, 20.0, 20.0], [30.0, 30.0, 30.0]]),
        )


if __name__ == "__main__":
    unittest.main()
