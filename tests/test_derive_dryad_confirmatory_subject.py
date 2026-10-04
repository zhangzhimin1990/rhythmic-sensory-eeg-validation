import unittest

import numpy as np

from scripts.derive_dryad_confirmatory_subject import (
    condition_metrics,
    leave_one_trial_out_phase_alignment,
    mean_itpc,
)


class DryadConfirmatorySubjectTests(unittest.TestCase):
    def setUp(self):
        self.sfreq = 256.0
        self.frequency = 7.0
        self.time = np.arange(384) / self.sfreq

    def test_phase_locked_trials_have_high_itpc(self):
        base = np.sin(2 * np.pi * self.frequency * self.time)
        epochs = np.tile(base, (24, 1))
        self.assertGreater(mean_itpc(epochs, self.sfreq, self.frequency), 0.98)

    def test_random_phase_trials_have_lower_itpc(self):
        phases = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        epochs = np.asarray([
            np.sin(2 * np.pi * self.frequency * self.time + phase) for phase in phases
        ])
        self.assertLess(mean_itpc(epochs, self.sfreq, self.frequency), 0.15)

    def test_leave_one_out_alignment_detects_phase_outlier(self):
        base = np.sin(2 * np.pi * self.frequency * self.time)
        epochs = np.tile(base, (12, 1))
        epochs[-1] = -base
        scores = leave_one_trial_out_phase_alignment(epochs, self.sfreq, self.frequency)
        self.assertGreater(np.median(scores[:-1]), 0.95)
        self.assertLess(scores[-1], -0.9)

    def test_condition_metrics_preserve_frozen_outputs(self):
        base = np.sin(2 * np.pi * self.frequency * self.time)
        epochs = np.tile(base, (12, 1))
        metrics = condition_metrics(epochs, self.sfreq, self.frequency)
        required = {
            "mean_itpc_in_stimulation_window",
            "evoked_local_log_snr_db",
            "total_power_local_log_snr_db",
            "induced_local_log_snr_db",
            "odd_trial_itpc",
            "even_trial_itpc",
            "first_half_itpc",
            "second_half_itpc",
        }
        self.assertTrue(required.issubset(metrics))
        self.assertGreater(metrics["mean_itpc_in_stimulation_window"], 0.98)

    def test_full_epoch_filter_then_trim_preserves_locked_phase(self):
        full_time = np.arange(7 * int(self.sfreq)) / self.sfreq
        base = np.sin(2 * np.pi * self.frequency * full_time)
        epochs = np.tile(base, (12, 1))
        start = int((3.0 + 0.25) * self.sfreq)
        stop = int((3.0 + 1.75) * self.sfreq)
        observed = mean_itpc(
            epochs, self.sfreq, self.frequency, analysis_slice=(start, stop)
        )
        self.assertGreater(observed, 0.98)


if __name__ == "__main__":
    unittest.main()
