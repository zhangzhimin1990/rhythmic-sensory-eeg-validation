import unittest

import numpy as np
import pandas as pd

from scripts.develop_dryad_r2b_technical_subject import (
    aggregate_technical_summary,
    eeg_sequence_mask,
    robust_upper,
)


class DryadR2bTechnicalSubjectTests(unittest.TestCase):
    def make_trials(self, count=384):
        return pd.DataFrame(
            {
                "condition": [
                    "f_theta",
                    "2_hz",
                    "f_theta_plus",
                    "non_rhythmic",
                ]
                * (count // 4),
                "condition_code": [1, 2, 3, 4] * (count // 4),
                "stimulation_sample": np.arange(count) + 10000,
                "target_sample": np.arange(count) + 13000,
                "stimulation_to_target_ms": 1900.0,
                "response_class": "missing_or_unexpected",
            }
        )

    def make_channels(self, n_bad=0):
        channels = ["FC3", "FC1", "FCz", "FC2", "FC4"] + [
            f"X{index}" for index in range(59)
        ]
        return pd.DataFrame(
            {
                "channel": channels,
                "r2b_bad_channel_development": [True] * n_bad
                + [False] * (64 - n_bad),
            }
        )

    def test_eeg_sequence_does_not_require_a_behavioural_response(self):
        trials = self.make_trials()
        self.assertTrue(eeg_sequence_mask(trials).all())

    def test_robust_upper_is_not_driven_by_one_large_artifact(self):
        limit = robust_upper(np.asarray([10.0, 10.1, 9.9, 10.0, 1000.0]))
        self.assertLess(limit, 11.0)

    def test_candidate_gate_passes_complete_technical_summary(self):
        trials = self.make_trials()
        epochs = pd.DataFrame(
            {
                "cell_code_internal": [1, 2, 3, 4] * 96,
                "usable": [True] * 384,
            }
        )
        summary = aggregate_technical_summary(trials, epochs, self.make_channels())
        self.assertTrue(summary["technical_pass_development_candidate"])
        self.assertNotIn("response", " ".join(summary).lower())
        self.assertNotIn("condition", " ".join(summary).lower())

    def test_candidate_gate_fails_too_many_bad_channels(self):
        trials = self.make_trials()
        epochs = pd.DataFrame(
            {
                "cell_code_internal": [1, 2, 3, 4] * 96,
                "usable": [True] * 384,
            }
        )
        summary = aggregate_technical_summary(trials, epochs, self.make_channels(13))
        self.assertFalse(summary["channel_count_and_quality_gate"])
        self.assertFalse(summary["technical_pass_development_candidate"])

    def test_candidate_gate_fails_incomplete_event_cell(self):
        trials = self.make_trials()
        epochs = pd.DataFrame(
            {
                "cell_code_internal": [1] * 96 + [2] * 96 + [3] * 96 + [4] * 96,
                "usable": [True] * 288 + [True] * 70 + [False] * 26,
            }
        )
        summary = aggregate_technical_summary(trials, epochs, self.make_channels())
        self.assertLess(summary["minimum_event_cell_usable_fraction"], 0.80)
        self.assertFalse(summary["minimum_event_cell_usability_gate"])
        self.assertFalse(summary["technical_pass_development_candidate"])


if __name__ == "__main__":
    unittest.main()
