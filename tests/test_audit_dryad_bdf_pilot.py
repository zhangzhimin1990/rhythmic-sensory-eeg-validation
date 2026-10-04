import unittest

import numpy as np

from scripts.audit_dryad_bdf_pilot import (
    collapse_overlapping_status_edges,
    normalize_trigger,
    reconstruct_trials,
)


class DryadBdfPilotTests(unittest.TestCase):
    def test_biosemi_status_offset_is_reduced_to_low_byte(self):
        self.assertEqual(normalize_trigger(65280 + 42), 42)
        self.assertEqual(normalize_trigger(65536 + 3), 3)

    def test_condition_target_response_sequences_are_reconstructed(self):
        events = np.array(
            [
                [0, 0, 65281],
                [4096, 0, 65291],
                [6144, 0, 65331],
                [8192, 0, 65282],
                [12288, 0, 65302],
                [14336, 0, 65340],
            ],
            dtype=int,
        )
        trials = reconstruct_trials(events, sfreq=4096.0)
        self.assertEqual(list(trials["condition"]), ["f_theta", "2_hz"])
        self.assertEqual(list(trials["response_class"]), ["correct", "incorrect"])
        self.assertEqual(list(trials["congruency"]), ["congruent", "congruent"])
        self.assertTrue(trials["sequence_valid"].all())
        self.assertTrue(np.allclose(trials["rt_ms"], [500.0, 500.0]))

    def test_unexpected_response_is_retained_as_invalid(self):
        events = np.array([[0, 0, 1], [4096, 0, 12], [6144, 0, 61]], dtype=int)
        trials = reconstruct_trials(events, sfreq=4096.0)
        self.assertEqual(trials.loc[0, "response_class"], "missing_or_unexpected")
        self.assertFalse(bool(trials.loc[0, "sequence_valid"]))
        self.assertTrue(np.isnan(trials.loc[0, "rt_ms"]))

    def test_one_sample_bitwise_or_states_are_collapsed(self):
        step_events = np.array(
            [
                [0, 255, 2],
                [4096, 2, 14],   # 2 OR target 12
                [4097, 14, 12],
                [6144, 12, 60],  # target 12 OR correct response 52
                [6145, 60, 52],
                [8192, 52, 54],  # response 52 OR condition 2
                [8193, 54, 2],
            ],
            dtype=int,
        )
        semantic = collapse_overlapping_status_edges(step_events)
        self.assertEqual(list(semantic[:, 2]), [2, 12, 52, 2])
        trials = reconstruct_trials(semantic, sfreq=4096.0)
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials.loc[0, "condition"], "2_hz")
        self.assertEqual(trials.loc[0, "response_class"], "correct")
        self.assertAlmostEqual(trials.loc[0, "rt_ms"], 500.0)
        self.assertTrue(bool(trials.loc[0, "sequence_valid"]))


if __name__ == "__main__":
    unittest.main()
