import unittest

import pandas as pd

from scripts.evaluate_dryad_r2_gate import (
    behaviour_alignment,
    evaluate_group_gate,
    evaluate_participant,
)


class DryadR2GateTests(unittest.TestCase):
    @staticmethod
    def participant_fixture():
        audit = {
            "n_trials": 384,
            "n_valid_sequences": 384,
            "n_scalp_eeg_channels": 64,
        }
        signal = {
            "n_bad_channels": 0,
            "n_good_channels": 64,
            "n_usable_trials": 384,
            "n_trials": 384,
        }
        metrics = pd.DataFrame(
            {
                "condition": ["2_hz", "f_theta", "f_theta_plus", "non_rhythmic"],
                "frontocentral_roi": ["FC3;FC1;FCz;FC2;FC4"] * 4,
                "n_usable_trials": [96] * 4,
                "n_trials": [96] * 4,
                "mean_itpc_in_stimulation_window": [0.8, 0.4, 0.5, 0.1],
                "odd_trial_itpc": [0.8, 0.4, 0.5, 0.1],
                "even_trial_itpc": [0.8, 0.4, 0.5, 0.1],
                "evoked_local_log_snr_db": [-5.0, 5.0, 10.0, -10.0],
            }
        )
        alignment = pd.DataFrame(
            {
                "accuracy_absolute_difference": [0.0] * 8,
                "rt_absolute_difference_ms": [0.5] * 8,
            }
        )
        return audit, signal, metrics, alignment

    def test_behaviour_accuracy_excludes_omissions_from_denominator(self):
        rows = []
        for condition in ("2_hz", "f_theta", "f_theta_plus", "non_rhythmic"):
            for congruency in ("congruent", "incongruent"):
                rows.extend(
                    [
                        {"condition": condition, "congruency": congruency, "sequence_valid": True, "response_class": "correct", "rt_ms": 500.0},
                        {"condition": condition, "congruency": congruency, "sequence_valid": True, "response_class": "incorrect", "rt_ms": 600.0},
                        {"condition": condition, "congruency": congruency, "sequence_valid": True, "response_class": "omission", "rt_ms": float("nan")},
                    ]
                )
        author = {}
        for prefix in ("doshz", "fθ", "fθ+", "nr"):
            for suffix in ("con", "incon"):
                author[f"{prefix}_acc_{suffix}"] = 0.5
                author[f"{prefix}_rt_{suffix}"] = 500.0
        aligned = behaviour_alignment(pd.DataFrame(rows), pd.Series(author))
        self.assertTrue((aligned["reconstructed_accuracy"] == 0.5).all())
        self.assertTrue((aligned["n_scored_trials"] == 2).all())

    def test_group_gate_passes_frozen_positive_direction_rules(self):
        participants = pd.DataFrame({"subject": [2, 13, 27, 23, 37], "technical_pass": [True] * 4 + [False]})
        contrasts = pd.DataFrame(
            {
                "itpc_theta_plus_minus_non_rhythmic": [0.3, 0.2, 0.1, 0.05, -0.01],
                "odd_itpc_theta_plus_minus_non_rhythmic": [0.3, 0.2, 0.1, -0.01, -0.02],
                "even_itpc_theta_plus_minus_non_rhythmic": [0.3, 0.2, 0.1, -0.01, -0.02],
                "evoked_snr_theta_plus_minus_non_rhythmic_db": [10, 8, 2, -1, -2],
            }
        )
        summary = evaluate_group_gate(participants, contrasts)
        self.assertTrue(summary["technical_go_requires_at_least_4_of_5"])
        self.assertTrue(summary["measurement_go"])
        self.assertTrue(summary["overall_r2_go"])

    def test_group_gate_fails_when_full_itpc_direction_is_not_replicated(self):
        participants = pd.DataFrame({"subject": [2, 13, 27, 23, 37], "technical_pass": [True] * 5})
        contrasts = pd.DataFrame(
            {
                "itpc_theta_plus_minus_non_rhythmic": [0.3, 0.2, -0.1, -0.2, -0.3],
                "odd_itpc_theta_plus_minus_non_rhythmic": [0.3, 0.2, 0.1, -0.1, -0.2],
                "even_itpc_theta_plus_minus_non_rhythmic": [0.3, 0.2, 0.1, -0.1, -0.2],
                "evoked_snr_theta_plus_minus_non_rhythmic_db": [10, 8, 2, -1, -2],
            }
        )
        summary = evaluate_group_gate(participants, contrasts)
        self.assertFalse(summary["measurement_go"])
        self.assertFalse(summary["overall_r2_go"])

    def test_participant_gate_requires_100ms_pre_target_margin(self):
        audit, signal, metrics, alignment = self.participant_fixture()
        passing, _ = evaluate_participant(
            2, audit, signal, metrics, alignment, 1945.8
        )
        failing, _ = evaluate_participant(
            2, audit, signal, metrics, alignment, 1800.0
        )
        key = "analysis_window_precedes_target_by_at_least_100ms_gate"
        self.assertTrue(passing[key])
        self.assertTrue(passing["technical_pass"])
        self.assertFalse(failing[key])
        self.assertFalse(failing["technical_pass"])


if __name__ == "__main__":
    unittest.main()
