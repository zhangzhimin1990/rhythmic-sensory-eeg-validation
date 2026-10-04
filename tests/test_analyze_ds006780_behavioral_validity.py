import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_ds006780_behavioral_validity import (
    aggregate_subjects,
    benjamini_hochberg,
    corrected_dprime,
    original_claim_correlations,
)


class Ds006780BehavioralValidityTests(unittest.TestCase):
    def test_dprime_is_finite_at_ceiling(self):
        self.assertTrue(np.isfinite(corrected_dprime(30, 0, 0, 170)))

    def test_bh_is_monotone_in_rank(self):
        adjusted = benjamini_hochberg(np.array([0.03, 0.001, 0.04]))
        self.assertLessEqual(adjusted[1], adjusted[0])
        self.assertLessEqual(adjusted[0], adjusted[2])

    def test_runs_are_aggregated_before_subject_analysis(self):
        rows = []
        for run in ("01", "02"):
            for frequency in (27, 40):
                rows.append(
                    {
                        "subject": "sub-1", "run": run, "frequency_hz": frequency,
                        "n_standard_expected": 50, "n_standard_valid": 45,
                        "hits": 5, "misses": 1, "false_alarms": 1,
                        "correct_rejections": 43, "median_hit_rt": 0.5,
                        "unassigned_responses_total": 0, "n_bad_channels": 0,
                        "local_log_snr_db_mean": 4.0,
                        "local_log_snr_db_median": 4.0,
                        "morlet_power_percent_change_mean": 100.0,
                        "morlet_power_percent_change_median": 90.0,
                        "itpc": 0.4, "itpc_odd_trials": 0.35,
                        "itpc_even_trials": 0.45,
                    }
                )
        participants = pd.DataFrame(
            {"participant_id": ["sub-1"], "age": [10.0], "sex": ["m"],
             "group": ["TD"], "completed_ASSR": ["yes"], "fsiq": [100.0]}
        )
        result = aggregate_subjects(pd.DataFrame(rows), participants)
        self.assertEqual(result.loc[0, "n_standard_valid_40"], 90)
        self.assertFalse(result.loc[0, "technical_usable"])

    def test_original_claim_correlations_are_labeled_and_complete(self):
        frame = pd.DataFrame(
            {
                "group": ["ASD"] * 4 + ["TD"] * 4,
                "dprime_40": np.arange(8, dtype=float),
                "false_alarms_40": np.arange(8, dtype=float),
                "correct_rejections_40": np.full(8, 20.0),
                "local_log_snr_db_mean_40": np.arange(8, dtype=float),
                "morlet_power_percent_change_mean_40": np.arange(8, dtype=float),
                "itpc_40": np.arange(8, dtype=float),
            }
        )
        result = original_claim_correlations(frame)
        self.assertEqual(len(result), 12)
        self.assertEqual(set(result["subset"]), {"all_technical_usable", "ASD"})
        self.assertTrue(result["pearson_q_bh_twelve_tests"].notna().all())


if __name__ == "__main__":
    unittest.main()
