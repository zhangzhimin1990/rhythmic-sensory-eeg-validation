import unittest

import numpy as np
import pandas as pd

from scripts.select_dryad_r2_pilot import (
    EXPECTED_SUBJECTS,
    SELECTION_VARIABLES,
    empirical_rank_01,
    greedy_maximin_selection,
    validate_profile,
)


def synthetic_profile() -> pd.DataFrame:
    subjects = np.asarray(EXPECTED_SUBJECTS)
    return pd.DataFrame(
        {
            "subject": subjects,
            "individual_freq": 2.0 + subjects / 20.0,
            "bl_rt": 400.0 + subjects * 4.0,
            "uncompressed_bytes": 700_000_000 + subjects * 1_000_000,
        }
    )


class DryadR2SelectionTests(unittest.TestCase):
    def test_selector_variable_allowlist_excludes_post_stimulation_outcomes(self):
        self.assertEqual(
            SELECTION_VARIABLES,
            ("individual_freq", "bl_rt", "uncompressed_bytes"),
        )

    def test_empirical_rank_preserves_ties(self):
        ranked = empirical_rank_01(pd.Series([1.0, 2.0, 2.0, 4.0], name="x"))
        self.assertTrue(np.allclose(ranked, [0.0, 0.5, 0.5, 1.0]))

    def test_selection_is_deterministic_and_keeps_seed(self):
        profile = synthetic_profile()
        first, selected_first = greedy_maximin_selection(profile, seed_subject=2, n_selected=5)
        second, selected_second = greedy_maximin_selection(profile, seed_subject=2, n_selected=5)
        self.assertEqual(selected_first, selected_second)
        self.assertEqual(selected_first[0], 2)
        self.assertEqual(len(set(selected_first)), 5)
        self.assertTrue(first.equals(second))

    def test_validation_rejects_duplicate_subjects(self):
        profile = synthetic_profile()
        profile.loc[1, "subject"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate subjects"):
            validate_profile(profile)


if __name__ == "__main__":
    unittest.main()
