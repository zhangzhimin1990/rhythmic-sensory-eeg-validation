import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_dryad_confirmatory_subject_amended import (
    AMENDMENT_ID,
    ANALYSIS_FREQUENCY_MULTIPLIER,
    load_subject,
)


ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ROOT / "outputs/qc/dryad_raw_stream/stim_zip_members.csv"
PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"


class DryadConfirmatoryRunnerAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.metadata = Path(self.temporary.name) / "metadata.xlsx"
        frequencies = [3.0] * 44
        frequencies[7] = 2.47
        pd.DataFrame({
            "subject": range(1, 45),
            "individual_freq": frequencies,
        }).to_excel(self.metadata, sheet_name="Dataset", index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_amendment_has_stable_identifier(self):
        self.assertEqual(AMENDMENT_ID, "dryad_confirmatory_runner_input_guard_v1")

    def test_originally_blocked_locked_subject_is_accepted(self):
        result = load_subject(8, MEMBERS, PROTOCOL, self.metadata)
        self.assertEqual(result["individual_theta_hz"], 2.47)
        self.assertGreaterEqual(
            ANALYSIS_FREQUENCY_MULTIPLIER * result["individual_theta_hz"], 3.0
        )

    def test_all_locked_subject_analysis_frequencies_are_valid(self):
        subjects = json.loads(PROTOCOL.read_text(encoding="utf-8"))["confirmatory_subjects"]
        values = [load_subject(subject, MEMBERS, PROTOCOL, self.metadata) for subject in subjects]
        frequencies = [ANALYSIS_FREQUENCY_MULTIPLIER * row["individual_theta_hz"] for row in values]
        self.assertEqual(len(values), 35)
        self.assertGreaterEqual(min(frequencies), 3.0)
        self.assertLessEqual(max(frequencies), 8.0)

    def test_excluded_subject_remains_rejected(self):
        with self.assertRaisesRegex(ValueError, "locked confirmatory set"):
            load_subject(1, MEMBERS, PROTOCOL, self.metadata)


if __name__ == "__main__":
    unittest.main()
