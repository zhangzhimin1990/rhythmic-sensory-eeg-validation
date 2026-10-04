import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluate_dryad_r2b_technical_gate import (
    DEVELOPMENT_SUBJECTS,
    PARTICIPANT_OUTPUT_COLUMNS,
    SUMMARY_NAME,
    VALIDATION_SUBJECTS,
    evaluate_group,
    load_validation_summaries,
    subject_from_name,
)


class DryadR2bTechnicalGateTests(unittest.TestCase):
    def make_participants(self, pass_count):
        subjects = sorted(VALIDATION_SUBJECTS)
        rows = []
        for index, subject in enumerate(subjects):
            row = {column: 0 for column in PARTICIPANT_OUTPUT_COLUMNS}
            row.update(
                {
                    "subject": subject,
                    "technical_pass_development_candidate": index < pass_count,
                    "source_bdf_sha256": f"hash-{subject}",
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)[PARTICIPANT_OUTPUT_COLUMNS]

    def test_subject_name_parser_is_strict(self):
        self.assertEqual(subject_from_name("S44_Stim.bdf"), 44)
        with self.assertRaises(ValueError):
            subject_from_name("subject44.bdf")

    def test_validation_set_has_39_and_excludes_all_five_development_subjects(self):
        self.assertEqual(len(VALIDATION_SUBJECTS), 39)
        self.assertFalse(VALIDATION_SUBJECTS.intersection(DEVELOPMENT_SUBJECTS))

    def test_group_gate_requires_at_least_32_passes(self):
        failed = evaluate_group(self.make_participants(31))
        passed = evaluate_group(self.make_participants(32))
        self.assertFalse(failed["technical_go"])
        self.assertTrue(passed["technical_go"])
        self.assertEqual(
            passed["status"], "frozen_protocol_technical_gate_complete"
        )

    def test_group_gate_rejects_inexact_subject_set(self):
        incomplete = self.make_participants(32).iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "exact frozen validation set"):
            evaluate_group(incomplete)

    def test_loader_derives_subject_instead_of_requiring_json_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for subject in sorted(VALIDATION_SUBJECTS):
                directory = root / f"S{subject}"
                directory.mkdir()
                summary = {
                    column: (f"hash-{subject}" if column == "source_bdf_sha256" else 1)
                    for column in PARTICIPANT_OUTPUT_COLUMNS
                    if column != "subject"
                }
                summary.update(
                    {
                        "source_bdf_name": f"S{subject}_Stim.bdf",
                        "result_bearing_fields_emitted": False,
                    }
                )
                (directory / SUMMARY_NAME).write_text(
                    json.dumps(summary), encoding="utf-8"
                )
            observed = load_validation_summaries(root)
            self.assertEqual(set(observed["subject"]), VALIDATION_SUBJECTS)


if __name__ == "__main__":
    unittest.main()
