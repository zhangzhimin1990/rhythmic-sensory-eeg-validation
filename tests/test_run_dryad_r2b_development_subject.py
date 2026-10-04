import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_dryad_r2b_development_subject import (
    DEVELOPMENT_SUBJECTS,
    load_development_subject,
    require_capacity,
)


class DryadR2bDevelopmentRunnerTests(unittest.TestCase):
    def write_selection(self, directory: Path) -> Path:
        path = directory / "selection.csv"
        pd.DataFrame(
            {
                "subject": [2, 13, 27, 23, 37],
                "selection_order": [1, 2, 3, 4, 5],
                "uncompressed_bytes": [100, 200, 300, 400, 500],
            }
        ).to_csv(path, index=False)
        return path

    def test_only_frozen_development_subjects_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection = self.write_selection(Path(temporary))
            for subject in DEVELOPMENT_SUBJECTS:
                observed = load_development_subject(selection, subject)
                self.assertEqual(observed["subject"], subject)
            with self.assertRaises(ValueError):
                load_development_subject(selection, 1)

    def test_selection_order_cannot_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection = self.write_selection(Path(temporary))
            frame = pd.read_csv(selection)
            frame.loc[0, "selection_order"] = 9
            frame.to_csv(selection, index=False)
            with self.assertRaises(ValueError):
                load_development_subject(selection, 2)

    def test_scratch_preflight_reports_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            observed = require_capacity(Path(temporary), 1)
            self.assertGreaterEqual(
                observed["scratch_free_bytes_before_extraction"],
                observed["scratch_required_bytes"],
            )


if __name__ == "__main__":
    unittest.main()
