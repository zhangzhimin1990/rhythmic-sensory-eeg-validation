import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_dryad_confirmatory_figure7 as figure7
from scripts import build_dryad_confirmatory_supplement as supplement


ROOT = Path(__file__).resolve().parents[1]


class DryadReleaseFallbackTests(unittest.TestCase):
    def test_figure7_accepts_deidentified_participant_contrasts_without_subject_files(self):
        fallback = ROOT / "outputs/manuscript_figures_v03/Figure7B_participant_itpc_contrasts.csv"
        with tempfile.TemporaryDirectory() as temporary:
            empty_subject_root = Path(temporary) / "subjects"
            empty_subject_root.mkdir()
            with patch.object(figure7, "SUBJECTS", empty_subject_root):
                frame, inputs = figure7.load_subject_neural(fallback)
        self.assertEqual(len(frame), 35)
        self.assertEqual(inputs, [fallback])
        self.assertIn("itpc_theta_plus_minus_non_rhythmic", frame.columns)

    def test_supplement_accepts_deidentified_flow_without_subject_files(self):
        fallback = ROOT / "outputs/manuscript_supplement_v03/TableS6_dryad_participant_flow.csv"
        with tempfile.TemporaryDirectory() as temporary:
            empty_subject_root = Path(temporary) / "subjects"
            empty_subject_root.mkdir()
            with patch.object(supplement, "SUBJECTS", empty_subject_root):
                frame, inputs = supplement.subject_flow_table(fallback)
        self.assertEqual(len(frame), 35)
        self.assertEqual(inputs, [fallback])
        self.assertEqual(frame.iloc[0]["analytic_id"], "C01")
        self.assertEqual(frame.iloc[-1]["analytic_id"], "C35")


if __name__ == "__main__":
    unittest.main()
