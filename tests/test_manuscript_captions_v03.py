import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = ROOT / "82_正文Figure1与Table2英文图表注_v0.1.md"
TRANSITIONS = ROOT / "outputs" / "manuscript_tables_v02" / "Table2_inferential_transitions.csv"


class ManuscriptCaptionsV03Tests(unittest.TestCase):
    def test_caption_covers_every_main_table_transition_and_decision(self):
        text = CAPTIONS.read_text(encoding="utf-8")
        with TRANSITIONS.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.assertIn(row["Transition"], text)
            self.assertIn(row["Decision"], text)

    def test_caption_preserves_design_and_scope_boundaries(self):
        text = CAPTIONS.read_text(encoding="utf-8")
        required = [
            "do not encode effect direction, evidence strength, or a meta-analytic weight",
            "participants and raw EEG amplitudes are not pooled",
            "do not test chronic treatment efficacy or disease modification",
            "Within-recording split-half reliability is not between-day test–retest reliability",
        ]
        for phrase in required:
            self.assertIn(phrase, text)
        self.assertNotIn("private cohort", text.lower())


if __name__ == "__main__":
    unittest.main()
