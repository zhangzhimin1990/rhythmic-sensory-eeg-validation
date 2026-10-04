import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "81_英文标题摘要与核心贡献_v0.3.md"
INTRO = ROOT / "27_英文引言与方法初稿_v0.2.md"
RESULTS = ROOT / "28_英文结果与讨论初稿_v0.2.md"
CAPTIONS_1 = ROOT / "82_正文Figure1与Table2英文图表注_v0.1.md"
CAPTIONS_2 = ROOT / "83_正文Figure2-7英文图注_v0.1.md"
TERMS = ROOT / "configs" / "manuscript_canonical_terms_v1.csv"


class ManuscriptConsistencyV04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.front = FRONT.read_text(encoding="utf-8")
        cls.intro = INTRO.read_text(encoding="utf-8")
        cls.results = RESULTS.read_text(encoding="utf-8")
        cls.captions = CAPTIONS_1.read_text(encoding="utf-8") + CAPTIONS_2.read_text(encoding="utf-8")

    def test_inferential_transition_framework_is_canonical(self):
        for text in (self.front, self.intro, self.results, self.captions):
            self.assertIn("inferential transitions", text)
        self.assertNotIn("testing four validation transitions", self.front.lower())
        self.assertNotIn("testing four validation transitions", self.intro.lower())

    def test_results_no_longer_use_validation_ladder_headings(self):
        self.assertNotIn("## Validation level", self.results)
        self.assertNotIn("this hierarchy", self.results)

    def test_results_reference_all_frozen_main_figures(self):
        for number in range(1, 8):
            self.assertIn(f"Figure {number}", self.results)
        self.assertIn(
            "Figure 7 incorporates the completed locked 35-participant confirmatory analysis",
            CAPTIONS_2.read_text(encoding="utf-8"),
        )

    def test_canonical_term_ledger_is_resolved_and_unique(self):
        with TERMS.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["concept_id"] for row in rows}), 8)
        self.assertTrue(all(row["status"] == "resolved" for row in rows))


if __name__ == "__main__":
    unittest.main()
