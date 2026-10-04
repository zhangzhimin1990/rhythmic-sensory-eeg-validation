import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = ROOT / "83_正文Figure2-7英文图注_v0.1.md"


class ManuscriptCaptionsV04Tests(unittest.TestCase):
    def test_all_remaining_figure_legends_are_present(self):
        text = CAPTIONS.read_text(encoding="utf-8")
        for number in range(2, 8):
            self.assertIn(f"**Figure {number}.", text)

    def test_legends_preserve_key_claim_boundaries(self):
        text = CAPTIONS.read_text(encoding="utf-8")
        required = [
            "counts must therefore not be added across columns",
            "do not estimate between-day test–retest reliability",
            "not treated as proof of no association",
            "does not show that experimentally increasing the response would improve accuracy",
            "rather than effect sizes, a validation score, or a meta-analysis",
            "completed locked 35-participant confirmatory analysis",
            "but not condition-specific neural–behavioral coupling, mediation, a deployable treatment-selection rule, chronic efficacy, or disease modification",
        ]
        for phrase in required:
            self.assertIn(phrase, text)
        self.assertNotIn("private cohort", text.lower())


if __name__ == "__main__":
    unittest.main()
