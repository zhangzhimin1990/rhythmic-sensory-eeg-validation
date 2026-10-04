import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "107_Alzheimers_Dementia英文全文_v0.1.md"
MANIFEST = ROOT / "107_Alzheimers_Dementia英文全文_v0.1.manifest.json"
PACKAGE = ROOT / "106_Alzheimers_Dementia投稿适配包_v1.md"


def words(text):
    return len(re.findall(r"\b[\w’–-]+\b", text, flags=re.UNICODE))


def between(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0].strip()


class AlzheimersDementiaManuscriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = MANUSCRIPT.read_text(encoding="utf-8")
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.package = PACKAGE.read_text(encoding="utf-8")

    def test_manifest_matches_manuscript(self):
        self.assertEqual(
            self.manifest["sha256"], hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest()
        )

    def test_journal_length_and_display_constraints(self):
        self.assertLessEqual(self.manifest["title_characters"], 85)
        counts = self.manifest["section_word_counts"]
        self.assertLessEqual(counts["abstract"], 250)
        self.assertLessEqual(counts["research_in_context"], 150)
        self.assertLessEqual(counts["introduction"], 650)
        self.assertLessEqual(counts["discussion"], 1500)
        self.assertLessEqual(self.manifest["main_display_items"]["total"], 8)

    def test_required_submission_components(self):
        for label in ["INTRODUCTION:", "METHODS:", "RESULTS:", "DISCUSSION:"]:
            self.assertIn(label, between(self.text, "# Abstract", "# Research in Context"))
        context = between(self.text, "# Research in Context", "# 1. Introduction")
        for label in ["Systematic review:", "Interpretation:", "Future directions:"]:
            self.assertIn(label, context)

    def test_highlights_meet_character_limit(self):
        block = between(self.package, "## Highlights", "## Abbreviated summary")
        bullets = [line[2:] for line in block.splitlines() if line.startswith("- ")]
        self.assertGreaterEqual(len(bullets), 3)
        self.assertLessEqual(len(bullets), 5)
        self.assertTrue(all(len(item) <= 85 for item in bullets))

    def test_numbering_and_table_reallocation(self):
        for heading in ["# 1. Introduction", "# 2. Methods", "# 3. Results", "# 4. Discussion"]:
            self.assertIn(heading, self.text)
        main_legends = between(self.text, "# Table legend", "# Figure legends")
        self.assertIn("## Table 1", main_legends)
        self.assertNotIn("Table 2", main_legends)
        self.assertIn("## Supplementary Table S8", self.text)
        self.assertEqual(self.text.count("## Figure "), 7)

    def test_claim_boundary_and_latest_review_are_retained(self):
        self.assertIn("@L043_Qi2026", self.text)
        self.assertIn(
            "Sensory target engagement is therefore an initial observation, not by itself a cognitive biomarker",
            self.text,
        )
        self.assertNotIn("Figure 7 remains conditional", self.text)
        self.assertIsNone(re.search(r"\[L\d{3}", self.text))


if __name__ == "__main__":
    unittest.main()
