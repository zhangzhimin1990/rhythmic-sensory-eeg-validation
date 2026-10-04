import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "104_英文全文整合稿_v0.5.md"
MANIFEST = ROOT / "104_英文全文整合稿_v0.5.manifest.json"
BIBLIOGRAPHY = ROOT / "references" / "manuscript_references_v02.bib"


class ConsolidatedManuscriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = MANUSCRIPT.read_text(encoding="utf-8")
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_matches_current_manuscript(self):
        digest = hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest()
        self.assertEqual(self.manifest["sha256"], digest)
        body = self.text.split("# Table legends", 1)[0]
        words = re.findall(r"\b[\w’–-]+\b", body, flags=re.UNICODE)
        self.assertEqual(self.manifest["body_word_count_including_abstract"], len(words))

    def test_no_internal_scaffolding_or_unresolved_citations(self):
        forbidden = [
            "# Scope notice",
            "# Internal citation crosswalk",
            "Figure 7 remains conditional",
        ]
        for token in forbidden:
            self.assertNotIn(token, self.text)
        self.assertIsNone(re.search(r"\[L\d{3}", self.text))

    def test_every_citation_key_exists_in_bibliography(self):
        cited = set(re.findall(r"@([A-Za-z0-9_]+)", self.text))
        bib = BIBLIOGRAPHY.read_text(encoding="utf-8")
        available = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
        self.assertTrue(cited)
        self.assertEqual(cited, set(self.manifest["citation_keys"]))
        self.assertFalse(cited - available, cited - available)
        self.assertIn("L043_Qi2026", cited)

    def test_primary_locked_results_and_claim_boundary_are_present(self):
        required = [
            "0.088 (95% CI 0.051 to 0.124)",
            "7.52 dB (95% CI 4.52 to 10.51)",
            "27.73 ms (95% CI −36.17 to −19.28)",
            "25.86 ms (95% CI −35.45 to −16.26)",
            "not a condition-specific neural–behavioral mechanism proxy",
        ]
        for token in required:
            self.assertIn(token, self.text)

    def test_section_and_legend_inventory(self):
        for heading in ["# Abstract", "# Introduction", "# Methods", "# Results", "# Discussion"]:
            self.assertEqual(self.text.count(heading + "\n"), 1, heading)
        for number in range(1, 8):
            self.assertEqual(self.text.count(f"## Figure {number}\n"), 1)
        for number in range(1, 7):
            self.assertEqual(self.text.count(f"## Supplementary Figure S{number}."), 1)
        self.assertIn("## Supplementary Table S6", self.text)
        self.assertIn("## Supplementary Table S7", self.text)


if __name__ == "__main__":
    unittest.main()
