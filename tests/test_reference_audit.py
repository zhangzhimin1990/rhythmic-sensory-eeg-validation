import csv
import re
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "62_参考文献管理器级核验.csv"
BIB = ROOT / "references" / "manuscript_references_v02.bib"
MANUSCRIPTS = [
    ROOT / "27_英文引言与方法初稿_v0.2.md",
    ROOT / "28_英文结果与讨论初稿_v0.2.md",
]


def load_rows():
    with AUDIT.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ReferenceAuditTests(unittest.TestCase):
    @staticmethod
    def manuscript_keys():
        used = set()
        for path in MANUSCRIPTS:
            used.update(re.findall(r"L\d{3}", path.read_text(encoding="utf-8")))
        return used

    def test_all_manuscript_keys_resolve(self):
        used = self.manuscript_keys()
        audited = {row["internal_key"] for row in load_rows()}
        self.assertEqual(used - audited, set())
        self.assertIn("L040", used)
        self.assertIn("L041", used)
        self.assertIn("L042", used)
        self.assertIn("L013", used)

    def test_required_audit_fields_are_populated(self):
        required = {
            "internal_key", "title", "first_author", "year",
            "doi_or_registry", "publication_status", "verified_source",
            "verification_status", "notes",
        }
        for row in load_rows():
            self.assertTrue(required.issubset(row))
            for field in required:
                self.assertTrue(row[field].strip(), f"{row['internal_key']} missing {field}")

    def test_only_intentional_identifier_duplicate(self):
        by_identifier = defaultdict(list)
        for row in load_rows():
            by_identifier[row["doi_or_registry"].lower()].append(row["internal_key"])
        duplicates = {key: sorted(value) for key, value in by_identifier.items() if len(value) > 1}
        self.assertEqual(
            duplicates,
            {"10.3389/fneur.2024.1343588": ["L020", "L027"]},
        )

    def test_bibtex_covers_all_unique_sources(self):
        bib = BIB.read_text(encoding="utf-8").lower()
        identifiers = {row["doi_or_registry"].lower() for row in load_rows()}
        for identifier in identifiers:
            self.assertIn(identifier, bib)
        self.assertEqual(len(re.findall(r"^@", bib, flags=re.MULTILINE)), len(identifiers))

    def test_nonstandard_sources_are_labeled(self):
        rows = {row["internal_key"]: row for row in load_rows()}
        self.assertEqual(rows["L024"]["publication_status"], "preprint")
        self.assertEqual(rows["L037"]["publication_status"], "reviewed preprint")
        self.assertEqual(rows["L036"]["verification_status"], "verified dynamic")
        self.assertIn("submission", rows["L036"]["notes"].lower())
        self.assertEqual(rows["L042"]["publication_status"], "trial registration")
        self.assertEqual(rows["L042"]["verification_status"], "verified dynamic")
        self.assertIn("no posted results", rows["L042"]["notes"].lower())
        self.assertIn("young healthy", rows["L041"]["notes"].lower())

    def test_reference_summary_counts_match_manuscript(self):
        used = self.manuscript_keys()
        rows = [row for row in load_rows() if row["internal_key"] in used]
        unique_sources = {row["doi_or_registry"].lower() for row in rows}
        summary = (ROOT / "63_参考文献管理器级核验说明.md").read_text(encoding="utf-8")
        self.assertIn(f"使用{len(used)}个内部键", summary)
        self.assertIn(f"对应{len(unique_sources)}个唯一来源", summary)


if __name__ == "__main__":
    unittest.main()
