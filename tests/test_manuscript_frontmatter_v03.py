import csv
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONT_MATTER = ROOT / "81_英文标题摘要与核心贡献_v0.3.md"
INTRO_METHODS = ROOT / "27_英文引言与方法初稿_v0.2.md"
TABLE2 = ROOT / "outputs" / "manuscript_tables_v02" / "Table2_validation_claims.csv"


def section(text: str, heading: str, next_heading: str) -> str:
    return text.split(heading, 1)[1].split(next_heading, 1)[0].strip()


class ManuscriptFrontMatterV03Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.front = FRONT_MATTER.read_text(encoding="utf-8")
        cls.intro = INTRO_METHODS.read_text(encoding="utf-8")
        with TABLE2.open(newline="", encoding="utf-8") as handle:
            cls.claims = list(csv.DictReader(handle))

    def test_title_is_identical_across_front_matter_and_manuscript(self):
        front_title = section(self.front, "# Title", "# Abstract")
        manuscript_title = section(self.intro, "# Working title", "# Introduction")
        self.assertEqual(front_title, manuscript_title)

    def test_abstract_length_is_250_to_300_words(self):
        abstract = section(self.front, "# Abstract", "# Keywords")
        words = re.findall(r"\b[\w’–-]+\b", abstract, flags=re.UNICODE)
        self.assertGreaterEqual(len(words), 250)
        self.assertLessEqual(len(words), 300)

    def test_abstract_declares_seven_cohorts_and_four_transitions(self):
        abstract = section(self.front, "# Abstract", "# Keywords")
        self.assertIn("seven public human EEG datasets", abstract)
        self.assertIn("four inferential transitions", abstract)

    def test_key_estimates_are_traceable_to_table2(self):
        abstract = section(self.front, "# Abstract", "# Keywords")
        table_text = "\n".join(row["Key estimate"] for row in self.claims)
        for token in (
            "32/35",
            "0.838",
            "0.085",
            "0.972",
            "0.931",
            "0.058",
            "0.088",
            "7.52 dB",
            "27.73 ms",
            "25.86 ms",
        ):
            self.assertIn(token, abstract)

        self.assertIn("32/35", table_text)
        self.assertIn("0.838", table_text)
        self.assertIn("0.085", table_text)
        self.assertIn("0.972", table_text)
        self.assertIn("0.931", table_text)
        self.assertIn("0.058", table_text)
        self.assertIn("ITPC=0.088", table_text)
        self.assertIn("7.518 dB", table_text)
        self.assertIn("RT=-27.727 ms", table_text)
        self.assertIn("inverse efficiency=-25.856 ms", table_text)


if __name__ == "__main__":
    unittest.main()
