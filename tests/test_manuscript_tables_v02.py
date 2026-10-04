import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "manuscript_tables_v02"


def read_rows(name: str):
    with (TABLE_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ManuscriptTablesV02Tests(unittest.TestCase):
    def test_table1_preserves_seven_public_dataset_roles(self):
        rows = read_rows("Table1_public_cohorts.csv")
        self.assertEqual(len(rows), 7)
        datasets = {row["Dataset"] for row in rows}
        self.assertEqual(
            datasets,
            {
                "ds004504 v1.0.9",
                "ds006036 v1.0.6",
                "ds005048 v1.0.1",
                "ds006222 v1.0.1",
                "ds007648 v1.1.0",
                "ds006780 v1.0.0",
                "Dryad t76hdr8dm",
            },
        )
        joined = " ".join(" ".join(row.values()) for row in rows).lower()
        self.assertNotIn("private cohort", joined)
        self.assertIn("same participants and ids as ds006036", joined)
        self.assertIn("same participants and ids as ds004504", joined)
        self.assertIn("raw bdf gate pending", joined)

    def test_table1_keeps_outcome_specific_denominators(self):
        rows = {row["Dataset"]: row for row in read_rows("Table1_public_cohorts.csv")}
        self.assertIn("N=20", rows["ds007648 v1.1.0"]["Primary effective sample"])
        self.assertIn("8579", rows["ds007648 v1.1.0"]["Primary effective sample"])
        self.assertIn("Technical N=111", rows["ds006780 v1.0.0"]["Primary effective sample"])
        self.assertIn("FSIQ N=109", rows["ds006780 v1.0.0"]["Primary effective sample"])
        self.assertIn("N=44", rows["Dryad t76hdr8dm"]["Primary effective sample"])
        self.assertIn("132", rows["Dryad t76hdr8dm"]["Primary effective sample"])

    def test_table2_keeps_nine_claims_and_boundaries(self):
        rows = read_rows("Table2_validation_claims.csv")
        self.assertEqual(len(rows), 9)
        claims = {row["Validation claim"] for row in rows}
        self.assertIn("Frozen personalized-theta target engagement and acute behavior", claims)
        self.assertIn("Frozen personalized-theta coupling and prediction", claims)
        joined = " ".join(" ".join(row.values()) for row in rows).lower()
        self.assertIn(
            "reliability does not guarantee validity when metric and aggregation level are matched",
            joined,
        )
        self.assertIn("causal mediation", joined)
        self.assertIn("long-term or clinical efficacy", joined)
        self.assertNotIn("private", joined)

    def test_main_table2_has_four_distinct_inferential_transitions(self):
        rows = read_rows("Table2_inferential_transitions.csv")
        self.assertEqual([row["Transition"] for row in rows], ["T1", "T2", "T3", "T4"])
        joined = " ".join(" ".join(row.values()) for row in rows).lower()
        self.assertIn("state-general endogenous mechanism", joined)
        self.assertIn("matched cognitive validity", joined)
        self.assertIn("within-participant condition-specific neural–behavioral coupling", joined)
        self.assertIn("held-out incremental individual prediction", joined)
        self.assertIn("not supported for tested coupling estimand", joined)
        self.assertIn("no consistent predictive increment", joined)

    def test_transition_table_preserves_positive_antecedents_and_prohibited_upgrades(self):
        rows = {row["Transition"]: row for row in read_rows("Table2_inferential_transitions.csv")}
        self.assertIn("ITPC difference=0.088", rows["T3"]["Key result"])
        self.assertIn("RT=-27.727 ms", rows["T3"]["Key result"])
        self.assertIn("all four coupling Holm p=1.000", rows["T3"]["Key result"])
        self.assertIn("reliability=0.972", rows["T2"]["Key result"])
        self.assertIn("necessarily cognitively informative", rows["T2"]["Prohibited inference"])
        self.assertIn("mediates", rows["T3"]["Prohibited inference"])
        self.assertIn("deployable", rows["T4"]["Prohibited inference"])


if __name__ == "__main__":
    unittest.main()
