import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EstimandAlignmentTests(unittest.TestCase):
    def test_table2_preserves_level_boundary(self):
        with (ROOT / "outputs/manuscript_tables_v02/Table2_validation_claims.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        row = next(item for item in rows if item["Validation claim"] == "Aggregate and trial-level 40-Hz behavioral validity")
        self.assertIn("metric and aggregation level are matched", row["Permitted interpretation"])
        self.assertIn("trial-state reliability inferred from aggregate ITPC", row["Prohibited inference"])

    def test_ds006780_precision_boundary_stays_metric_specific(self):
        with (ROOT / "outputs/manuscript_tables_v02/Table2_validation_claims.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        row = next(
            item
            for item in rows
            if item["Validation claim"] == "Developmental sensitivity versus d-prime validity"
        )
        self.assertIn("+/-0.30 outcome SD or larger excluded", row["Validation status"])
        self.assertIn("+/-0.25 outcome SD was not excluded", row["Permitted interpretation"])
        self.assertIn("local-SNR precision transferred to ITPC", row["Prohibited inference"])


if __name__ == "__main__":
    unittest.main()
