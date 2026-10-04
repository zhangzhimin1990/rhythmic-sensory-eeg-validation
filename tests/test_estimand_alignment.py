import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EstimandAlignmentTests(unittest.TestCase):
    def test_ds007648_aggregate_reliability_is_not_used_for_trial_inference(self):
        manuscript = (ROOT / "28_英文结果与讨论初稿_v0.2.md").read_text(encoding="utf-8")
        self.assertIn("Aggregate inter-trial phase-coherence reliability cannot be assigned to this trial-state predictor", manuscript)
        self.assertIn("aggregate neural and behavioral reliability coefficients", manuscript)

    def test_table2_preserves_level_boundary(self):
        with (ROOT / "outputs/manuscript_tables_v02/Table2_validation_claims.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        row = next(item for item in rows if item["Validation claim"] == "Aggregate and trial-level 40-Hz behavioral validity")
        self.assertIn("metric and aggregation level are matched", row["Permitted interpretation"])
        self.assertIn("trial-state reliability inferred from aggregate ITPC", row["Prohibited inference"])

    def test_ds006780_reliability_is_labeled_as_aligned(self):
        audit = (ROOT / "69_行为终点测量信度与效度衰减风险审计.md").read_text(encoding="utf-8")
        self.assertIn("ds006780：估计层级对齐", audit)
        self.assertIn("不能跨层解释", audit)

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
