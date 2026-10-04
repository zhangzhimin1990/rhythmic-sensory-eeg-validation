import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "30_稿件关键数字溯源表.csv"
MANUSCRIPT = ROOT / "107_Alzheimers_Dementia英文全文_v0.1.md"
LOCKED = ROOT / "outputs/manuscript_supplement_v03/TableS7_dryad_confirmatory_estimates.csv"


class KeyNumberTraceabilityTests(unittest.TestCase):
    def test_claim_ids_are_unique_and_sequential(self):
        with TRACE.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ids = [row["claim_id"] for row in rows]
        self.assertEqual(ids, [f"N{index:03d}" for index in range(1, 58)])

    def test_locked_confirmatory_claims_have_source_files(self):
        with TRACE.open(encoding="utf-8", newline="") as stream:
            rows = {row["claim_id"]: row for row in csv.DictReader(stream)}
        for claim_id in ("N051", "N052", "N053", "N054", "N055", "N056", "N057"):
            self.assertEqual(rows[claim_id]["evidence_status"], "verified_locked")
            for source in rows[claim_id]["source_file"].split("; "):
                self.assertTrue((ROOT / source).is_file(), source)

    def test_locked_table_preserves_primary_neural_and_behavioral_values(self):
        with LOCKED.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        keyed = {(row["family"], row["endpoint"], row["comparison"]): row for row in rows}
        primary = keyed[("primary_neural", "mean_itpc_in_stimulation_window", "theta_plus_minus_non_rhythmic")]
        rt = keyed[("behaviour", "median_correct_rt_ms", "theta_plus_minus_non_rhythmic")]
        ies = keyed[("behaviour", "inverse_efficiency_ms", "theta_plus_minus_non_rhythmic")]
        self.assertAlmostEqual(float(primary["estimate"]), 0.0876443700466359)
        self.assertAlmostEqual(float(rt["estimate"]), -27.727399553571427)
        self.assertAlmostEqual(float(ies["estimate"]), -25.85613522516293)

    def test_manuscript_uses_rounded_locked_values_and_claim_boundary(self):
        text = MANUSCRIPT.read_text(encoding="utf-8")
        for fragment in (
            "mean paired difference 0.088, 95% CI 0.051 to 0.124",
            "27.727 ms relative to the non-rhythmic control",
            "25.856 ms (95% CI −35.455 to −16.257",
            "all Holm-adjusted p values were 1.000",
            "not evidence that individual benefit is intrinsically unpredictable",
        ):
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
