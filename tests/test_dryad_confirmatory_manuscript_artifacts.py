import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "manuscript_figures_v03"
SUPPLEMENT = ROOT / "outputs" / "manuscript_supplement_v03"


class DryadConfirmatoryManuscriptArtifactTests(unittest.TestCase):
    def test_figure7_manifest_tracks_locked_claim_ceiling(self):
        manifest = json.loads(
            (FIGURES / "Figure7_dryad_confirmatory_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["n_confirmatory"], 35)
        self.assertEqual(
            manifest["claim_ceiling"],
            "target_engagement_and_behaviour_without_specific_coupling",
        )
        inputs = set(manifest["inputs"])
        full_subject_chain = len(inputs) >= 40
        public_release_fallback = (
            len(inputs) >= 6
            and any(path.endswith("Figure7B_participant_itpc_contrasts.csv") for path in inputs)
        )
        self.assertTrue(full_subject_chain or public_release_fallback)

    def test_main_and_supplementary_figures_are_nonempty(self):
        paths = [
            FIGURES / "Figure7_dryad_confirmatory.png",
            FIGURES / "Figure7_dryad_confirmatory.pdf",
            SUPPLEMENT / "FigureS5_dryad_component_decomposition.png",
            SUPPLEMENT / "FigureS5_dryad_component_decomposition.pdf",
            SUPPLEMENT / "FigureS6_dryad_internal_consistency.png",
            SUPPLEMENT / "FigureS6_dryad_internal_consistency.pdf",
        ]
        for path in paths:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 20_000, path)

    def test_table_s6_is_complete_and_deidentified(self):
        path = SUPPLEMENT / "TableS6_dryad_participant_flow.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 35)
        self.assertEqual(rows[0]["analytic_id"], "C01")
        self.assertEqual(rows[-1]["analytic_id"], "C35")
        forbidden = {"subject", "source_bdf_name", "source_bdf_sha256"}
        self.assertTrue(forbidden.isdisjoint(rows[0]))
        self.assertTrue(all(row["derivation_status"] == "confirmatory_subject_derivation_complete" for row in rows))

    def test_table_s7_preserves_all_inferential_families(self):
        path = SUPPLEMENT / "TableS7_dryad_confirmatory_estimates.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        families = {row["family"] for row in rows}
        self.assertEqual(
            families,
            {
                "primary_neural",
                "key_secondary_neural",
                "behaviour",
                "within_condition_coupling",
                "incremental_prediction",
            },
        )
        self.assertEqual(len(rows), 13)


if __name__ == "__main__":
    unittest.main()
