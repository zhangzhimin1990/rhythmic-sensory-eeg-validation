import unittest
import csv
from pathlib import Path

import numpy as np

from scripts.build_behavioral_validity_figures import PDF_METADATA as BEHAVIOR_PDF_METADATA
from scripts.build_dryad_theta_figure import PDF_METADATA as DRYAD_PDF_METADATA
from scripts.build_foundation_figures_v02 import PDF_METADATA, bootstrap_spearman_ci


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "outputs" / "manuscript_figures_v02"


class FoundationFiguresV02Tests(unittest.TestCase):
    def test_manuscript_pdf_metadata_excludes_generation_timestamps(self):
        for metadata in (PDF_METADATA, BEHAVIOR_PDF_METADATA, DRYAD_PDF_METADATA):
            self.assertIsNone(metadata["CreationDate"])
            self.assertIsNone(metadata["ModDate"])
            self.assertEqual(metadata["Creator"], "public-eeg-validation")

    def test_bootstrap_spearman_is_deterministic(self):
        x = np.arange(20, dtype=float)
        y = x * 2
        first = bootstrap_spearman_ci(x, y, n_boot=500, seed=11)
        second = bootstrap_spearman_ci(x, y, n_boot=500, seed=11)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 1.0)
        self.assertEqual(first[3], 20)

    def test_bootstrap_spearman_drops_missing_pairs(self):
        result = bootstrap_spearman_ci([1, 2, np.nan, 4], [1, 2, 3, 4], n_boot=100, seed=3)
        self.assertEqual(result[3], 3)

    def test_figure1_role_matrix_covers_seven_cohorts_and_four_transitions(self):
        with (FIGURE_DIR / "Figure1_transition_role_matrix.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 28)
        self.assertEqual(len({row["dataset"] for row in rows}), 7)
        self.assertEqual(
            {row["transition"].split()[0] for row in rows},
            {"T1", "T2", "T3", "T4"},
        )
        direct = {
            (row["dataset"], row["transition"].split()[0])
            for row in rows
            if row["role"] == "Direct"
        }
        self.assertIn(("ds006036 visual", "T1"), direct)
        self.assertIn(("ds005048 auditory", "T2"), direct)
        self.assertIn(("Dryad theta", "T3"), direct)
        self.assertIn(("Dryad theta", "T4"), direct)

    def test_figure1_transition_evidence_preserves_claim_boundaries(self):
        with (FIGURE_DIR / "Figure1_transition_evidence.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["transition_id"] for row in rows], ["T1", "T2", "T3", "T4"])
        results = {row["transition_id"]: row["result"] for row in rows}
        self.assertEqual(results["T1"], "Not established")
        self.assertEqual(results["T2"], "Not established in tested settings")
        self.assertEqual(results["T3"], "Not supported for tested coupling estimand")
        self.assertEqual(results["T4"], "No consistent increment")


if __name__ == "__main__":
    unittest.main()
