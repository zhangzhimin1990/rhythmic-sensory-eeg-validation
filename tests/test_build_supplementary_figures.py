import unittest

from scripts.build_supplementary_figures import q_family_summary, visual_block_tables


class TestBuildSupplementaryFigures(unittest.TestCase):
    def test_visual_block_flow_is_frozen(self):
        blocks, flow = visual_block_tables()
        self.assertEqual(len(blocks), 397)
        self.assertEqual(flow["n_blocks"].tolist(), [407, 372, 351])
        self.assertEqual(sorted(blocks["frequency_hz"].unique().tolist()), [5.0, 10.0, 15.0, 20.0])

    def test_q_family_minima_distinguish_full_and_incremental_auditory_sets(self):
        summary = q_family_summary().set_index("analysis_family")
        self.assertAlmostEqual(summary.loc["Auditory MMSE · exploratory", "minimum_q_bh"], 0.8826937567925888)
        self.assertAlmostEqual(summary.loc["Auditory exploratory · incremental subset", "minimum_q_bh"], 0.9953462621295998)
        self.assertAlmostEqual(summary.loc["Rest → advanced visual", "minimum_q_bh"], 0.1098652639162452)


if __name__ == "__main__":
    unittest.main()
