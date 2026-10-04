import unittest

import numpy as np
import pandas as pd

from scripts.analyze_estimand_alignment import bootstrap_spearman_ci, single_row


class AnalyzeEstimandAlignmentTests(unittest.TestCase):
    def test_bootstrap_spearman_is_deterministic(self):
        x = pd.Series(np.arange(12, dtype=float))
        y = pd.Series(np.arange(12, dtype=float))
        first = bootstrap_spearman_ci(x, y, seed=7, n_boot=100)
        second = bootstrap_spearman_ci(x, y, seed=7, n_boot=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 1.0)

    def test_single_row_accepts_csv_boolean_strings(self):
        frame = pd.DataFrame({"flag": ["False", "True"], "value": [1, 2]})
        self.assertEqual(single_row(frame, flag=False)["value"], 1)


if __name__ == "__main__":
    unittest.main()
