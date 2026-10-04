import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_ds007648_behavioral_validity import benjamini_hochberg, within_subject_zscore


class Ds007648BehavioralValidityTests(unittest.TestCase):
    def test_within_subject_zscore_removes_subject_means(self):
        frame = pd.DataFrame({"subject": ["a"] * 3 + ["b"] * 3, "x": [1, 2, 3, 10, 12, 14]})
        result = within_subject_zscore(frame, "x")
        means = result.groupby(frame.subject).mean()
        self.assertTrue(np.allclose(means, 0))

    def test_bh_is_monotone_in_rank(self):
        pvalues = np.array([0.04, 0.001, 0.03, 0.2])
        adjusted = benjamini_hochberg(pvalues)
        order = np.argsort(pvalues)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= 0))
        self.assertTrue(np.all(adjusted >= pvalues))


if __name__ == "__main__":
    unittest.main()
