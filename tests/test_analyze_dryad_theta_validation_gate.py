import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_dryad_theta_validation_gate import make_long, residualize


class DryadThetaValidationGateTests(unittest.TestCase):
    def test_residualize_removes_design_projection(self):
        design = np.column_stack([np.ones(6), np.arange(6)])
        vector = 3 + 2 * np.arange(6) + np.array([1, -1, 1, -1, 1, -1])
        residual = residualize(vector.astype(float), design.astype(float))
        self.assertTrue(np.allclose(design.T @ residual, 0, atol=1e-10))

    def test_make_long_decomposes_entrainment(self):
        rows = []
        for subject in (1, 2):
            rows.append(
                {
                    "subject": subject,
                    "gender": "F",
                    "age": 65 + subject,
                    "individual_freq": 3.0 + subject,
                    "bl_rt": 500 + subject,
                    "cat_rt": "Good",
                    "cat_entrain": "High",
                    "doshz_rt": 490,
                    "fθ_rt": 480,
                    "fθ+_rt": 475,
                    "doshz_acc_con": 0.98,
                    "doshz_acc_incon": 0.95,
                    "fθ_acc_con": 0.98,
                    "fθ_acc_incon": 0.95,
                    "fθ+_acc_con": 0.98,
                    "fθ+_acc_incon": 0.95,
                    "base_doshz_rt_percent": 2,
                    "base_fθ_rt_percent": 4,
                    "base_fθ+_rt_percent": 5,
                    "entrain_doshz": 1.0 * subject,
                    "entrain_fθ": 2.0 * subject,
                    "entrain_fθ+": 3.0 * subject,
                }
            )
        long = make_long(pd.DataFrame(rows))
        self.assertEqual(len(long), 6)
        sums = long.groupby("subject")["entrainment_within"].sum()
        self.assertTrue(np.allclose(sums, 0))
        self.assertEqual(long.groupby("subject").size().tolist(), [3, 3])


if __name__ == "__main__":
    unittest.main()
