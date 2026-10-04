import unittest

import numpy as np
import pandas as pd

from scripts.build_manuscript_figures import bootstrap_spearman, fit_rest_to_visual_models


class TestBuildManuscriptFigures(unittest.TestCase):
    def test_rest_to_visual_model_family_is_frozen(self):
        models = fit_rest_to_visual_models()
        self.assertEqual(len(models), 28)
        self.assertEqual(sorted(models["frequency_hz"].unique().tolist()), [5.0, 10.0, 15.0, 20.0])
        self.assertEqual(models["predictor"].nunique(), 7)
        self.assertAlmostEqual(models["q_bh"].min(), 0.6413180695031035, places=10)
        strongest = models.sort_values("p_value").iloc[0]
        self.assertEqual(strongest["predictor"], "posterior_iaf_hz")
        self.assertEqual(strongest["frequency_hz"], 20.0)

    def test_bootstrap_spearman_is_deterministic(self):
        x = np.arange(12, dtype=float)
        y = x**2
        first = bootstrap_spearman(x, y, n_boot=500, seed=7)
        second = bootstrap_spearman(x, y, n_boot=500, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 1.0)


if __name__ == "__main__":
    unittest.main()
