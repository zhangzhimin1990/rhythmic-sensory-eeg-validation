import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class DryadThetaFigureTests(unittest.TestCase):
    def test_active_control_panel_preserves_direction_and_accuracy_boundary(self):
        paired = pd.read_csv(
            ROOT / "outputs/models/dryad_theta_validation_gate/paired_condition_effects.csv"
        )
        personalized = paired.loc[paired.contrast.eq("personalized_mean_minus_2Hz")].iloc[0]
        accuracy = paired.loc[
            paired.contrast.eq("personalized_mean_minus_2Hz_accuracy")
        ].iloc[0]
        self.assertLess(personalized.estimate, 0)
        self.assertLess(personalized.ci95_high, 0)
        self.assertLess(accuracy.estimate, 0)

    def test_between_within_panel_keeps_validation_levels_separate(self):
        models = pd.read_csv(
            ROOT / "outputs/models/dryad_theta_validation_gate/association_models.csv"
        )
        selected = models.loc[
            models.model.eq("continuous_post_rt_ms")
            & models.term.isin(["entrainment_between_z", "entrainment_within_z"])
        ]
        self.assertEqual(len(selected), 2)
        between = selected.loc[selected.term.eq("entrainment_between_z")].iloc[0]
        within = selected.loc[selected.term.eq("entrainment_within_z")].iloc[0]
        self.assertLess(between.ci95_high, 0)
        self.assertLess(within.ci95_low, 0)
        self.assertGreater(within.ci95_high, 0)


if __name__ == "__main__":
    unittest.main()
