import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_behavioral_validity_figures import load_figure5_data, load_figure6_data


class BehavioralValidityFigureTests(unittest.TestCase):
    def test_figure5_uses_two_frequencies_and_three_dprime_metrics(self):
        accuracy, rt, scatter, dprime = load_figure5_data()
        self.assertEqual(set(accuracy["label"]), {"36 Hz", "40 Hz"})
        self.assertEqual(set(rt["label"]), {"36 Hz", "40 Hz"})
        self.assertEqual(set(scatter["frequency_hz"]), {36, 40})
        self.assertEqual(set(dprime["label"]), {"Local log-SNR", "Morlet power", "ITPC"})

    def test_figure6_preserves_prediction_direction_and_matrix_coverage(self):
        _, _, prediction, matrix = load_figure6_data()
        dprime = prediction.loc[
            prediction.dataset.eq("ds006780") & prediction.outcome.eq("dprime_40")
        ]
        self.assertTrue((dprime.relative_delta_percent > 0).all())
        self.assertEqual(matrix.dataset.nunique(), 4)
        self.assertEqual(matrix.stage.nunique(), 6)
        self.assertEqual(len(matrix), 24)


if __name__ == "__main__":
    unittest.main()
