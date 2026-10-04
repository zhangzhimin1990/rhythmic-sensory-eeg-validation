import unittest

import pandas as pd

from scripts.summarize_dryad_r2b_development import technical_flags


class DryadR2bDevelopmentSummaryTests(unittest.TestCase):
    def base(self):
        return pd.DataFrame(
            {
                "finite_fraction": [1.0],
                "rms_uv": [10.0],
                "z_log_rms": [0.0],
                "z_line_noise_db": [0.0],
                "line_noise_db": [0.0],
                "z_high_frequency_db": [0.0],
                "high_frequency_db": [-10.0],
                "z_low_neighbor_correlation": [0.0],
                "median_neighbor_correlation": [0.8],
                "maximum_neighbor_correlation": [0.9],
            }
        )

    def test_clean_channel_is_not_flagged(self):
        self.assertFalse(technical_flags(self.base(), 5.0, 0.2).iloc[0])

    def test_low_neighbour_requires_both_absolute_and_relative_evidence(self):
        table = self.base()
        table.loc[0, "z_low_neighbor_correlation"] = 6.0
        table.loc[0, "median_neighbor_correlation"] = 0.3
        self.assertFalse(technical_flags(table, 5.0, 0.2).iloc[0])
        table.loc[0, "median_neighbor_correlation"] = -0.3
        self.assertTrue(technical_flags(table, 5.0, 0.2).iloc[0])

    def test_line_noise_requires_both_absolute_and_relative_evidence(self):
        table = self.base()
        table.loc[0, "z_line_noise_db"] = 20.0
        table.loc[0, "line_noise_db"] = 2.0
        self.assertFalse(technical_flags(table, 5.0, 0.2).iloc[0])
        table.loc[0, "line_noise_db"] = 4.0
        self.assertTrue(technical_flags(table, 5.0, 0.2).iloc[0])


if __name__ == "__main__":
    unittest.main()
