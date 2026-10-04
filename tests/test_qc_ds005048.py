import unittest

import numpy as np
import pandas as pd

from scripts.qc_ds005048 import (
    BROAD_NOISE_BANDS,
    NARROW_NOISE_BANDS,
    normalize_marker,
    snr_db,
    summarize_channels,
)


class Ds005048QcTests(unittest.TestCase):
    def test_marker_normalization_accepts_bids_and_mne_forms(self):
        self.assertEqual(normalize_marker("2"), 2)
        self.assertEqual(normalize_marker("2.0"), 2)
        self.assertIsNone(normalize_marker("Stimulus"))

    def test_snr_is_ten_db_for_tenfold_target_to_neighbor_ratio(self):
        frequencies = np.arange(0.0, 60.25, 0.25)
        psd = np.ones((2, len(frequencies)))
        target_index = int(np.where(frequencies == 40.0)[0][0])
        psd[:, target_index] = 10.0
        narrow = snr_db(psd, frequencies, 40.0, NARROW_NOISE_BANDS)
        broad = snr_db(psd, frequencies, 40.0, BROAD_NOISE_BANDS)
        np.testing.assert_allclose(narrow, [10.0, 10.0], atol=1e-12)
        np.testing.assert_allclose(broad, [10.0, 10.0], atol=1e-12)

    def test_channel_summary_uses_first_six_blocks_only(self):
        rows = []
        for block_index in range(1, 8):
            for condition, snr in (("rest", 0.0), ("stimulus", 2.0)):
                rows.append(
                    {
                        "participant_id": "sub-test",
                        "channel": "Fz",
                        "condition": condition,
                        "block_index": block_index,
                        "snr_narrow_db": 100.0 if block_index == 7 else snr,
                    }
                )
        subject_channels, summary = summarize_channels(pd.DataFrame(rows))
        self.assertEqual(len(subject_channels), 1)
        self.assertAlmostEqual(subject_channels.loc[0, "snr_delta_db"], 2.0)
        self.assertAlmostEqual(summary.loc[0, "median"], 2.0)


if __name__ == "__main__":
    unittest.main()
