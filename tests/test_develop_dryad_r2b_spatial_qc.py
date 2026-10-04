import unittest

import mne
import numpy as np
import pandas as pd

from scripts.develop_dryad_r2b_spatial_qc import (
    adjacency_map,
    biosemi64_positions,
    build_adjacency,
    flag_channels_r2b_development,
    require_technical_only_schema,
    r2b_channel_metrics,
    spatial_correlation_metrics,
)


class DryadR2bSpatialQCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.edges = build_adjacency(4)
        cls.neighbours = adjacency_map(cls.edges)
        cls.channels = sorted(cls.neighbours)

    def test_graph_covers_exact_biosemi64_and_has_minimum_degree_four(self):
        self.assertEqual(set(self.neighbours), set(biosemi64_positions()))
        self.assertEqual(len(self.neighbours), 64)
        self.assertGreaterEqual(min(map(len, self.neighbours.values())), 4)
        self.assertTrue((self.edges["channel_a"] < self.edges["channel_b"]).all())

    def test_graph_is_deterministic(self):
        second = build_adjacency(4)
        self.assertEqual(self.edges.to_csv(index=False), second.to_csv(index=False))

    def test_isolated_polarity_reversal_has_negative_local_correlation(self):
        rng = np.random.default_rng(20260923)
        base = rng.normal(size=4000)
        data = np.vstack([base + rng.normal(scale=0.08, size=base.size) for _ in self.channels])
        target = self.channels.index("Cz")
        data[target] *= -1
        metrics = spatial_correlation_metrics(data, self.channels, self.edges).set_index("channel")
        self.assertLess(metrics.loc["Cz", "median_neighbor_correlation"], -0.95)
        self.assertGreater(
            metrics.drop(index="Cz")["median_neighbor_correlation"].median(), 0.95
        )

    def test_coherent_spatial_cluster_is_not_wholesale_labeled_as_isolated(self):
        rng = np.random.default_rng(17)
        base = rng.normal(size=4000)
        data = np.vstack([base + rng.normal(scale=0.08, size=base.size) for _ in self.channels])
        positions = biosemi64_positions()
        posterior = [channel for channel in self.channels if positions[channel][1] < -0.35]
        for channel in posterior:
            data[self.channels.index(channel)] *= -1
        metrics = spatial_correlation_metrics(data, self.channels, self.edges).set_index("channel")
        positive_local_fraction = float(
            (metrics.loc[posterior, "median_neighbor_correlation"] > 0.8).mean()
        )
        negative_global_fraction = float(
            (metrics.loc[posterior, "median_global_correlation_diagnostic"] < 0).mean()
        )
        self.assertGreaterEqual(positive_local_fraction, 0.75)
        self.assertGreaterEqual(negative_global_fraction, 0.75)

    def test_channel_order_does_not_change_metrics(self):
        rng = np.random.default_rng(41)
        data = rng.normal(size=(len(self.channels), 500))
        original = spatial_correlation_metrics(data, self.channels, self.edges).set_index("channel")
        order = rng.permutation(len(self.channels))
        permuted_names = [self.channels[index] for index in order]
        permuted = spatial_correlation_metrics(data[order], permuted_names, self.edges).set_index("channel")
        np.testing.assert_allclose(
            original.sort_index()["median_neighbor_correlation"],
            permuted.sort_index()["median_neighbor_correlation"],
        )

    def test_development_flag_rejects_isolated_polarity_reversal(self):
        rng = np.random.default_rng(29)
        base = rng.normal(size=4000)
        data = np.vstack([base + rng.normal(scale=0.08, size=base.size) for _ in self.channels])
        data[self.channels.index("Cz")] *= -1
        spatial = spatial_correlation_metrics(data, self.channels, self.edges)
        metrics = spatial.assign(
            finite_fraction=1.0,
            rms_uv=np.linspace(9.9, 10.1, len(spatial)),
            line_noise_db=0.0,
            high_frequency_db=-10.0,
        )
        flagged = flag_channels_r2b_development(metrics).set_index("channel")
        self.assertTrue(bool(flagged.loc["Cz", "r2b_bad_channel_development"]))
        self.assertIn(
            "low_neighbor_correlation_outlier",
            flagged.loc["Cz", "r2b_bad_channel_reason_development"],
        )

    def test_development_flag_keeps_most_of_coherent_posterior_cluster(self):
        rng = np.random.default_rng(31)
        base = rng.normal(size=4000)
        data = np.vstack([base + rng.normal(scale=0.08, size=base.size) for _ in self.channels])
        positions = biosemi64_positions()
        posterior = [channel for channel in self.channels if positions[channel][1] < -0.35]
        for channel in posterior:
            data[self.channels.index(channel)] *= -1
        spatial = spatial_correlation_metrics(data, self.channels, self.edges)
        metrics = spatial.assign(
            finite_fraction=1.0,
            rms_uv=np.linspace(9.9, 10.1, len(spatial)),
            line_noise_db=0.0,
            high_frequency_db=-10.0,
        )
        flagged = flag_channels_r2b_development(metrics).set_index("channel")
        retained_fraction = float(
            (~flagged.loc[posterior, "r2b_bad_channel_development"]).mean()
        )
        self.assertGreaterEqual(retained_fraction, 0.75)

    def test_flat_channel_is_rejected_without_defined_correlation(self):
        metrics = pd.DataFrame(
            {
                "channel": self.channels,
                "finite_fraction": 1.0,
                "rms_uv": [0.0] + [10.0] * 63,
                "line_noise_db": 0.0,
                "high_frequency_db": -10.0,
                "median_neighbor_correlation": [np.nan] + [0.8] * 63,
                "maximum_neighbor_correlation": [np.nan] + [0.9] * 63,
            }
        )
        flagged = flag_channels_r2b_development(metrics)
        self.assertTrue(bool(flagged.iloc[0]["r2b_bad_channel_development"]))
        self.assertIn("flat", flagged.iloc[0]["r2b_bad_channel_reason_development"])

    def test_near_duplicate_neighbor_pair_is_flagged_as_possible_bridge(self):
        rng = np.random.default_rng(37)
        data = rng.normal(size=(len(self.channels), 5000))
        channel = "Cz"
        neighbour = self.neighbours[channel][0]
        data[self.channels.index(neighbour)] = (
            data[self.channels.index(channel)]
            + rng.normal(scale=1e-4, size=data.shape[1])
        )
        spatial = spatial_correlation_metrics(data, self.channels, self.edges)
        metrics = spatial.assign(
            finite_fraction=1.0,
            rms_uv=np.linspace(9.9, 10.1, len(spatial)),
            line_noise_db=0.0,
            high_frequency_db=-10.0,
        )
        flagged = flag_channels_r2b_development(metrics).set_index("channel")
        for member in (channel, neighbour):
            self.assertIn(
                "possible_spatial_bridge",
                flagged.loc[member, "r2b_bad_channel_reason_development"],
            )

    def test_line_and_high_frequency_outliers_remain_independently_flagged(self):
        metrics = pd.DataFrame(
            {
                "channel": self.channels,
                "finite_fraction": 1.0,
                "rms_uv": np.linspace(9.9, 10.1, len(self.channels)),
                "line_noise_db": [12.0, 0.0] + list(np.linspace(-0.1, 0.1, 62)),
                "high_frequency_db": [-10.0, 5.0] + list(np.linspace(-10.1, -9.9, 62)),
                "median_neighbor_correlation": np.linspace(0.79, 0.81, 64),
                "maximum_neighbor_correlation": 0.9,
            }
        )
        flagged = flag_channels_r2b_development(metrics)
        self.assertIn(
            "line_noise_outlier",
            flagged.iloc[0]["r2b_bad_channel_reason_development"],
        )
        self.assertIn(
            "high_frequency_outlier",
            flagged.iloc[1]["r2b_bad_channel_reason_development"],
        )

    def test_technical_schema_guard_rejects_result_bearing_column(self):
        safe = pd.DataFrame({"channel": ["Cz"], "rms_uv": [10.0]})
        require_technical_only_schema(safe, {"n_scalp_channels": 64})
        unsafe = safe.assign(condition=["f_theta_plus"])
        with self.assertRaisesRegex(RuntimeError, "prohibited result fields"):
            require_technical_only_schema(unsafe, {"n_scalp_channels": 64})

    def test_rawarray_channel_pipeline_emits_only_technical_rows(self):
        rng = np.random.default_rng(43)
        sfreq = 1024.0
        samples = int(20 * sfreq)
        data = rng.normal(scale=5e-6, size=(len(self.channels), samples))
        info = mne.create_info(self.channels, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose="error")
        metrics = r2b_channel_metrics(raw, self.channels, self.edges)
        self.assertEqual(len(metrics), 64)
        self.assertEqual(set(metrics["channel"]), set(self.channels))
        require_technical_only_schema(metrics, {"n_scalp_channels": 64})


if __name__ == "__main__":
    unittest.main()
