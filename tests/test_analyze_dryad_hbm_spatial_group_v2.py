import numpy as np
import pandas as pd

from scripts.analyze_dryad_hbm_spatial_group_v2 import cluster_permutation


def test_no_observed_cluster_returns_empty_list_with_no_inference_error():
    rng = np.random.default_rng(17)
    differences = pd.DataFrame(
        rng.normal(0, 0.001, size=(24, 4)), columns=["A", "B", "C", "D"]
    )
    differences = differences - differences.mean(axis=0)
    adjacency = {"A": {"B"}, "B": {"A", "C"}, "C": {"B", "D"}, "D": {"C"}}
    assert cluster_permutation(differences, adjacency, permutations=100, seed=9) == []
