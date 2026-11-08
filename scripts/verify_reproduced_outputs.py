#!/usr/bin/env python3
"""Compare reproduced focused-analysis CSV outputs with the archived release values."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


GROUPS = {
    "estimand_alignment": ["matched_reliability_validity.csv"],
    "hbm_prediction_audit_v1": [
        "prediction_audit_summary.csv",
        "calibration_summary.csv",
        "participant_loss_increments.csv",
        "held_out_prediction_rows.csv",
        "participant_losses.csv",
    ],
    "hbm_reviewer_uncertainty_audit_v1": [
        "prediction_increment_bootstrap.csv",
        "matched_itpc_reliability_bootstrap.csv",
    ],
    "dryad_hbm_spatial_group_v1": [
        "roi_specificity.csv",
        "spatial_clusters.csv",
        "channel_effects.csv",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=Path("outputs/models"))
    parser.add_argument("--reproduced-root", type=Path, required=True)
    args = parser.parse_args()
    checked = 0
    for group, names in GROUPS.items():
        for name in names:
            archived = pd.read_csv(args.archive_root / group / name)
            reproduced = pd.read_csv(args.reproduced_root / group / name)
            assert_frame_equal(
                archived,
                reproduced,
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
            checked += 1
    print(f"reproduction verification passed: {checked} CSV files")


if __name__ == "__main__":
    main()

