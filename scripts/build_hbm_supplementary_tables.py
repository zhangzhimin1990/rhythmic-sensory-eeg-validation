#!/usr/bin/env python3
"""Build focused HBM supplementary tables from frozen machine-readable outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tables"


def build_ds007648_family() -> pd.DataFrame:
    source = ROOT / "outputs/models/ds007648_behavioral_validity/clustered_models.csv"
    frame = pd.read_csv(source)
    keep = frame.loc[
        frame.specification.eq("main")
        & frame.feature_family.eq("phase_projected")
        & frame.q_value_within_outcome_four_tests.notna(),
        [
            "analysis",
            "term",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "q_value_within_outcome_four_tests",
            "scale",
            "n_trials",
            "n_subjects",
        ],
    ].copy()
    keep = keep.rename(
        columns={
            "analysis": "outcome",
            "ci_low": "ci95_low",
            "ci_high": "ci95_high",
            "q_value_within_outcome_four_tests": "holm_q_within_outcome_four_tests",
        }
    )
    expected = 8
    if len(keep) != expected:
        raise RuntimeError(f"expected {expected} ds007648 family rows, found {len(keep)}")
    return keep


def build_fsiq_sensitivity() -> pd.DataFrame:
    source = ROOT / "outputs/models/ds006780_behavioral_validity/fsiq_audit/robustness_models.csv"
    frame = pd.read_csv(source).copy()
    frame = frame.rename(columns={"ci_low": "ci95_low", "ci_high": "ci95_high"})
    expected = 9
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} FSIQ sensitivity rows, found {len(frame)}")
    return frame


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    s2 = build_ds007648_family()
    s3 = build_fsiq_sensitivity()
    s2.to_csv(OUTPUT / "155_HBM_Supplementary_Table_S2_ds007648_model_family.csv", index=False)
    s3.to_csv(OUTPUT / "156_HBM_Supplementary_Table_S3_ds006780_FSIQ_sensitivity.csv", index=False)
    print({"table_s2_rows": len(s2), "table_s3_rows": len(s3), "status": "complete"})


if __name__ == "__main__":
    main()
