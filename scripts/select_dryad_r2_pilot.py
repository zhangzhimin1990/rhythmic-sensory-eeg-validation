#!/usr/bin/env python3
"""Freeze a result-blind five-participant Dryad R2 signal-QC pilot.

The selector uses only two pretreatment variables from the released workbook
(`individual_freq` and `bl_rt`) plus the uncompressed BDF member size from the
remote ZIP central directory. Post-stimulation EEG and behavioural outcomes are
deliberately unavailable to the selection function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data/public/dryad_t76hdr8dm/v5/metadata/Dataset.xlsx"
DEFAULT_MEMBERS = ROOT / "outputs/qc/dryad_raw_stream/stim_zip_members.csv"
DEFAULT_OUTPUT = ROOT / "outputs/qc/dryad_raw_stream/R2_selection_frozen_v1"
SELECTION_VARIABLES = ("individual_freq", "bl_rt", "uncompressed_bytes")
EXPECTED_SUBJECTS = tuple(range(1, 45))
SEED_SUBJECT = 2
N_SELECTED = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empirical_rank_01(values: pd.Series) -> pd.Series:
    """Map values to average empirical ranks on [0, 1], preserving ties."""
    if len(values) < 2:
        raise ValueError("at least two observations are required for rank scaling")
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"non-finite values in {values.name}")
    return (numeric.rank(method="average") - 1.0) / (len(numeric) - 1.0)


def validate_profile(profile: pd.DataFrame) -> None:
    required = {"subject", *SELECTION_VARIABLES}
    missing = required - set(profile.columns)
    if missing:
        raise ValueError(f"selection profile missing columns: {sorted(missing)}")
    if profile["subject"].duplicated().any():
        duplicated = profile.loc[profile["subject"].duplicated(False), "subject"].tolist()
        raise ValueError(f"duplicate subjects in selection profile: {duplicated}")
    observed = tuple(sorted(profile["subject"].astype(int)))
    if observed != EXPECTED_SUBJECTS:
        absent = sorted(set(EXPECTED_SUBJECTS) - set(observed))
        unexpected = sorted(set(observed) - set(EXPECTED_SUBJECTS))
        raise ValueError(f"expected subjects 1-44; missing={absent}, unexpected={unexpected}")
    numeric = profile[list(SELECTION_VARIABLES)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"missing/non-finite selector values: {bad}")
    if (numeric["individual_freq"] <= 0).any():
        raise ValueError("individual_freq must be positive")
    if (numeric["bl_rt"] <= 0).any():
        raise ValueError("bl_rt must be positive")
    if (numeric["uncompressed_bytes"] <= 0).any():
        raise ValueError("uncompressed_bytes must be positive")


def build_profile(metadata_path: Path, members_path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(
        metadata_path,
        usecols=["subject", "individual_freq", "bl_rt"],
    )
    members = pd.read_csv(
        members_path,
        usecols=["member", "is_directory", "uncompressed_bytes"],
    )
    members = members.loc[~members["is_directory"].astype(bool)].copy()
    extracted = members["member"].str.extract(r"(?:^|/)S(\d+)_Stim\.bdf$")[0]
    if extracted.isna().any():
        bad = members.loc[extracted.isna(), "member"].tolist()
        raise ValueError(f"unexpected non-directory members: {bad}")
    members["subject"] = extracted.astype(int)
    profile = metadata.merge(
        members[["subject", "uncompressed_bytes"]],
        on="subject",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not profile["_merge"].eq("both").all():
        mismatch = profile.loc[~profile["_merge"].eq("both"), ["subject", "_merge"]]
        raise ValueError(f"metadata/member mismatch:\n{mismatch.to_string(index=False)}")
    profile = profile.drop(columns="_merge").sort_values("subject").reset_index(drop=True)
    validate_profile(profile)
    return profile


def greedy_maximin_selection(
    profile: pd.DataFrame,
    seed_subject: int = SEED_SUBJECT,
    n_selected: int = N_SELECTED,
) -> tuple[pd.DataFrame, list[int]]:
    """Select a deterministic, dispersed pilot in empirical-rank space."""
    validate_profile(profile)
    if seed_subject not in set(profile["subject"]):
        raise ValueError(f"seed subject {seed_subject} is absent")
    if not 1 <= n_selected <= len(profile):
        raise ValueError("n_selected must be between 1 and the number of subjects")

    ranked = profile.copy()
    rank_columns = []
    for column in SELECTION_VARIABLES:
        rank_column = f"{column}_rank01"
        ranked[rank_column] = empirical_rank_01(ranked[column])
        rank_columns.append(rank_column)

    coordinates = ranked[rank_columns].to_numpy(dtype=float)
    selected = [int(seed_subject)]
    selection_scores = {int(seed_subject): np.nan}
    while len(selected) < n_selected:
        selected_indices = ranked.index[ranked["subject"].isin(selected)].to_numpy()
        distances = np.linalg.norm(
            coordinates[:, np.newaxis, :] - coordinates[selected_indices][np.newaxis, :, :],
            axis=2,
        )
        nearest = distances.min(axis=1)
        candidates = ranked.loc[~ranked["subject"].isin(selected), ["subject"]].copy()
        candidates["nearest_distance"] = nearest[candidates.index]
        chosen = candidates.sort_values(
            ["nearest_distance", "subject"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        subject = int(chosen["subject"])
        selected.append(subject)
        selection_scores[subject] = float(chosen["nearest_distance"])

    ranked["selected_r2"] = ranked["subject"].isin(selected)
    ranked["selection_order"] = ranked["subject"].map(
        {subject: order for order, subject in enumerate(selected, start=1)}
    ).astype("Int64")
    ranked["min_rank_distance_at_selection"] = ranked["subject"].map(selection_scores)
    ranked["selection_role"] = np.where(
        ranked["subject"].eq(seed_subject),
        "fixed_R1_seed",
        np.where(ranked["selected_r2"], "result_blind_maximin_addition", "not_selected"),
    )
    return ranked, selected


def marginal_coverage(profile: pd.DataFrame, selected: list[int]) -> dict[str, dict[str, float]]:
    chosen = profile.loc[profile["subject"].isin(selected)]
    coverage: dict[str, dict[str, float]] = {}
    for column in SELECTION_VARIABLES:
        full_min = float(profile[column].min())
        full_max = float(profile[column].max())
        chosen_min = float(chosen[column].min())
        chosen_max = float(chosen[column].max())
        denominator = full_max - full_min
        coverage[column] = {
            "full_min": full_min,
            "full_median": float(profile[column].median()),
            "full_max": full_max,
            "selected_min": chosen_min,
            "selected_median": float(chosen[column].median()),
            "selected_max": chosen_max,
            "selected_span_fraction_of_full_range": (
                float((chosen_max - chosen_min) / denominator) if denominator > 0 else 1.0
            ),
        }
    return coverage


def run(metadata_path: Path, members_path: Path, output_dir: Path) -> dict[str, object]:
    profile = build_profile(metadata_path, members_path)
    candidate_profile, selected = greedy_maximin_selection(profile)
    selected_table = candidate_profile.loc[candidate_profile["selected_r2"]].sort_values(
        "selection_order"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_profile.to_csv(output_dir / "candidate_profile.csv", index=False)
    selected_table.to_csv(output_dir / "r2_selected_subjects.csv", index=False)
    summary: dict[str, object] = {
        "algorithm_version": "result_blind_empirical_rank_greedy_maximin_v1",
        "metadata_sha256": sha256(metadata_path),
        "members_csv_sha256": sha256(members_path),
        "n_candidates": int(len(profile)),
        "seed_subject": SEED_SUBJECT,
        "n_selected": N_SELECTED,
        "selected_subjects_in_order": selected,
        "selection_variables_only": list(SELECTION_VARIABLES),
        "post_stimulation_outcomes_used": False,
        "tie_breaker": "lowest_subject_number",
        "marginal_coverage": marginal_coverage(profile, selected),
    }
    with (output_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--members", type=Path, default=DEFAULT_MEMBERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.metadata, args.members, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
