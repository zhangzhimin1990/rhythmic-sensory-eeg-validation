#!/usr/bin/env python3
"""Evaluate the result-blind 39-participant Dryad R2b technical gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

try:
    from scripts.develop_dryad_r2b_spatial_qc import require_technical_only_schema
except ModuleNotFoundError:  # Direct execution via ``python scripts/...``.
    from develop_dryad_r2b_spatial_qc import require_technical_only_schema


ALL_SUBJECTS = frozenset(range(1, 45))
DEVELOPMENT_SUBJECTS = frozenset({2, 13, 23, 27, 37})
VALIDATION_SUBJECTS = ALL_SUBJECTS - DEVELOPMENT_SUBJECTS
REQUIRED_TECHNICAL_PASSES = 32
SUMMARY_NAME = "technical_summary_r2b_development.json"
PARTICIPANT_OUTPUT_COLUMNS = [
    "subject",
    "n_scalp_channels",
    "n_bad_channels_development",
    "n_good_channels_development",
    "n_good_frontocentral_roi_channels",
    "n_recorded_target_events",
    "n_eeg_usable_event_sequences",
    "expected_event_coverage_fraction",
    "n_technically_usable_epochs",
    "overall_epoch_usable_fraction",
    "minimum_event_cell_usable_fraction",
    "n_event_cells_observed",
    "minimum_stimulation_to_target_ms",
    "minimum_analysis_window_pre_target_margin_ms",
    "channel_count_and_quality_gate",
    "eeg_event_coverage_gate",
    "four_event_cells_observed_gate",
    "overall_epoch_usability_gate",
    "minimum_event_cell_usability_gate",
    "analysis_window_margin_gate",
    "technical_pass_development_candidate",
    "source_bdf_size_bytes",
    "source_bdf_sha256",
]


def subject_from_name(name: str) -> int:
    match = re.fullmatch(r"S(\d+)_Stim\.bdf", name)
    if match is None:
        raise ValueError(f"unexpected source BDF name: {name}")
    return int(match.group(1))


def load_validation_summaries(input_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(input_root.glob(f"S*/{SUMMARY_NAME}")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        require_technical_only_schema(pd.DataFrame(), summary)
        # ``subject`` is intentionally derived from the source BDF name and
        # checked against the containing directory below; it is not emitted by
        # the subject-level technical JSON.
        missing = [
            column
            for column in PARTICIPANT_OUTPUT_COLUMNS
            if column != "subject" and column not in summary
        ]
        if missing:
            raise ValueError(f"{path}: missing technical fields: {missing}")
        subject = subject_from_name(str(summary.get("source_bdf_name", "")))
        directory_match = re.fullmatch(r"S(\d+)", path.parent.name)
        if directory_match is None or int(directory_match.group(1)) != subject:
            raise ValueError(f"{path}: directory and source subject do not match")
        rows.append({"subject": subject, **summary})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no R2b technical summaries found")
    if table["subject"].duplicated().any():
        duplicates = table.loc[table["subject"].duplicated(False), "subject"].tolist()
        raise ValueError(f"duplicate validation subjects: {duplicates}")
    observed = set(table["subject"].astype(int))
    if observed != VALIDATION_SUBJECTS:
        raise ValueError(
            f"R2b requires the exact 39-person validation set; "
            f"missing={sorted(VALIDATION_SUBJECTS-observed)}, "
            f"extra={sorted(observed-VALIDATION_SUBJECTS)}"
        )
    output = table[PARTICIPANT_OUTPUT_COLUMNS].sort_values("subject").reset_index(drop=True)
    require_technical_only_schema(output, {})
    return output


def evaluate_group(participants: pd.DataFrame) -> dict[str, object]:
    if set(participants["subject"].astype(int)) != VALIDATION_SUBJECTS:
        raise ValueError("participant table is not the exact frozen validation set")
    pass_count = int(participants["technical_pass_development_candidate"].astype(bool).sum())
    return {
        "n_validation_participants": int(len(participants)),
        "n_technical_pass": pass_count,
        "required_technical_pass": REQUIRED_TECHNICAL_PASSES,
        "technical_go": pass_count >= REQUIRED_TECHNICAL_PASSES,
        "development_subjects_excluded": sorted(DEVELOPMENT_SUBJECTS),
        "validation_subjects": sorted(VALIDATION_SUBJECTS),
        "result_bearing_fields_emitted": False,
        "decision_scope": "technical_gate_only",
        "status": "frozen_protocol_technical_gate_complete",
    }


def run(input_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    participants = load_validation_summaries(input_root)
    group = evaluate_group(participants)
    require_technical_only_schema(participants, group)
    output_dir.mkdir(parents=True, exist_ok=True)
    participants.to_csv(output_dir / "r2b_validation_technical_gate.csv", index=False)
    (output_dir / "r2b_validation_technical_gate.json").write_text(
        json.dumps(group, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.input_root, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
