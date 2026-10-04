#!/usr/bin/env python3
"""Build a result-blind completion ledger for Dryad confirmatory subjects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
DEFAULT_SUBJECT_ROOT = ROOT / "outputs/models/dryad_confirmatory_subjects_v1"
DEFAULT_OUTPUT = ROOT / "outputs/status/dryad_confirmatory_completion_v1"


def build_ledger(protocol_path: Path, subject_root: Path) -> pd.DataFrame:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = []
    for subject in map(int, protocol["confirmatory_subjects"]):
        run_path = subject_root / f"S{subject}/confirmatory_run_summary.json"
        derivation_path = subject_root / f"S{subject}/confirmatory_subject_summary.json"
        row = {"subject": subject, "status": "pending"}
        if run_path.is_file() and derivation_path.is_file():
            run = json.loads(run_path.read_text(encoding="utf-8"))
            derivation = json.loads(derivation_path.read_text(encoding="utf-8"))
            if run.get("status") != "confirmatory_subject_complete":
                raise ValueError(f"S{subject} run summary is not complete")
            if derivation.get("status") != "confirmatory_subject_derivation_complete":
                raise ValueError(f"S{subject} derivation summary is not complete")
            if run["source_bdf_sha256"] != derivation["source_bdf_sha256"]:
                raise ValueError(f"S{subject} source hashes disagree")
            if not run.get("temporary_bdf_deleted_after_verification"):
                raise ValueError(f"S{subject} temporary BDF deletion is not verified")
            row.update({
                "status": "complete",
                "individual_theta_hz": run["individual_theta_hz"],
                "analysis_frequency_hz": derivation["analysis_frequency_hz"],
                "technical_usable_epochs_reproduced": derivation["n_technical_usable_epochs_reproduced"],
                "good_frozen_qc_channels": derivation["n_good_frozen_qc_channels"],
                "frontocentral_roi_channels": len(derivation["frontocentral_roi"]),
                "source_bdf_sha256": run["source_bdf_sha256"],
                "temporary_bdf_deleted": True,
                "amendment_ids": "|".join(run.get("post_freeze_amendment_ids", [])),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--subject-root", type=Path, default=DEFAULT_SUBJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledger = build_ledger(args.protocol, args.subject_root)
    args.output.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.output / "subject_completion_ledger.csv", index=False)
    complete = int(ledger["status"].eq("complete").sum())
    summary = {
        "status": "result_blind_completion_ledger",
        "confirmatory_subjects": int(len(ledger)),
        "complete_subjects": complete,
        "pending_subjects": int(len(ledger) - complete),
        "group_analysis_unlocked": complete == len(ledger),
        "contains_condition_effects": False,
    }
    (args.output / "completion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
