#!/usr/bin/env python3
"""Run pending Dryad confirmatory subjects sequentially without viewing effects."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/models/dryad_confirmatory_subjects_v1"
DEFAULT_LEDGER = ROOT / "outputs/status/dryad_confirmatory_completion_v1"
DEFAULT_SCRATCH = Path("/tmp/dryad_confirmatory")
MINIMUM_FREE_GIB = 2.0


def completed_subjects(output_root: Path) -> set[int]:
    complete: set[int] = set()
    for run_path in output_root.glob("S*/confirmatory_run_summary.json"):
        derivation_path = run_path.with_name("confirmatory_subject_summary.json")
        if not derivation_path.is_file():
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        derivation = json.loads(derivation_path.read_text(encoding="utf-8"))
        if (
            run.get("status") == "confirmatory_subject_complete"
            and derivation.get("status") == "confirmatory_subject_derivation_complete"
            and run.get("source_bdf_sha256") == derivation.get("source_bdf_sha256")
            and run.get("temporary_bdf_deleted_after_verification") is True
        ):
            complete.add(int(run_path.parent.name.removeprefix("S")))
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ledger-output", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--minimum-free-gib", type=float, default=MINIMUM_FREE_GIB)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    locked = list(map(int, protocol["confirmatory_subjects"]))
    complete = completed_subjects(args.output_root)
    pending = [subject for subject in locked if subject not in complete]
    print(json.dumps({
        "status": "result_blind_batch_start",
        "complete_subjects": len(complete),
        "pending_subjects": pending,
    }, ensure_ascii=False), flush=True)

    for index, subject in enumerate(pending, start=1):
        free_gib = shutil.disk_usage(args.scratch_root).free / 2**30
        if free_gib < args.minimum_free_gib:
            raise SystemExit(
                f"batch stopped before S{subject}: free space {free_gib:.2f} GiB "
                f"is below {args.minimum_free_gib:.2f} GiB"
            )
        print(json.dumps({
            "status": "subject_start",
            "subject": subject,
            "batch_index": index,
            "batch_total": len(pending),
            "free_gib": round(free_gib, 2),
        }, ensure_ascii=False), flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.run_dryad_confirmatory_subject_amended_v4",
                str(subject),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/summarize_dryad_confirmatory_completion.py",
                "--output",
                str(args.ledger_output),
            ],
            cwd=ROOT,
            check=True,
        )
        remaining = len(pending) - index
        print(json.dumps({
            "status": "subject_complete",
            "subject": subject,
            "remaining_in_batch": remaining,
            "free_gib": round(shutil.disk_usage(args.scratch_root).free / 2**30, 2),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
