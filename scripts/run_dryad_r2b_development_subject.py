#!/usr/bin/env python3
"""Safely stream and run one frozen Dryad R2b development participant."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    ROOT / "outputs/qc/dryad_raw_stream/R2_selection_frozen_v1/r2_selected_subjects.csv"
)
DEFAULT_URL_FILE = Path("/tmp/dryad_stim_signed_url.txt")
DEFAULT_SCRATCH = Path("/tmp/dryad_r2b_development")
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/qc/dryad_raw_stream/R2b_development_v1"
DEVELOPMENT_SUBJECTS = (2, 13, 23, 27, 37)
MINIMUM_SCRATCH_RESERVE_BYTES = 64 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development_subject(selection_path: Path, subject: int) -> dict[str, int]:
    selection = pd.read_csv(selection_path).sort_values("selection_order")
    observed = selection["subject"].astype(int).tolist()
    if observed != [2, 13, 27, 23, 37]:
        raise ValueError(f"selection file is not the frozen five-person set: {observed}")
    if subject not in DEVELOPMENT_SUBJECTS:
        raise ValueError(f"subject must be in development set {DEVELOPMENT_SUBJECTS}")
    row = selection.loc[selection["subject"].eq(subject)]
    if len(row) != 1:
        raise ValueError(f"expected exactly one row for S{subject}")
    return {
        "subject": int(subject),
        "selection_order": int(row.iloc[0]["selection_order"]),
        "expected_uncompressed_bytes": int(row.iloc[0]["uncompressed_bytes"]),
    }


def require_capacity(scratch_root: Path, expected_bytes: int) -> dict[str, int]:
    scratch_root.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(scratch_root).free)
    required = int(expected_bytes) + MINIMUM_SCRATCH_RESERVE_BYTES
    if free < required:
        raise OSError(f"insufficient scratch space: free={free}, required={required}")
    return {
        "scratch_free_bytes_before_extraction": free,
        "scratch_required_bytes": required,
    }


def run_command(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def run_subject(
    subject: int,
    selection_path: Path,
    url_file: Path,
    scratch_root: Path,
    output_root: Path,
) -> dict[str, object]:
    frozen = load_development_subject(selection_path, subject)
    if not url_file.is_file():
        raise FileNotFoundError(f"temporary signed URL file not found: {url_file}")
    capacity = require_capacity(scratch_root, frozen["expected_uncompressed_bytes"])
    output_dir = output_root / f"S{subject}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    bdf = scratch_root / f"S{subject}_Stim.bdf"
    if bdf.exists():
        raise FileExistsError(f"refusing to overwrite scratch BDF: {bdf}")

    member = f"Raw data stim/S{subject}_Stim.bdf"
    run_command(
        [
            sys.executable,
            "scripts/stream_remote_zip.py",
            "--url-file",
            str(url_file),
            "--extract-member",
            member,
            "--output",
            str(bdf),
        ]
    )
    observed_bytes = int(bdf.stat().st_size)
    if observed_bytes != frozen["expected_uncompressed_bytes"]:
        raise RuntimeError(
            f"S{subject}: size mismatch; expected={frozen['expected_uncompressed_bytes']}, "
            f"observed={observed_bytes}"
        )
    source_hash = sha256(bdf)
    run_command(
        [
            sys.executable,
            "scripts/develop_dryad_r2b_technical_subject.py",
            str(bdf),
            str(output_dir),
        ]
    )
    summary_path = output_dir / "technical_summary_r2b_development.json"
    with summary_path.open(encoding="utf-8") as handle:
        technical = json.load(handle)
    if technical["source_bdf_sha256"] != source_hash:
        raise RuntimeError(f"S{subject}: technical output hash mismatch")
    if int(technical["source_bdf_size_bytes"]) != observed_bytes:
        raise RuntimeError(f"S{subject}: technical output size mismatch")

    bdf.unlink()
    summary = {
        **frozen,
        **capacity,
        "source_bdf_sha256": source_hash,
        "source_bdf_size_bytes": observed_bytes,
        "temporary_bdf_deleted_after_verification": True,
        "technical_pass_development_candidate": bool(
            technical["technical_pass_development_candidate"]
        ),
        "result_bearing_fields_emitted": bool(
            technical["result_bearing_fields_emitted"]
        ),
        "status": "complete_development_only_not_confirmatory_freeze",
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--url-file", type=Path, default=DEFAULT_URL_FILE)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = run_subject(
            args.subject,
            args.selection,
            args.url_file,
            args.scratch_root,
            args.output_root,
        )
    except Exception as error:
        raise SystemExit(
            f"R2b development run stopped: {error}. Any extracted BDF was retained."
        ) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
