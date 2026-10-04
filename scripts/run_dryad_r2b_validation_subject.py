#!/usr/bin/env python3
"""Run one untouched Dryad R2b validation participant under the frozen protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMBERS = ROOT / "outputs/qc/dryad_raw_stream/stim_zip_members.csv"
DEFAULT_URL_FILE = Path("/tmp/dryad_stim_signed_url.txt")
DEFAULT_SCRATCH = Path("/tmp/dryad_r2b_validation")
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/qc/dryad_raw_stream/R2b_validation_technical_v1"
DEFAULT_FREEZE = ROOT / "outputs/qc/dryad_raw_stream/R2b_protocol_freeze_v1"
DEVELOPMENT_SUBJECTS = frozenset({2, 13, 23, 27, 37})
VALIDATION_SUBJECTS = frozenset(range(1, 45)) - DEVELOPMENT_SUBJECTS
MINIMUM_SCRATCH_RESERVE_BYTES = 64 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze(freeze_dir: Path, root: Path = ROOT) -> str:
    manifest = freeze_dir / "protocol_freeze_manifest.csv"
    summary_path = freeze_dir / "protocol_freeze_summary.json"
    if not manifest.is_file() or not summary_path.is_file():
        raise FileNotFoundError("R2b protocol freeze package is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "frozen_before_validation_data_access":
        raise RuntimeError("R2b protocol is not in frozen state")
    if summary.get("manifest_sha256") != sha256(manifest):
        raise RuntimeError("R2b freeze manifest hash mismatch")
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"frozen asset changed or missing: {row['relative_path']}")
    return summary["manifest_sha256"]


def load_inventory(members_path: Path, subject: int) -> dict[str, int]:
    if subject not in VALIDATION_SUBJECTS:
        raise ValueError("subject is not in the frozen 39-person validation set")
    members = pd.read_csv(members_path)
    extracted = members["member"].astype(str).str.extract(r"(?:^|/)S(\d+)_Stim\.bdf$")[0]
    table = members.loc[extracted.notna()].copy()
    table["subject"] = extracted.dropna().astype(int).to_numpy()
    if set(table["subject"]) != set(range(1, 45)) or table["subject"].duplicated().any():
        raise ValueError("stimulus ZIP inventory is not the exact 44-person set")
    row = table.loc[table["subject"].eq(subject)].iloc[0]
    return {"subject": subject, "expected_uncompressed_bytes": int(row["uncompressed_bytes"])}


def run_command(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def run_subject(
    subject: int,
    members_path: Path,
    url_file: Path,
    scratch_root: Path,
    output_root: Path,
    freeze_dir: Path,
) -> dict[str, object]:
    freeze_hash = verify_freeze(freeze_dir)
    frozen = load_inventory(members_path, subject)
    if not url_file.is_file():
        raise FileNotFoundError(f"temporary signed URL file not found: {url_file}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(scratch_root).free)
    required = frozen["expected_uncompressed_bytes"] + MINIMUM_SCRATCH_RESERVE_BYTES
    if free < required:
        raise OSError(f"insufficient scratch space: free={free}, required={required}")
    output_dir = output_root / f"S{subject}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    bdf = scratch_root / f"S{subject}_Stim.bdf"
    if bdf.exists():
        raise FileExistsError(f"refusing to overwrite scratch BDF: {bdf}")
    run_command(
        [sys.executable, "scripts/stream_remote_zip.py", "--url-file", str(url_file),
         "--extract-member", f"Raw data stim/S{subject}_Stim.bdf", "--output", str(bdf)]
    )
    if bdf.stat().st_size != frozen["expected_uncompressed_bytes"]:
        raise RuntimeError(f"S{subject}: extracted BDF size mismatch")
    source_hash = sha256(bdf)
    run_command(
        [sys.executable, "scripts/develop_dryad_r2b_technical_subject.py", str(bdf), str(output_dir)]
    )
    technical_path = output_dir / "technical_summary_r2b_development.json"
    technical = json.loads(technical_path.read_text(encoding="utf-8"))
    if technical["source_bdf_sha256"] != source_hash:
        raise RuntimeError(f"S{subject}: technical output hash mismatch")
    bdf.unlink()
    result = {
        **frozen,
        "source_bdf_sha256": source_hash,
        "protocol_freeze_manifest_sha256": freeze_hash,
        "technical_pass": bool(technical["technical_pass_development_candidate"]),
        "result_bearing_fields_emitted": False,
        "temporary_bdf_deleted_after_verification": True,
        "status": "complete_frozen_validation_technical_only",
    }
    (output_dir / "validation_run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int)
    parser.add_argument("--members", type=Path, default=DEFAULT_MEMBERS)
    parser.add_argument("--url-file", type=Path, default=DEFAULT_URL_FILE)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    try:
        observed = run_subject(
            args.subject, args.members, args.url_file, args.scratch_root,
            args.output_root, args.freeze_dir
        )
    except Exception as error:
        raise SystemExit(
            f"R2b validation run stopped: {error}. Any extracted BDF was retained."
        ) from error
    print(json.dumps(observed, ensure_ascii=False, indent=2))
