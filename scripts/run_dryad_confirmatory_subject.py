#!/usr/bin/env python3
"""Stream, verify, derive, and delete one locked Dryad confirmatory BDF."""

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
DEFAULT_SCRATCH = Path("/tmp/dryad_confirmatory")
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/models/dryad_confirmatory_subjects_v1"
DEFAULT_FREEZE = ROOT / "outputs/models/dryad_confirmatory_protocol_freeze_v1"
DEFAULT_PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
DEFAULT_METADATA = ROOT / "data/public/dryad_t76hdr8dm/v5/metadata/Dataset.xlsx"
DEFAULT_TECHNICAL = ROOT / "outputs/qc/dryad_raw_stream/R2b_validation_technical_v1"
MINIMUM_SCRATCH_RESERVE_BYTES = 64 * 1024 * 1024
MINIMUM_FREE_AFTER_EXTRACTION_GIB = 1.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze(freeze_dir: Path, root: Path = ROOT) -> str:
    manifest = freeze_dir / "confirmatory_freeze_manifest.csv"
    summary_path = freeze_dir / "confirmatory_freeze_summary.json"
    if not manifest.is_file() or not summary_path.is_file():
        raise FileNotFoundError("confirmatory freeze package is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "frozen_before_confirmatory_signal_reaccess":
        raise RuntimeError("confirmatory protocol is not frozen")
    if summary.get("manifest_sha256") != sha256(manifest):
        raise RuntimeError("confirmatory freeze manifest hash mismatch")
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        asset = root / row["relative_path"]
        if not asset.is_file() or sha256(asset) != row["sha256"]:
            raise RuntimeError(f"frozen confirmatory asset changed: {row['relative_path']}")
    return str(summary["manifest_sha256"])


def load_subject(
    subject: int, members_path: Path, protocol_path: Path, metadata_path: Path
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if subject not in set(map(int, protocol["confirmatory_subjects"])):
        raise ValueError("subject is not in the locked confirmatory set")
    members = pd.read_csv(members_path)
    extracted = members["member"].astype(str).str.extract(r"(?:^|/)S(\d+)_Stim\.bdf$")[0]
    inventory = members.loc[extracted.notna()].copy()
    inventory["subject"] = extracted.dropna().astype(int).to_numpy()
    if set(inventory["subject"]) != set(range(1, 45)) or inventory["subject"].duplicated().any():
        raise ValueError("stimulus ZIP inventory is not the exact 44-person set")
    source = inventory.loc[inventory["subject"].eq(subject)].iloc[0]
    metadata = pd.read_excel(metadata_path, sheet_name="Dataset")
    participant = metadata.loc[metadata["subject"].eq(subject)]
    if len(participant) != 1:
        raise ValueError("metadata must contain exactly one participant row")
    theta = float(participant.iloc[0]["individual_freq"])
    if not 3.0 <= theta <= 8.0:
        raise ValueError("individual theta frequency is outside the frozen plausible range")
    return {
        "subject": subject,
        "expected_uncompressed_bytes": int(source["uncompressed_bytes"]),
        "individual_theta_hz": theta,
    }


def run_command(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def run_subject(
    subject: int,
    members_path: Path,
    url_file: Path,
    scratch_root: Path,
    output_root: Path,
    freeze_dir: Path,
    protocol_path: Path,
    metadata_path: Path,
    technical_root: Path,
) -> dict[str, object]:
    freeze_hash = verify_freeze(freeze_dir)
    locked = load_subject(subject, members_path, protocol_path, metadata_path)
    if not url_file.is_file():
        raise FileNotFoundError(f"temporary signed URL file not found: {url_file}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(scratch_root).free)
    required = int(locked["expected_uncompressed_bytes"]) + MINIMUM_SCRATCH_RESERVE_BYTES
    if free < required:
        raise OSError(f"insufficient scratch space: free={free}, required={required}")
    output_dir = output_root / f"S{subject}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    bdf = scratch_root / f"S{subject}_Stim.bdf"
    if bdf.exists():
        raise FileExistsError(f"refusing to overwrite scratch BDF: {bdf}")
    run_command([
        sys.executable, "scripts/stream_remote_zip.py", "--url-file", str(url_file),
        "--extract-member", f"Raw data stim/S{subject}_Stim.bdf", "--output", str(bdf),
        "--cache-mib", "1", "--timeout", "30",
        "--minimum-free-gib-after", str(MINIMUM_FREE_AFTER_EXTRACTION_GIB),
    ])
    if bdf.stat().st_size != locked["expected_uncompressed_bytes"]:
        raise RuntimeError("extracted BDF size mismatch")
    source_hash = sha256(bdf)
    technical = json.loads(
        (technical_root / f"S{subject}/technical_summary_r2b_development.json").read_text(encoding="utf-8")
    )
    if source_hash != technical["source_bdf_sha256"]:
        raise RuntimeError("BDF hash differs from frozen technical-validation input")
    run_command([
        sys.executable, "scripts/derive_dryad_confirmatory_subject.py",
        str(bdf), str(subject), str(output_dir),
        "--individual-theta-hz", str(locked["individual_theta_hz"]),
        "--technical-root", str(technical_root), "--protocol", str(protocol_path),
    ])
    bdf.unlink()
    result = {
        **locked,
        "source_bdf_sha256": source_hash,
        "confirmatory_freeze_manifest_sha256": freeze_hash,
        "temporary_bdf_deleted_after_verification": True,
        "status": "confirmatory_subject_complete",
    }
    (output_dir / "confirmatory_run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int)
    parser.add_argument("--members", type=Path, default=DEFAULT_MEMBERS)
    parser.add_argument("--url-file", type=Path, default=DEFAULT_URL_FILE)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--freeze-dir", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--technical-root", type=Path, default=DEFAULT_TECHNICAL)
    args = parser.parse_args()
    try:
        result = run_subject(
            args.subject, args.members, args.url_file, args.scratch_root,
            args.output_root, args.freeze_dir, args.protocol, args.metadata,
            args.technical_root,
        )
    except Exception as error:
        raise SystemExit(
            f"confirmatory run stopped: {error}. Any extracted BDF was retained."
        ) from error
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
