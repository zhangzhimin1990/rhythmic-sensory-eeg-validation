#!/usr/bin/env python3
"""Freeze the Dryad confirmatory analysis assets before signal re-access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from scripts.validate_dryad_confirmatory_protocol import validate
except ModuleNotFoundError:  # Direct execution via ``python scripts/...``.
    from validate_dryad_confirmatory_protocol import validate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/models/dryad_confirmatory_protocol_freeze_v1"
ASSETS = (
    "configs/dryad_confirmatory_protocol_v1.json",
    "86_Dryad确认性神经行为分析冻结方案.md",
    "scripts/validate_dryad_confirmatory_protocol.py",
    "scripts/derive_dryad_confirmatory_subject.py",
    "scripts/analyze_dryad_confirmatory_group.py",
    "scripts/run_dryad_confirmatory_subject.py",
    "scripts/stream_remote_zip.py",
    "scripts/freeze_dryad_confirmatory_protocol.py",
    "tests/test_dryad_confirmatory_protocol.py",
    "tests/test_derive_dryad_confirmatory_subject.py",
    "tests/test_analyze_dryad_confirmatory_group.py",
    "tests/test_run_dryad_confirmatory_subject.py",
    "tests/test_stream_remote_zip.py",
    "outputs/qc/dryad_raw_stream/R2b_validation_gate_v1/r2b_validation_technical_gate.csv",
    "outputs/qc/dryad_raw_stream/R2b_validation_gate_v1/r2b_validation_technical_gate.json",
    "outputs/qc/dryad_raw_stream/R2b_validation_summary_v1/validation_summary.json",
    "data/public/dryad_t76hdr8dm/v5/metadata/Dataset.xlsx",
    "outputs/qc/dryad_raw_stream/stim_zip_members.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty freeze directory: {output_dir}")
    validation = validate(
        ROOT / "configs/dryad_confirmatory_protocol_v1.json",
        ROOT / "outputs/qc/dryad_raw_stream/R2b_validation_gate_v1/r2b_validation_technical_gate.csv",
    )
    missing = [asset for asset in ASSETS if not (ROOT / asset).is_file()]
    if missing:
        raise FileNotFoundError(f"confirmatory freeze assets missing: {missing}")
    rows = [{
        "relative_path": asset,
        "size_bytes": (ROOT / asset).stat().st_size,
        "sha256": sha256(ROOT / asset),
    } for asset in ASSETS]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "confirmatory_freeze_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol_id": validation["protocol_id"],
        "status": "frozen_before_confirmatory_signal_reaccess",
        "n_frozen_assets": len(rows),
        "n_confirmatory_subjects": validation["n_confirmatory_subjects"],
        "primary_metric": validation["primary_metric"],
        "active_contrast": validation["active_contrast"],
        "manifest_sha256": sha256(manifest),
        "outcomes_accessed_during_freeze": False,
    }
    (output_dir / "confirmatory_freeze_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
