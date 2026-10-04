#!/usr/bin/env python3
"""Create the immutable-hash Dryad R2b technical protocol freeze package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/qc/dryad_raw_stream/R2b_protocol_freeze_v1"
AUDIT_SUMMARY = (
    ROOT
    / "outputs/qc/dryad_raw_stream/R2b_development_audit_v1/development_audit_summary.json"
)
FROZEN_ASSETS = (
    "configs/dryad_r2b_protocol_frozen_v1.json",
    "configs/dryad_biosemi64_adjacency_v1.csv",
    "configs/dryad_biosemi64_adjacency_v1.json",
    "scripts/develop_dryad_r2b_spatial_qc.py",
    "scripts/develop_dryad_r2b_technical_subject.py",
    "scripts/run_dryad_r2b_validation_subject.py",
    "scripts/evaluate_dryad_r2b_technical_gate.py",
    "scripts/stream_remote_zip.py",
    "scripts/summarize_dryad_r2b_development.py",
    "tests/test_develop_dryad_r2b_spatial_qc.py",
    "tests/test_develop_dryad_r2b_technical_subject.py",
    "tests/test_run_dryad_r2b_validation_subject.py",
    "tests/test_evaluate_dryad_r2b_technical_gate.py",
    "tests/test_summarize_dryad_r2b_development.py",
    "outputs/qc/dryad_raw_stream/R2b_development_audit_v1/development_audit_summary.json",
    "outputs/qc/dryad_raw_stream/R2b_development_audit_v1/participant_technical_summary.csv",
    "outputs/qc/dryad_raw_stream/R2b_development_audit_v1/flagged_channel_details.csv",
    "outputs/qc/dryad_raw_stream/R2b_development_audit_v1/technical_flag_reason_counts.csv",
    "outputs/qc/dryad_raw_stream/R2b_development_audit_v1/threshold_sensitivity.csv"
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
    missing = [relative for relative in FROZEN_ASSETS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"freeze assets missing: {missing}")
    development = json.loads(AUDIT_SUMMARY.read_text(encoding="utf-8"))
    if development["n_technical_pass_candidate"] != 5:
        raise RuntimeError("five-person development technical gate is incomplete")
    if not development["all_threshold_grid_channel_count_gates_pass"]:
        raise RuntimeError("development threshold sensitivity gate failed")
    protocol = json.loads(
        (ROOT / "configs/dryad_r2b_protocol_frozen_v1.json").read_text(encoding="utf-8")
    )
    if len(protocol["validation_subjects"]) != 39:
        raise RuntimeError("frozen validation set must contain exactly 39 subjects")

    rows = [
        {
            "relative_path": relative,
            "size_bytes": (ROOT / relative).stat().st_size,
            "sha256": sha256(ROOT / relative),
        }
        for relative in FROZEN_ASSETS
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "protocol_freeze_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol_id": protocol["protocol_id"],
        "status": "frozen_before_validation_data_access",
        "n_frozen_assets": len(rows),
        "n_development_subjects": 5,
        "n_development_technical_pass": 5,
        "n_validation_subjects": 39,
        "required_validation_technical_passes": 32,
        "threshold_sensitivity_grid_all_pass": True,
        "manifest_sha256": sha256(manifest),
        "result_bearing_fields_emitted": False,
    }
    (output_dir / "protocol_freeze_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), ensure_ascii=False, indent=2))
