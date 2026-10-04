#!/usr/bin/env python3
"""Execute the frozen Dryad group analysis only after machine-verified 35/35 completion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

try:
    from scripts.analyze_dryad_confirmatory_group import run as run_group
    from scripts.summarize_dryad_confirmatory_completion import build_ledger
except ModuleNotFoundError:  # Direct execution via ``python scripts/...``.
    from analyze_dryad_confirmatory_group import run as run_group
    from summarize_dryad_confirmatory_completion import build_ledger


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
DEFAULT_SUBJECT_ROOT = ROOT / "outputs/models/dryad_confirmatory_subjects_v1"
DEFAULT_METADATA = ROOT / "data/public/dryad_t76hdr8dm/v5/metadata/Dataset.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs/models/dryad_confirmatory_group_v1"
DEFAULT_PROTOCOL_MANIFEST = (
    ROOT / "outputs/models/dryad_confirmatory_protocol_freeze_v1/confirmatory_freeze_manifest.csv"
)
DEFAULT_INFERENCE_MANIFEST = (
    ROOT / "outputs/models/dryad_confirmatory_inference_audit_freeze_v1/inference_audit_freeze_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_csv_freeze_manifest(manifest_path: Path, root: Path = ROOT) -> int:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("protocol freeze manifest is empty")
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file():
            raise RuntimeError(f"frozen asset missing: {row['relative_path']}")
        if path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"frozen asset changed: {row['relative_path']}")
    return len(rows)


def verify_json_freeze_manifest(manifest_path: Path, root: Path = ROOT) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if not files:
        raise RuntimeError("inference freeze manifest is empty")
    for row in files:
        path = root / row["path"]
        if not path.is_file():
            raise RuntimeError(f"frozen inference asset missing: {row['path']}")
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"frozen inference asset changed: {row['path']}")
    return len(files)


def require_complete_cohort(protocol_path: Path, subject_root: Path) -> dict[str, object]:
    ledger = build_ledger(protocol_path, subject_root)
    complete = int(ledger["status"].eq("complete").sum())
    total = int(len(ledger))
    if complete != total:
        pending = ledger.loc[ledger["status"].ne("complete"), "subject"].astype(int).tolist()
        raise RuntimeError(
            f"group analysis remains locked: {complete}/{total} complete; pending={pending}"
        )
    return {
        "status": "group_analysis_unlocked",
        "complete_subjects": complete,
        "confirmatory_subjects": total,
        "contains_condition_effects": False,
    }


def execute(
    protocol_path: Path,
    subject_root: Path,
    metadata_path: Path,
    output_dir: Path,
    protocol_manifest: Path,
    inference_manifest: Path,
) -> dict[str, object]:
    verify_csv_freeze_manifest(protocol_manifest)
    verify_json_freeze_manifest(inference_manifest)
    require_complete_cohort(protocol_path, subject_root)
    return run_group(subject_root, metadata_path, protocol_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--subject-root", type=Path, default=DEFAULT_SUBJECT_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST)
    parser.add_argument("--inference-manifest", type=Path, default=DEFAULT_INFERENCE_MANIFEST)
    args = parser.parse_args()
    result = execute(
        args.protocol,
        args.subject_root,
        args.metadata,
        args.output_dir,
        args.protocol_manifest,
        args.inference_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
