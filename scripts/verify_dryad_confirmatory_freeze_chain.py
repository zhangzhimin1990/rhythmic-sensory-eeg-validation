#!/usr/bin/env python3
"""Verify every pre-result Dryad confirmatory freeze and amendment manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/status/dryad_confirmatory_freeze_chain_v1"
CSV_MANIFESTS = (
    "outputs/models/dryad_confirmatory_protocol_freeze_v1/confirmatory_freeze_manifest.csv",
    "outputs/models/dryad_confirmatory_runner_amendment_freeze_v1/runner_amendment_freeze_manifest.csv",
    "outputs/models/dryad_confirmatory_transport_amendment_freeze_v1/transport_amendment_freeze_manifest.csv",
    "outputs/models/dryad_confirmatory_resumable_amendment_freeze_v1/resumable_amendment_freeze_manifest.csv",
    "outputs/models/dryad_incomplete_read_amendment_freeze_v1/incomplete_read_amendment_manifest.csv",
)
JSON_MANIFESTS = (
    "outputs/models/dryad_confirmatory_inference_audit_freeze_v1/inference_audit_freeze_manifest.json",
    "outputs/models/dryad_group_execution_gate_freeze_v1/group_execution_gate_freeze_manifest.json",
    "outputs/models/dryad_primary_branch_amendment_freeze_v1/primary_branch_amendment_freeze_manifest.json",
    "outputs/models/dryad_behaviour_branch_amendment_freeze_v1/behaviour_branch_amendment_freeze_manifest.json",
    "outputs/models/dryad_coupling_branch_amendment_freeze_v1/coupling_branch_amendment_freeze_manifest.json",
    "outputs/models/dryad_prediction_branch_amendment_freeze_v1/prediction_branch_amendment_freeze_manifest.json",
    "outputs/models/dryad_claim_release_freeze_v1/claim_release_freeze_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_csv_manifest(path: Path, root: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty freeze manifest: {path}")
    for row in rows:
        asset = root / row["relative_path"]
        if not asset.is_file():
            raise RuntimeError(f"missing frozen asset: {row['relative_path']}")
        if asset.stat().st_size != int(row["size_bytes"]) or sha256(asset) != row["sha256"]:
            raise RuntimeError(f"changed frozen asset: {row['relative_path']}")
    return {"manifest": str(path.relative_to(root)), "format": "csv", "assets_verified": len(rows)}


def verify_json_manifest(path: Path, root: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if not files:
        raise RuntimeError(f"empty freeze manifest: {path}")
    for row in files:
        asset = root / row["path"]
        if not asset.is_file():
            raise RuntimeError(f"missing frozen asset: {row['path']}")
        if asset.stat().st_size != int(row["bytes"]) or sha256(asset) != row["sha256"]:
            raise RuntimeError(f"changed frozen asset: {row['path']}")
    canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    canonical_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if canonical_hash != manifest.get("manifest_sha256"):
        raise RuntimeError(f"manifest file-list hash mismatch: {path}")
    return {
        "manifest": str(path.relative_to(root)),
        "format": "json",
        "freeze_id": manifest.get("freeze_id"),
        "assets_verified": len(files),
        "manifest_sha256": canonical_hash,
    }


def verify(root: Path = ROOT) -> dict[str, object]:
    results = []
    for relative in CSV_MANIFESTS:
        results.append(verify_csv_manifest(root / relative, root))
    for relative in JSON_MANIFESTS:
        results.append(verify_json_manifest(root / relative, root))
    return {
        "status": "dryad_confirmatory_freeze_chain_intact",
        "manifests_verified": len(results),
        "assets_verified_with_repetition": sum(item["assets_verified"] for item in results),
        "contains_condition_effects": False,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "freeze_chain_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
