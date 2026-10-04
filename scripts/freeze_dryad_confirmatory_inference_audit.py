#!/usr/bin/env python3
"""Freeze the result-blind Dryad confirmatory inferential completion extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_freeze_v1"
ASSETS = (
    "configs/dryad_confirmatory_protocol_v1.json",
    "86_Dryad确认性神经行为分析冻结方案.md",
    "87_Dryad确认性冻结与结果前偏离审计.md",
    "88_Dryad确认性结果判定与推断完整性锁定.md",
    "scripts/analyze_dryad_confirmatory_group.py",
    "scripts/audit_dryad_confirmatory_group_inference.py",
    "scripts/freeze_dryad_confirmatory_inference_audit.py",
    "tests/test_analyze_dryad_confirmatory_group.py",
    "tests/test_audit_dryad_confirmatory_group_inference.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path = ROOT) -> dict[str, object]:
    files = []
    for relative in ASSETS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "freeze_id": "dryad_confirmatory_inference_audit_v1",
        "status": "frozen_before_completed_cohort_or_group_effect_access",
        "frozen_on": date.today().isoformat(),
        "n_assets": len(files),
        "participant_level_derivation_changed": False,
        "primary_estimator_changed": False,
        "files": files,
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty freeze directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    (output_dir / "inference_audit_freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        key: manifest[key]
        for key in (
            "freeze_id", "status", "frozen_on", "n_assets",
            "participant_level_derivation_changed", "primary_estimator_changed",
            "manifest_sha256",
        )
    }
    (output_dir / "inference_audit_freeze_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
