#!/usr/bin/env python3
"""Execute the locked group, inference-audit, and claim-release stages once."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.assemble_dryad_confirmatory_claim_release import run as release_claims
from scripts.audit_dryad_confirmatory_group_inference import run as audit_inference
from scripts.execute_dryad_confirmatory_group_locked import (
    DEFAULT_INFERENCE_MANIFEST,
    DEFAULT_METADATA,
    DEFAULT_PROTOCOL,
    DEFAULT_PROTOCOL_MANIFEST,
    DEFAULT_SUBJECT_ROOT,
    execute as execute_group,
)
from scripts.render_dryad_confirmatory_manuscript_insert import main as render_manuscript_insert


ROOT = Path(__file__).resolve().parents[1]
GROUP_ROOT = ROOT / "outputs/models/dryad_confirmatory_group_v1"
AUDIT_ROOT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_v1"
CLAIM_ROOT = ROOT / "outputs/status/dryad_confirmatory_claim_release_v1"


def main() -> None:
    group = execute_group(
        DEFAULT_PROTOCOL,
        DEFAULT_SUBJECT_ROOT,
        DEFAULT_METADATA,
        GROUP_ROOT,
        DEFAULT_PROTOCOL_MANIFEST,
        DEFAULT_INFERENCE_MANIFEST,
    )
    print(json.dumps({"stage": "group", **group}, ensure_ascii=False), flush=True)

    audit = audit_inference(
        DEFAULT_SUBJECT_ROOT,
        GROUP_ROOT,
        DEFAULT_PROTOCOL,
        AUDIT_ROOT,
    )
    print(json.dumps({"stage": "inference_audit", **audit}, ensure_ascii=False), flush=True)

    claims = release_claims(GROUP_ROOT, AUDIT_ROOT, CLAIM_ROOT)
    print(json.dumps({"stage": "claim_release", **claims}, ensure_ascii=False), flush=True)
    render_manuscript_insert()


if __name__ == "__main__":
    main()
