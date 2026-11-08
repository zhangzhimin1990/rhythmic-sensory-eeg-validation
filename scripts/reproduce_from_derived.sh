#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3}"
TMP_ROOT="$(mktemp -d "./.reproduce_tmp.XXXXXX")"
TMP_ROOT="$(cd "$TMP_ROOT" && pwd)"
trap 'rm -rf "$TMP_ROOT"' EXIT
mkdir -p \
  "$TMP_ROOT/estimand_alignment" \
  "$TMP_ROOT/hbm_prediction_audit_v1" \
  "$TMP_ROOT/hbm_reviewer_uncertainty_audit_v1" \
  "$TMP_ROOT/dryad_hbm_spatial_group_v1"

"$PYTHON_BIN" -m scripts.analyze_estimand_alignment \
  --output "$TMP_ROOT/estimand_alignment"
"$PYTHON_BIN" -m scripts.analyze_hbm_prediction_audit \
  --output-dir "$TMP_ROOT/hbm_prediction_audit_v1"
"$PYTHON_BIN" -m scripts.audit_hbm_reviewer_uncertainty \
  --prediction "$TMP_ROOT/hbm_prediction_audit_v1/participant_loss_increments.csv" \
  --output "$TMP_ROOT/hbm_reviewer_uncertainty_audit_v1"
"$PYTHON_BIN" -m scripts.analyze_dryad_hbm_spatial_group_v2 \
  --output-dir "$TMP_ROOT/dryad_hbm_spatial_group_v1"
"$PYTHON_BIN" -m scripts.build_hbm_supplementary_tables
"$PYTHON_BIN" -m scripts.verify_reproduced_outputs --reproduced-root "$TMP_ROOT"
"$PYTHON_BIN" -m pytest -q tests

