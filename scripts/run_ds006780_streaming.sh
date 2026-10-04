#!/usr/bin/env bash
set -euo pipefail

metadata_root="${1:-data/public/ds006780/v1.0.0}"
qc_root="${2:-data/public/behavioral_validation_qc/ds006780}"
output_root="${3:-outputs/derived/ds006780_assr_features}"

mkdir -p "$output_root"
stream_tmp="$(mktemp -d /tmp/ds006780_stream.XXXXXX)"
exclusion_log="$output_root/run_exclusions.tsv"
cleanup() {
  rm -rf -- "$stream_tmp"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

printf 'subject\trun\treason\n' > "$exclusion_log"

download_parallel() {
  local url="$1"
  local destination="$2"
  local expected_bytes="$3"
  local workers=12
  local chunk_size=$(( (expected_bytes + workers - 1) / workers ))
  local parts=()
  local pids=()
  local index start end part
  for (( index=0; index<workers; index++ )); do
    start=$(( index * chunk_size ))
    end=$(( start + chunk_size - 1 ))
    (( end >= expected_bytes )) && end=$(( expected_bytes - 1 ))
    part="${destination}.part${index}"
    parts+=("$part")
    curl -fsSL --retry 8 --retry-all-errors --retry-delay 2 \
      --speed-time 20 --speed-limit 65536 \
      -r "${start}-${end}" "$url" -o "$part" &
    pids+=("$!")
  done
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}"
  done
  .venv/bin/python scripts/join_binary_parts.py \
    "$destination" "$expected_bytes" "${parts[@]}"
}

while IFS= read -r annex_bdf; do
  stem="$(basename "$annex_bdf" _eeg.bdf)"
  subject="${stem%%_*}"
  run="${stem##*run-}"
  output_csv="$output_root/${stem}_features.csv"
  if [[ -s "$output_csv" ]] && grep -q ',3.0.0,' "$output_csv"; then
    continue
  fi

  metadata_eeg_dir="$(dirname "$annex_bdf")"
  metadata_events="$metadata_eeg_dir/${stem}_events.tsv"
  metadata_channels="$metadata_eeg_dir/${stem}_channels.tsv"
  if [[ ! -f "$metadata_events" ]]; then
    printf '%s\t%s\t%s\n' "$subject" "$run" "missing_events_tsv" >> "$exclusion_log"
    continue
  fi
  if [[ ! -f "$metadata_channels" ]]; then
    printf '%s\t%s\t%s\n' "$subject" "$run" "missing_channels_tsv" >> "$exclusion_log"
    continue
  fi

  qc_bdf="$qc_root/$subject/eeg/${stem}_eeg.bdf"
  if [[ -s "$qc_bdf" ]]; then
    bdf="$qc_bdf"
    events="$qc_root/$subject/eeg/${stem}_events.tsv"
    channels="$qc_root/$subject/eeg/${stem}_channels.tsv"
  else
    run_tmp="$stream_tmp/$stem"
    mkdir -p "$run_tmp"
    cp "$metadata_events" "$run_tmp/${stem}_events.tsv"
    cp "$metadata_channels" "$run_tmp/${stem}_channels.tsv"
    annex_target="$(readlink "$annex_bdf")"
    expected_bytes="$(printf '%s' "$annex_target" | sed -E 's/.*SHA256E-s([0-9]+)--.*/\1/')"
    expected_sha="$(printf '%s' "$annex_target" | sed -E 's/.*SHA256E-s[0-9]+--([0-9a-f]+)\.bdf.*/\1/')"
    bdf="$run_tmp/${stem}_eeg.bdf"
    download_parallel \
      "https://s3.amazonaws.com/openneuro.org/ds006780/$subject/eeg/${stem}_eeg.bdf" \
      "$bdf" "$expected_bytes"
    actual_sha="$(shasum -a 256 "$bdf" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      printf '%s\t%s\t%s\n' "$subject" "$run" "sha256_mismatch" >> "$exclusion_log"
      rm -rf -- "$run_tmp"
      continue
    fi
    events="$run_tmp/${stem}_events.tsv"
    channels="$run_tmp/${stem}_channels.tsv"
  fi

  MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python scripts/derive_ds006780_assr_features.py \
    --bdf "$bdf" --events "$events" --channels "$channels" --output "$output_root"

  if [[ "$bdf" == "$stream_tmp"/* ]]; then
    rm -rf -- "$(dirname "$bdf")"
  fi
done < <(find "$metadata_root" -path '*/eeg/*task-ASSR_run-*_eeg.bdf' -print | sort)

.venv/bin/python - "$metadata_root" "$output_root" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

metadata_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
feature_paths = sorted(output_root.glob("sub-*_task-ASSR_run-*_features.csv"))
frames = [pd.read_csv(path) for path in feature_paths]
all_runs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
all_runs.to_csv(output_root / "all_run_features.csv", index=False)

declared = list(metadata_root.glob("sub-*/eeg/*task-ASSR_run-*_eeg.bdf"))
exclusions = pd.read_csv(output_root / "run_exclusions.tsv", sep="\t")
manifest = {
    "declared_assr_runs": len(declared),
    "derived_runs": len(feature_paths),
    "derived_subjects": int(all_runs.subject.nunique()) if len(all_runs) else 0,
    "feature_rows": len(all_runs),
    "excluded_runs": len(exclusions),
    "exclusion_reasons": exclusions.reason.value_counts().to_dict(),
}
(output_root / "streaming_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
PY
