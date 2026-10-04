#!/usr/bin/env bash
set -euo pipefail

metadata_root="${1:-data/public/ds007648/v1.1.0}"
qc_root="${2:-data/public/behavioral_validation_qc/ds007648}"
output_root="${3:-outputs/derived/ds007648_trial_features}"

stream_tmp="$(mktemp -d /tmp/ds007648_stream.XXXXXX)"
cleanup() {
  rm -rf -- "$stream_tmp"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

download_resumable() {
  local url="$1"
  local destination="$2"
  local attempt=1
  local max_attempts=8
  until curl -fsSL -C - --speed-time 45 --speed-limit 1024 \
    "$url" -o "${destination}.partial"; do
    if (( attempt >= max_attempts )); then
      echo "Download failed after ${max_attempts} attempts: ${url}" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    sleep 3
  done
  mv "${destination}.partial" "$destination"
}

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
    curl -fsSL --retry 8 --speed-time 20 --speed-limit 65536 \
      -r "${start}-${end}" "$url" -o "$part" &
    pids+=("$!")
  done
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}"
  done
  .venv/bin/python scripts/join_binary_parts.py "$destination" "$expected_bytes" "${parts[@]}"
}

while IFS=$'\t' read -r subject _; do
  [[ "$subject" == "participant_id" ]] && continue
  [[ -s "$output_root/${subject}_summary.json" ]] && continue

  stem="${subject}_task-CrossModal"
  if [[ -s "$qc_root/$subject/eeg/${stem}_eeg.eeg" ]]; then
    vhdr="$qc_root/$subject/eeg/${stem}_eeg.vhdr"
    events="$qc_root/$subject/eeg/${stem}_events.tsv"
  else
    subject_tmp="$stream_tmp/$subject/eeg"
    mkdir -p "$subject_tmp"
    cp "$metadata_root/$subject/eeg/${stem}_events.tsv" "$subject_tmp/${stem}_events.tsv"
    for extension in vhdr vmrk; do
      filename="${stem}_eeg.${extension}"
      download_resumable \
        "https://s3.amazonaws.com/openneuro.org/ds007648/$subject/eeg/$filename" \
        "$subject_tmp/$filename"
    done
    filename="${stem}_eeg.eeg"
    annex_target="$(readlink "$metadata_root/$subject/eeg/$filename")"
    expected_bytes="$(printf '%s' "$annex_target" | sed -E 's/.*SHA256E-s([0-9]+)--.*/\1/')"
    download_parallel \
      "https://s3.amazonaws.com/openneuro.org/ds007648/$subject/eeg/$filename" \
      "$subject_tmp/$filename" "$expected_bytes"
    vhdr="$subject_tmp/${stem}_eeg.vhdr"
    events="$subject_tmp/${stem}_events.tsv"
  fi

  MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python scripts/derive_ds007648_trial_features.py \
    --vhdr "$vhdr" --events "$events" --output "$output_root"

  if [[ "$vhdr" == "$stream_tmp"/* ]]; then
    rm -f -- "$(dirname "$vhdr")/${stem}_eeg.eeg" \
      "$(dirname "$vhdr")/${stem}_eeg.vhdr" \
      "$(dirname "$vhdr")/${stem}_eeg.vmrk" \
      "$(dirname "$vhdr")/${stem}_events.tsv"
    rmdir "$(dirname "$vhdr")" "$(dirname "$(dirname "$vhdr")")"
  fi
done < "$metadata_root/participants.tsv"

.venv/bin/python -c '
import json
import sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
summaries = [json.loads(path.read_text()) for path in sorted(root.glob("sub-*_summary.json"))]
pd.DataFrame(summaries).to_csv(root / "subject_summary.csv", index=False)
trials = [pd.read_csv(path) for path in sorted(root.glob("sub-*_trial_features.csv"))]
pd.concat(trials, ignore_index=True).to_csv(root / "all_trial_features.csv", index=False)
print(json.dumps({"subjects": len(summaries), "trials": sum(len(x) for x in trials)}, indent=2))
' "$output_root"
