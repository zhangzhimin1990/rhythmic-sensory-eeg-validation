#!/usr/bin/env bash
set -euo pipefail

metadata_root="${1:-data/public/ds006222/v1.0.1}"
destination_root="${2:-data/public/ds006222/qc_sample}"
base_url="https://s3.amazonaws.com/openneuro.org/ds006222"

# Two long recordings per released group, selected before signal inspection.
subjects=(S060 S018 S058 S055 S081 S084)

mkdir -p "$destination_root"
for file in dataset_description.json participants.tsv participants.json README; do
  cp "$metadata_root/$file" "$destination_root/$file"
done

for subject in "${subjects[@]}"; do
  subject_id="sub-${subject}"
  session_id="ses-1"
  source_dir="$metadata_root/$subject_id/$session_id/eeg"
  destination_dir="$destination_root/$subject_id/$session_id/eeg"
  stem="${subject_id}_${session_id}_task-PVT"
  mkdir -p "$destination_dir"

  for suffix in channels.tsv eeg.json events.tsv; do
    cp "$source_dir/${stem}_${suffix}" "$destination_dir/${stem}_${suffix}"
  done

  for extension in set fdt; do
    filename="${stem}_eeg.${extension}"
    destination="$destination_dir/$filename"
    if [[ -s "$destination" ]]; then
      continue
    fi
    # This host has an older curl whose built-in retry truncates a partial file.
    # Retry at the shell level so every new curl process re-evaluates `-C -` and
    # resumes from the bytes already on disk.
    attempt=1
    max_attempts=8
    until curl -fL -C - \
      --speed-time 45 \
      --speed-limit 1024 \
      "$base_url/$subject_id/$session_id/eeg/$filename" \
      -o "${destination}.partial"; do
      if (( attempt >= max_attempts )); then
        echo "Download failed after ${max_attempts} resumable attempts: ${filename}" >&2
        exit 1
      fi
      attempt=$((attempt + 1))
      sleep 3
    done
    mv "${destination}.partial" "$destination"
  done
done

find "$destination_root" -type f \( -name '*_eeg.set' -o -name '*_eeg.fdt' \) -print0 \
  | xargs -0 -n1 stat -f '%z %N'
