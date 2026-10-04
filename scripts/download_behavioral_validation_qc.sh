#!/usr/bin/env bash
set -euo pipefail

crossmodal_metadata="${1:-data/public/ds007648/v1.1.0}"
sfari_metadata="${2:-data/public/ds006780/v1.0.0}"
destination_root="${3:-data/public/behavioral_validation_qc}"

# Fixed before neural-signal inspection. CrossModal spans metadata/behavior
# edge cases; SFARI spans group and age without using any neural result.
crossmodal_subjects=(03 04 10 13 14 23)
sfari_subjects=(10025 10764 11038 11325 1501 1589)

download_resumable() {
  local url="$1"
  local destination="$2"
  local attempt=1
  local max_attempts=8
  if [[ -s "$destination" ]]; then
    return
  fi
  mkdir -p "$(dirname "$destination")"
  until curl -fL -C - --speed-time 45 --speed-limit 1024 \
    "$url" -o "${destination}.partial"; do
    if (( attempt >= max_attempts )); then
      echo "Download failed after ${max_attempts} resumable attempts: ${url}" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    sleep 3
  done
  mv "${destination}.partial" "$destination"
}

for subject in "${crossmodal_subjects[@]}"; do
  subject_id="sub-${subject}"
  source_dir="$crossmodal_metadata/$subject_id/eeg"
  destination_dir="$destination_root/ds007648/$subject_id/eeg"
  stem="${subject_id}_task-CrossModal"
  mkdir -p "$destination_dir"
  for suffix in events.tsv events.json channels.tsv eeg.json electrodes.tsv coordsystem.json; do
    source="$source_dir/${stem}_${suffix}"
    [[ -f "$source" ]] || source="$source_dir/${subject_id}_${suffix}"
    [[ -f "$source" ]] && cp "$source" "$destination_dir/$(basename "$source")"
  done
  for extension in vhdr vmrk eeg; do
    filename="${stem}_eeg.${extension}"
    download_resumable \
      "https://s3.amazonaws.com/openneuro.org/ds007648/$subject_id/eeg/$filename" \
      "$destination_dir/$filename"
  done
done

for subject in "${sfari_subjects[@]}"; do
  subject_id="sub-${subject}"
  source_dir="$sfari_metadata/$subject_id/eeg"
  destination_dir="$destination_root/ds006780/$subject_id/eeg"
  mkdir -p "$destination_dir"
  for bdf_link in "$source_dir"/${subject_id}_task-ASSR_run-*_eeg.bdf; do
    stem="$(basename "$bdf_link" _eeg.bdf)"
    for suffix in events.tsv events.json channels.tsv eeg.json; do
      source="$source_dir/${stem}_${suffix}"
      [[ -f "$source" ]] && cp "$source" "$destination_dir/$(basename "$source")"
    done
    filename="${stem}_eeg.bdf"
    download_resumable \
      "https://s3.amazonaws.com/openneuro.org/ds006780/$subject_id/eeg/$filename" \
      "$destination_dir/$filename"
  done
done

find "$destination_root" -type f \( -name '*.eeg' -o -name '*.vhdr' -o -name '*.vmrk' -o -name '*.bdf' \) -print0 \
  | xargs -0 -n1 stat -f '%z %N'
