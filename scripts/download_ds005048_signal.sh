#!/usr/bin/env bash
set -euo pipefail

destination_root="${1:-data/public/ds005048/v1.0.1}"
seed_root="${2:-data/public/ds005048/qc_sample}"
base_url="https://s3.amazonaws.com/openneuro.org/ds005048"

for subject_number in $(seq -w 1 35); do
  subject="sub-${subject_number}"
  destination_dir="${destination_root}/${subject}/eeg"
  seed_dir="${seed_root}/${subject}/eeg"
  mkdir -p "${destination_dir}"

  for extension in set fdt; do
    filename="${subject}_task-40HzAuditoryEntrainment_eeg.${extension}"
    destination="${destination_dir}/${filename}"
    seed="${seed_dir}/${filename}"
    if [[ -s "${destination}" ]]; then
      continue
    fi
    if [[ -s "${seed}" ]]; then
      ln "${seed}" "${destination}"
      continue
    fi
    curl -fsSL --retry 3 -o "${destination}.partial" \
      "${base_url}/${subject}/eeg/${filename}"
    mv "${destination}.partial" "${destination}"
  done
done

find "${destination_root}" -type f \( -name '*_eeg.set' -o -name '*_eeg.fdt' \) -print0 \
  | xargs -0 -n1 stat -f '%z %N'
