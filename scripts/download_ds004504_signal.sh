#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-data/public/ds004504/qc_sample_derivatives}"
meta_dir="${2:-/tmp/eeg-paper-ds004504-meta-20260917}"
source_prefix="${3:-derivatives}"
subject_mode="${4:-sample}"
base_url="https://s3.amazonaws.com/openneuro.org/ds004504"
subjects=(001 002 037 038 066 067)
if [[ "$subject_mode" == "all" ]]; then
  subjects=()
  for index in $(seq 1 88); do
    subjects+=("$(printf '%03d' "$index")")
  done
fi

mkdir -p "$out_dir"
cp "$meta_dir/dataset_description.json" "$out_dir/"
cp "$meta_dir/participants.tsv" "$out_dir/"
cp "$meta_dir/participants.json" "$out_dir/"

for subject in "${subjects[@]}"; do
  subject_id="sub-${subject}"
  eeg_dir="$out_dir/$subject_id/eeg"
  mkdir -p "$eeg_dir"
  stem="${subject_id}_task-eyesclosed"
  cp "$meta_dir/$subject_id/eeg/${stem}_eeg.json" "$eeg_dir/"
  cp "$meta_dir/$subject_id/eeg/${stem}_channels.tsv" "$eeg_dir/"

  target="$eeg_dir/${stem}_eeg.set"
  sample_source="data/public/ds004504/qc_sample_derivatives/$subject_id/eeg/${stem}_eeg.set"
  if [[ ! -s "$target" && "$source_prefix" == "derivatives" && -s "$sample_source" ]]; then
    ln "$sample_source" "$target"
  fi
  if [[ ! -s "$target" ]]; then
    partial="${target}.partial"
    source_path="$subject_id/eeg/${stem}_eeg.set"
    if [[ -n "$source_prefix" ]]; then
      source_path="$source_prefix/$source_path"
    fi
    curl -fL --retry 4 --retry-delay 2 "$base_url/$source_path" -o "$partial"
    mv "$partial" "$target"
  fi
done

echo "Downloaded ds004504 ${source_prefix:-raw} signal to $out_dir"
