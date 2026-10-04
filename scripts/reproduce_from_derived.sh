#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXECUTABLE="${1:-python3}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EEG_MPL_CACHE="${TMPDIR:-/tmp}/rhythmic_eeg_mplconfig"

cd "$PROJECT_ROOT"
mkdir -p "$EEG_MPL_CACHE"

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$EEG_MPL_CACHE" "$PYTHON_EXECUTABLE" scripts/analyze_validation_strength.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$EEG_MPL_CACHE" "$PYTHON_EXECUTABLE" scripts/build_foundation_figures_v02.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$EEG_MPL_CACHE" "$PYTHON_EXECUTABLE" scripts/build_behavioral_validity_figures.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$EEG_MPL_CACHE" "$PYTHON_EXECUTABLE" scripts/build_dryad_theta_figure.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$EEG_MPL_CACHE" "$PYTHON_EXECUTABLE" scripts/build_dryad_confirmatory_figure7.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$EEG_MPL_CACHE" "$PYTHON_EXECUTABLE" scripts/build_dryad_confirmatory_supplement.py
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_EXECUTABLE" scripts/build_consolidated_manuscript.py
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_EXECUTABLE" scripts/build_alzheimers_dementia_manuscript.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$EEG_MPL_CACHE" "$PYTHON_EXECUTABLE" -m unittest discover -s tests -v
if [[ -f "$PROJECT_ROOT/release_candidate_manifest.csv" ]]; then
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_EXECUTABLE" scripts/refresh_release_manifest.py "$PROJECT_ROOT"
else
  echo "No release_candidate_manifest.csv at project root; skipped manifest refresh."
fi

echo "Derived-output reproduction completed successfully."
