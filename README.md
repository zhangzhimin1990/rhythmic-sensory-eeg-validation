# Human rhythmic sensory EEG validation across public cohorts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23682820.svg)](https://doi.org/10.5281/zenodo.23682820)

This repository contains the analysis code, minimized derived data, model outputs, figure-source tables, frozen protocols, and automated tests supporting the article **"From target engagement to cognitive validity in rhythmic sensory EEG: a claim-matched evaluation of seven public datasets."**

The study uses seven public human EEG datasets to test four distinct inferential transitions: response to mechanism, reliability to cognitive validity, mean active-control effect to within-participant neural–behavioral coupling, and in-sample association to held-out prediction. Raw EEG recordings are not redistributed.

## Repository contents

- `configs/`: frozen analysis protocols and spatial-neighbour definitions.
- `data/public/`: minimized public metadata and derived inputs permitted for redistribution.
- `outputs/`: audited model outputs, figure-source tables, participant-flow tables, and manuscript-facing derived results.
- `scripts/`: download helpers, quality control, feature derivation, statistical analysis, figure generation, and release auditing.
- `tests/`: automated tests for the retained analysis and reproduction workflow.
- `release_audit/`: machine-readable release inventory and privacy/portability audit.
- `requirements-qc.txt`: pinned Python environment used for release verification.
- `release_candidate_manifest.csv`: file-level sizes and SHA-256 checksums.

Historical manuscript drafts, journal-specific submission files, internal planning notes, and process commentary are deliberately excluded because they are not required to reproduce the reported analyses.

## Reproduce manuscript-facing outputs

Python 3.12 is recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-qc.txt
bash scripts/reproduce_from_derived.sh .venv/bin/python
```

This command reruns the claim-strength analysis, regenerates the manuscript figures and supplementary outputs from the included derived inputs, executes the retained test suite, and refreshes the file manifest.

## Recreate derived features from public source data

The source datasets and exact versions are listed in `CODE_AVAILABILITY.md`. Place downloaded source data under `data/public/` using the documented dataset identifiers. Raw-to-derived scripts and streaming helpers are provided in `scripts/`.

Large source archives are intentionally absent. Before downloading them, verify storage requirements and the source repository's licence. The Dryad streaming utilities process one archive member at a time and do not retain signed URLs.

## Scientific boundaries

- Frequency-following or phase-consistent responses are not labelled endogenous entrainment unless stronger dynamical criteria are tested.
- Split-half coefficients from one recording are not interpreted as between-day test–retest reliability.
- Detectability, reliability, population sensitivity, cognitive validity, within-participant coupling, and held-out prediction are separate measurement claims.
- Non-significant or imprecise estimates are not treated as proof of no association.
- The frozen 35-participant active-control analysis supports target engagement and acute task-performance advantage, but not mediation, a treatment-selection rule, chronic efficacy, or disease modification.

## Archived release

The immutable v1.0.1 software archive is available at [Zenodo](https://doi.org/10.5281/zenodo.23682820). Release metadata are provided in `CITATION.cff` and `.zenodo.json`.

## Licence and third-party material

Original project code is released under the MIT License. Public source datasets remain governed by their original repository terms and are not relicensed here. See `THIRD_PARTY_NOTICES.md` for the boundary between project code, public derived data, and third-party material.
