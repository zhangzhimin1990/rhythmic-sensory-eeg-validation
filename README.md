# Rhythmic sensory EEG claim-matched validation

This repository contains the author-generated analysis code, minimized derived inputs, machine-readable results, figure source data, and automated tests supporting the Research Article:

> Claim-matched validation dissociates robust stimulus-locked electroencephalographic responses from cognitive validity and incremental prediction

The article uses three public human EEG datasets to test four distinct measurement claims: stimulus-locked target engagement, within-recording reliability, matched concurrent cognitive validity, and participant-held-out incremental prediction. The package does not redistribute source EEG recordings.

## Datasets

- Dryad DOI `10.5061/dryad.t76hdr8dm`: personalized theta-plus versus non-rhythmic auditory stimulation in older adults; frozen confirmatory cohort `n=35`.
- OpenNeuro `ds007648` version `1.1.0`: simultaneous 36-Hz visual and 40-Hz auditory frequency tagging; analysis cohort `n=20`.
- OpenNeuro `ds006780` version `1.0.0`: 27/40-Hz auditory steady-state oddball recordings; `n=111` for d-prime and `n=109` for FSIQ.

## Package scope

- `configs/`: frozen confirmatory, spatial-robustness, and prediction-audit specifications.
- `scripts/`: group analyses, matched-validity analyses, participant-level prediction and uncertainty audits, and figure/table builders.
- `outputs/`: minimized deidentified derived inputs and machine-readable results needed to reproduce the reported summaries.
- `figures/`: final vector figures and their source-data tables.
- `tables/`: main and supplementary source tables.
- `tests/`: targeted regression tests for the focused analysis.

Participant-level files use analytic identifiers and contain no names, contact details, dates of birth, source archives, credentials, signed URLs, or copied raw EEG.

## Reproduce from derived inputs

Python 3.12 is recommended.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/reproduce_from_derived.sh .venv/bin/python
```

The reproduction script reruns the focused statistical audits and tests. Final vector figures and source tables are also supplied so the submitted visual results can be inspected without source EEG.

## Interpretation boundary

The package supports stimulus-locked scalp-EEG findings. It does not establish source localization, endogenous entrainment, mediation, chronic efficacy, disease modification, or a treatment-selection biomarker. Intervals spanning effects in both directions are reported as unresolved rather than as proof of absence.

## License and third-party materials

Author-generated code is released under the MIT License. Public datasets, third-party software, article text, and publisher material retain their own terms; see `THIRD_PARTY_NOTICES.md`.

