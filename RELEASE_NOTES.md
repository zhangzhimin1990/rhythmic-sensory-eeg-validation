# Release v1.0.0

This is the first archival release of the analysis package supporting **“Testing cognitive validity of rhythmic-sensory EEG target engagement.”**

## Scientific scope

- Seven public human EEG cohorts spanning rhythmic auditory, visual, and audiovisual stimulation and disease-relevant recordings.
- Separate tests of target-response detectability, within-recording reliability, experimental sensitivity, concurrent cognitive validity, active-control behavioral effects, within-participant neural–behavioral coupling, and held-out prediction.
- A condition-blind, frozen raw-EEG confirmation in 35 healthy older adults.
- Explicit claim-release rules that prevent frequency-following responses from being interpreted as cognitive mechanisms, treatment-selection markers, or surrogate endpoints without the required supporting evidence.

## Reproducibility and disclosure controls

- Figures 1–7, manuscript tables, and supplementary outputs can be regenerated from the included audited derived outputs.
- The independent package test suite contains 254 tests.
- The release-candidate audit reports zero Critical, High, Medium, or Low findings.
- Source EEG, source archives, signed URLs, direct identifiers, credentials, publisher files, and copied third-party scripts are excluded.
- Participant-level derived tables use minimized fields and sequential analytic identifiers where applicable.

## Licensing and citation

Author-generated analysis code is released under the MIT License. Source datasets, third-party software, article text, publisher materials, and external scripts retain their original terms; see `THIRD_PARTY_NOTICES.md`. Citation metadata are provided in `CITATION.cff` and `.zenodo.json`.

## Reproduction entry point

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-qc.txt
bash scripts/reproduce_from_derived.sh .venv/bin/python
```

The repository URL, immutable commit, Git tag, and archival DOI will be inserted into the manuscript after deposition.
