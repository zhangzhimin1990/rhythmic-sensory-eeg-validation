# Release v1.0.1

This maintenance release adds the previously omitted `openpyxl==3.1.5` test dependency and updates citation metadata for the NeuroImage submission. Analysis code, derived data, numerical results, figures, claim boundaries, and the 254-test suite are unchanged from v1.0.0.

The immutable release resolves to commit `8c8077ed892d330c2bee589c548b08da35575cf7`, Git tag `v1.0.1`, and Zenodo version DOI `10.5281/zenodo.23682820`.

## Release v1.0.0

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

The archived release resolves to commit `86ad30436aad2ecda28e59a4127188011b5eca4d`, Git tag `v1.0.0`, and Zenodo version DOI `10.5281/zenodo.23137958`.
