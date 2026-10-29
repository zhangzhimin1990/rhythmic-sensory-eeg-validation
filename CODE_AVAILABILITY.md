# Code and data availability

## Software archive

The analysis repository is available at:

- GitHub: https://github.com/zhangzhimin1990/rhythmic-sensory-eeg-validation
- Immutable v1.0.1 archive: https://doi.org/10.5281/zenodo.23682820
- License: MIT for original project code

The repository contains environment specifications, frozen protocols, dataset manifests, event-reconstruction logic, analysis and figure scripts, minimized derived inputs, model outputs, automated tests, and SHA-256 checksums.

## Public source datasets

The study reuses the following public datasets; source EEG remains under the original repositories and is not redistributed in this software archive.

- OpenNeuro ds004504, version 1.0.9: https://doi.org/10.18112/openneuro.ds004504.v1.0.9
- OpenNeuro ds006036, version 1.0.6: https://doi.org/10.18112/openneuro.ds006036.v1.0.6
- OpenNeuro ds005048, version 1.0.1: https://doi.org/10.18112/openneuro.ds005048.v1.0.1
- OpenNeuro ds006222, version 1.0.1: https://doi.org/10.18112/openneuro.ds006222.v1.0.1
- OpenNeuro ds007648, version 1.1.0: https://doi.org/10.18112/openneuro.ds007648.v1.1.0
- OpenNeuro ds006780, version 1.0.0: https://doi.org/10.18112/openneuro.ds006780.v1.0.0
- Personalized-theta stimulation dataset: https://doi.org/10.5061/dryad.t76hdr8dm

## Reproduction levels

### Derived-output reproduction

Run:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-qc.txt
bash scripts/reproduce_from_derived.sh .venv/bin/python
```

This level regenerates manuscript-facing figures and statistical summaries from the audited derived inputs included in the repository.

### Raw-to-derived reproduction

Download the source datasets from their repositories and place them under `data/public/`. Use the dataset-specific quality-control, event-reconstruction, feature-derivation, and modelling scripts in `scripts/`. Large source archives, credentials, signed URLs, and direct identifiers are excluded from this repository.

## Deliberate exclusions

The public package does not include source EEG, historical manuscript drafts, journal submission files, internal planning notes, correspondence, credentials, or temporary download artifacts. These materials are unnecessary for reproducing the reported numerical results and figures.
