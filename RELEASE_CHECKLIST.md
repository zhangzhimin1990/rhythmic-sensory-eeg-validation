# Public archive release checklist

Complete every item against the exact candidate directory before creating a public Git tag or archival DOI.

## Author and institutional approval

- [x] All authors approved the public code and derived-output package.
- [ ] Author order, affiliations, CRediT roles, corresponding author, and ORCID records are final.
- [ ] The institutional ethics approval, exemption, or formal determination for the secondary analysis is documented.
- [x] Funding and competing-interest statements are final.

## License and attribution

- [x] The authors selected the MIT License and added it as `LICENSE`.
- [ ] `THIRD_PARTY_NOTICES.md` has been reviewed by the authors or institutional research office.
- [x] No source EEG, source archives, signed URLs, copied third-party scripts, publisher PDFs, or third-party figures are present.
- [x] Dataset DOIs and source papers are cited in the manuscript and repository documentation.

## Reproducibility and security

- [x] `bash scripts/reproduce_from_derived.sh <python>` and the included independent test suite pass in a clean candidate.
- [x] All 254 package tests pass and the test count is recorded in the submission metadata.
- [x] `scripts/audit_public_release_package.py` reports zero Critical, High, Medium, and Low findings.
- [x] `release_candidate_manifest.csv` matches the audited local candidate directory.
- [ ] The manuscript's repository URL, tag, commit, license, and archive DOI match the deposited release.

## Git and archive deposition

- [ ] Create a clean public repository from the audited candidate, not from the full working directory.
- [x] Add the selected `LICENSE` and release-candidate `CITATION.cff`; update the latter with the public URL, final tag, and archive DOI after deposition.
- [ ] Commit the immutable candidate and record the commit SHA.
- [ ] Create and push a signed or annotated version tag.
- [ ] Archive that tag in Zenodo or OSF and obtain a version DOI.
- [ ] Update the manuscript and cover letter with the public URL and version DOI.
- [ ] Re-render the final manuscript after inserting author and archive metadata.
