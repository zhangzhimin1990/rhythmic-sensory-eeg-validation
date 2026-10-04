import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseArchiveReadinessTests(unittest.TestCase):
    def test_readme_records_authorized_mit_release_and_pending_deposition(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("authors approved public release", text)
        self.assertIn("MIT License", text)
        self.assertIn("remains local until a clean public repository", text)

    def test_third_party_notice_preserves_data_and_code_boundaries(self):
        text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for fragment in (
            "does not redistribute the source EEG recordings",
            "doi:10.1162/IMAG.a.1229",
            "authors approved public release",
            "does not relicense source datasets",
            "This notice is not itself a software license",
        ):
            self.assertIn(fragment, text)

    def test_release_checklist_requires_authorization_and_clean_candidate(self):
        text = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
        for fragment in (
            "[x] All authors approved",
            "selected the MIT License and added it as `LICENSE`",
            "clean public repository from the audited candidate",
            "Archive that tag in Zenodo or OSF",
        ):
            self.assertIn(fragment, text)

    def test_authorized_mit_license_and_citation_metadata_are_present(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2026 Zhimin Zhang, Hui Yang, and Yuwen Li", license_text)
        self.assertTrue((ROOT / "CITATION.cff").exists())
        self.assertFalse((ROOT / "LICENSE.txt").exists())

    def test_zenodo_metadata_and_release_notes_are_deposition_ready(self):
        metadata = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["upload_type"], "software")
        self.assertEqual(metadata["access_right"], "open")
        self.assertEqual(metadata["license"], "MIT")
        self.assertEqual([creator["name"] for creator in metadata["creators"]], [
            "Zhang, Zhimin", "Yang, Hui", "Li, Yuwen"
        ])
        self.assertNotIn("orcid", metadata["creators"][1])
        notes = (ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")
        self.assertIn("Release v1.0.0", notes)
        self.assertIn("254 tests", notes)
        self.assertIn("zero Critical, High, Medium, or Low findings", notes)


if __name__ == "__main__":
    unittest.main()
