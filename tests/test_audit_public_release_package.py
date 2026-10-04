import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_public_release_package import audit_release


class PublicReleaseAuditTests(unittest.TestCase):
    def test_clean_relative_release_passes_machine_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "release"
            output = root / "audit"
            package.mkdir()
            (package / "analysis.py").write_text(
                "from pathlib import Path\nDATA = Path('data/public')\n", encoding="utf-8"
            )
            summary = audit_release(package, output)
            self.assertTrue(summary["machine_audit_passed"])
            self.assertTrue(summary["ethics_license_manual_review_required"])

    def test_raw_eeg_and_absolute_path_fail_machine_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "release"
            output = root / "audit"
            package.mkdir()
            (package / "private.edf").write_bytes(b"not-real-eeg")
            (package / "config.txt").write_text(
                "input=" + "/" + "Users/researcher/private/patient.csv\n", encoding="utf-8"
            )
            summary = audit_release(package, output)
            self.assertFalse(summary["machine_audit_passed"])
            self.assertGreaterEqual(summary["issue_counts"]["Critical"], 1)
            self.assertGreaterEqual(summary["issue_counts"]["High"], 1)

    def test_credentials_and_identifiers_are_critical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "release"
            output = root / "audit"
            package.mkdir()
            (package / "notes.md").write_text(
                "contact=" + "test.person" + "@example.org\n" +
                "api_" + "key=abcdefghijk12345\n", encoding="utf-8"
            )
            summary = audit_release(package, output)
            self.assertFalse(summary["machine_audit_passed"])
            self.assertGreaterEqual(summary["issue_counts"]["Critical"], 2)

    def test_decimal_results_and_checksums_are_not_identifiers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "release"
            output = root / "audit"
            package.mkdir()
            (package / "results.csv").write_text(
                "estimate,sha256\n0.054287827134550826,a15610852975bcdef\n",
                encoding="utf-8",
            )
            summary = audit_release(package, output)
            self.assertTrue(summary["machine_audit_passed"])

    def test_standalone_phone_and_plausible_prc_id_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "release"
            output = root / "audit"
            package.mkdir()
            phone = "138" + "12345678"
            identity = "110105" + "19900101" + "123X"
            (package / "identifiers.csv").write_text(
                f"phone,id\n{phone},{identity}\n", encoding="utf-8"
            )
            summary = audit_release(package, output)
            self.assertFalse(summary["machine_audit_passed"])
            self.assertGreaterEqual(summary["issue_counts"]["Critical"], 2)


if __name__ == "__main__":
    unittest.main()
