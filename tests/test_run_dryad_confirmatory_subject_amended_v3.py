import unittest

from scripts.run_dryad_confirmatory_subject_amended_v3 import resumable_arguments


class DryadConfirmatoryRunnerAmendmentV3Tests(unittest.TestCase):
    def test_only_extraction_command_is_replaced(self):
        original = [
            "python", "scripts/stream_remote_zip.py", "--url-file", "/tmp/url",
            "--extract-member", "Raw data stim/S8_Stim.bdf", "--output", "/tmp/S8.bdf",
            "--cache-mib", "1", "--timeout", "30", "--minimum-free-gib-after", "1.5",
        ]
        adjusted = resumable_arguments(original)
        self.assertEqual(adjusted[0], "python")
        self.assertIn("scripts.extract_remote_zip_member_resumable", adjusted)
        self.assertIn("Raw data stim/S8_Stim.bdf", adjusted)
        self.assertIn("/tmp/S8.bdf", adjusted)
        self.assertEqual(adjusted[-1], "1.5")

    def test_derivation_command_is_unchanged(self):
        command = ["python", "scripts/derive_dryad_confirmatory_subject.py", "input.bdf"]
        self.assertEqual(resumable_arguments(command), command)


if __name__ == "__main__":
    unittest.main()
