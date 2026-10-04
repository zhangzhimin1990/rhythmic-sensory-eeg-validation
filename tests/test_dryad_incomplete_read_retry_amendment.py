import http.client
import unittest

from scripts.extract_remote_zip_member_resumable_v2 import RETRYABLE_EXCEPTIONS
from scripts.run_dryad_confirmatory_subject_amended_v4 import resumable_arguments_v2


class DryadIncompleteReadRetryAmendmentTests(unittest.TestCase):
    def test_incomplete_read_is_retryable(self):
        self.assertIn(http.client.IncompleteRead, RETRYABLE_EXCEPTIONS)

    def test_runner_uses_v2_extractor_and_preserves_member(self):
        original = [
            "python", "scripts/stream_remote_zip.py", "--url-file", "/tmp/url",
            "--extract-member", "Raw data stim/S20_Stim.bdf", "--output", "/tmp/S20.bdf",
            "--cache-mib", "1", "--timeout", "30", "--minimum-free-gib-after", "1.5",
        ]
        adjusted = resumable_arguments_v2(original)
        self.assertIn("scripts.extract_remote_zip_member_resumable_v2", adjusted)
        self.assertIn("Raw data stim/S20_Stim.bdf", adjusted)
        self.assertIn("/tmp/S20.bdf", adjusted)


if __name__ == "__main__":
    unittest.main()
