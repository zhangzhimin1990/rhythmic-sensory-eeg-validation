import unittest

from scripts.run_dryad_confirmatory_subject_amended_v2 import (
    AMENDMENT_IDS,
    resilient_arguments,
)


class DryadConfirmatoryRunnerAmendmentV2Tests(unittest.TestCase):
    def test_transport_change_is_limited_to_cache_and_timeout(self):
        original = [
            "python", "scripts/stream_remote_zip.py", "--url-file", "/tmp/url",
            "--extract-member", "Raw data stim/S8_Stim.bdf", "--output", "/tmp/S8.bdf",
            "--cache-mib", "1", "--timeout", "30", "--minimum-free-gib-after", "1.5",
        ]
        adjusted = resilient_arguments(original)
        changed = [(left, right) for left, right in zip(original, adjusted) if left != right]
        self.assertEqual(changed, [("1", "8"), ("30", "60")])

    def test_non_stream_command_is_unchanged(self):
        command = ["python", "scripts/derive_dryad_confirmatory_subject.py", "input.bdf"]
        self.assertEqual(resilient_arguments(command), command)

    def test_both_amendments_are_recorded(self):
        self.assertEqual(
            AMENDMENT_IDS,
            [
                "dryad_confirmatory_runner_input_guard_v1",
                "dryad_confirmatory_transport_resilience_v1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
