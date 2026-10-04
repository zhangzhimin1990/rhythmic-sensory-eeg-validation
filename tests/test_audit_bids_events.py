import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_bids_events import audit_auditory, audit_visual


def write_tsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class AuditBidsEventsTests(unittest.TestCase):
    def test_auditory_counts_blocks_and_checks_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tsv(
                root / "participants.tsv",
                ["participant_id", "Group"],
                [{"participant_id": "sub-01", "Group": "A"}],
            )
            write_tsv(
                root / "sub-01/eeg/sub-01_task-test_events.tsv",
                ["onset", "duration", "value", "trial_type"],
                [
                    {"onset": 5, "duration": 40, "value": 2, "trial_type": "Stimulus"},
                    {"onset": 45, "duration": 20, "value": 1, "trial_type": "Rest"},
                    {"onset": 65, "duration": 40, "value": 2, "trial_type": "Stimulus"},
                ],
            )
            eeg_json = root / "sub-01/eeg/sub-01_task-test_eeg.json"
            eeg_json.write_text(json.dumps({"RecordingDuration": 105}), encoding="utf-8")

            result = audit_auditory(root)

            self.assertEqual(result["n_event_files"], 1)
            self.assertEqual(result["stimulus_block_count"], {2: 1})
            self.assertTrue(result["all_blocks_alternate"])
            self.assertTrue(result["all_events_within_recording"])

    def test_visual_uses_eye_window_inside_current_pulse_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tsv(
                root / "participants.tsv",
                ["participant_id", "Group"],
                [{"participant_id": "sub-001", "Group": "C"}],
            )
            rows = [
                {"onset": 0.0, "duration": 0, "sample": 0, "value": "PHOTO 5Hz   "},
            ]
            rows.extend(
                {"onset": index / 5.0, "duration": 0, "sample": index * 100, "value": "Photo/HV mark"}
                for index in range(51)
            )
            rows.extend(
                [
                    {"onset": 2.0, "duration": 0, "sample": 1000, "value": "open eyes"},
                    {"onset": 5.5, "duration": 0, "sample": 2750, "value": "closed eyes"},
                    {"onset": 20.0, "duration": 0, "sample": 10000, "value": "PHOTO 10Hz"},
                ]
            )
            rows.extend(
                {"onset": 20.0 + index / 10.0, "duration": 0, "sample": 10000 + index * 50, "value": "Photo/HV mark"}
                for index in range(101)
            )
            # This pair is deliberately outside the second pulse train. It
            # must not be attached to the malformed 10 Hz block.
            rows.extend(
                [
                    {"onset": 50.0, "duration": 0, "sample": 25000, "value": "open eyes"},
                    {"onset": 53.0, "duration": 0, "sample": 26500, "value": "closed eyes"},
                ]
            )
            rows.sort(key=lambda row: row["onset"])
            write_tsv(
                root / "sub-001/eeg/sub-001_task-test_events.tsv",
                ["onset", "duration", "sample", "value"],
                rows,
            )

            result = audit_visual(root)

            self.assertEqual(result["n_frequency_blocks"], 2)
            self.assertEqual(result["n_usable_open_eye_blocks"], 1)
            self.assertEqual(result["n_blocks_missing_open_close_pair"], 1)
            self.assertAlmostEqual(result["open_eye_duration_s"]["median"], 3.5)
            self.assertLess(result["pulse_frequency_error_pct"]["max"], 0.01)


if __name__ == "__main__":
    unittest.main()
