import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_dryad_confirmatory_subject import (
    MINIMUM_FREE_AFTER_EXTRACTION_GIB,
    load_subject,
    verify_freeze,
)


class DryadConfirmatoryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.metadata = Path(self.temporary.name) / "metadata.xlsx"
        pd.DataFrame({
            "subject": range(1, 45),
            "individual_freq": [3.0] * 44,
        }).to_excel(self.metadata, sheet_name="Dataset", index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_sequential_extraction_keeps_fixed_disk_reserve(self):
        self.assertEqual(MINIMUM_FREE_AFTER_EXTRACTION_GIB, 1.5)

    def test_load_subject_rejects_excluded_participant(self):
        with self.assertRaisesRegex(ValueError, "locked confirmatory set"):
            load_subject(
                1,
                Path("outputs/qc/dryad_raw_stream/stim_zip_members.csv"),
                Path("configs/dryad_confirmatory_protocol_v1.json"),
                self.metadata,
            )

    def test_load_subject_returns_locked_theta_and_size(self):
        result = load_subject(
            3,
            Path("outputs/qc/dryad_raw_stream/stim_zip_members.csv"),
            Path("configs/dryad_confirmatory_protocol_v1.json"),
            self.metadata,
        )
        self.assertEqual(result["subject"], 3)
        self.assertGreater(result["expected_uncompressed_bytes"], 0)
        self.assertGreaterEqual(result["individual_theta_hz"], 3.0)

    def test_freeze_verifier_rejects_asset_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "asset.txt"
            asset.write_text("frozen", encoding="utf-8")
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            freeze = root / "freeze"
            freeze.mkdir()
            manifest = freeze / "confirmatory_freeze_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
                writer.writeheader()
                writer.writerow({"relative_path": "asset.txt", "size_bytes": asset.stat().st_size, "sha256": digest})
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (freeze / "confirmatory_freeze_summary.json").write_text(json.dumps({
                "status": "frozen_before_confirmatory_signal_reaccess",
                "manifest_sha256": manifest_hash,
            }), encoding="utf-8")
            self.assertEqual(verify_freeze(freeze, root), manifest_hash)
            asset.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                verify_freeze(freeze, root)


if __name__ == "__main__":
    unittest.main()
