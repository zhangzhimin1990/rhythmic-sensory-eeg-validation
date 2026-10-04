import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.execute_dryad_confirmatory_group_locked import (
    require_complete_cohort,
    verify_csv_freeze_manifest,
    verify_json_freeze_manifest,
)


class DryadConfirmatoryExecutionGateTests(unittest.TestCase):
    def _write_protocol(self, root: Path) -> Path:
        path = root / "protocol.json"
        path.write_text(json.dumps({"confirmatory_subjects": [1, 2]}), encoding="utf-8")
        return path

    def _write_subject(self, root: Path, subject: int) -> None:
        directory = root / f"S{subject}"
        directory.mkdir(parents=True)
        source_hash = f"hash-{subject}"
        (directory / "confirmatory_run_summary.json").write_text(json.dumps({
            "status": "confirmatory_subject_complete",
            "source_bdf_sha256": source_hash,
            "temporary_bdf_deleted_after_verification": True,
            "individual_theta_hz": 4.0,
            "post_freeze_amendment_ids": [],
        }), encoding="utf-8")
        (directory / "confirmatory_subject_summary.json").write_text(json.dumps({
            "status": "confirmatory_subject_derivation_complete",
            "source_bdf_sha256": source_hash,
            "analysis_frequency_hz": 5.32,
            "n_technical_usable_epochs_reproduced": 380,
            "n_good_frozen_qc_channels": 62,
            "frontocentral_roi": ["Fz", "FCz", "Cz", "FC1"],
        }), encoding="utf-8")

    def test_incomplete_cohort_blocks_group_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = self._write_protocol(root)
            subjects = root / "subjects"
            self._write_subject(subjects, 1)
            with self.assertRaisesRegex(RuntimeError, "1/2 complete"):
                require_complete_cohort(protocol, subjects)

    def test_exact_complete_cohort_unlocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = self._write_protocol(root)
            subjects = root / "subjects"
            self._write_subject(subjects, 1)
            self._write_subject(subjects, 2)
            result = require_complete_cohort(protocol, subjects)
            self.assertEqual(result["status"], "group_analysis_unlocked")
            self.assertEqual(result["complete_subjects"], 2)
            self.assertFalse(result["contains_condition_effects"])

    def test_csv_manifest_detects_frozen_asset_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "asset.txt"
            asset.write_text("frozen", encoding="utf-8")
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
                writer.writeheader()
                writer.writerow({"relative_path": "asset.txt", "size_bytes": 6, "sha256": digest})
            self.assertEqual(verify_csv_freeze_manifest(manifest, root), 1)
            asset.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen asset changed"):
                verify_csv_freeze_manifest(manifest, root)

    def test_json_manifest_detects_frozen_asset_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "asset.txt"
            asset.write_text("frozen", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": [{
                "path": "asset.txt",
                "bytes": 6,
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            }]}), encoding="utf-8")
            self.assertEqual(verify_json_freeze_manifest(manifest, root), 1)
            asset.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen inference asset changed"):
                verify_json_freeze_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
