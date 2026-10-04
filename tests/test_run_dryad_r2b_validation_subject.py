import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_dryad_r2b_validation_subject import (
    DEVELOPMENT_SUBJECTS,
    VALIDATION_SUBJECTS,
    load_inventory,
    verify_freeze,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DryadR2bValidationRunnerTests(unittest.TestCase):
    def test_validation_set_is_exact_and_disjoint(self):
        self.assertEqual(len(VALIDATION_SUBJECTS), 39)
        self.assertFalse(VALIDATION_SUBJECTS.intersection(DEVELOPMENT_SUBJECTS))

    def test_inventory_requires_exact_44_subjects(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "members.csv"
            pd.DataFrame(
                {
                    "member": [f"Raw data stim/S{s}_Stim.bdf" for s in range(1, 45)],
                    "uncompressed_bytes": range(101, 145),
                }
            ).to_csv(path, index=False)
            self.assertEqual(load_inventory(path, 1)["expected_uncompressed_bytes"], 101)
            with self.assertRaises(ValueError):
                load_inventory(path, 2)

    def test_freeze_verifier_rejects_asset_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            freeze = Path(temporary) / "freeze"
            root.mkdir()
            freeze.mkdir()
            asset = root / "asset.txt"
            asset.write_text("frozen", encoding="utf-8")
            manifest = freeze / "protocol_freeze_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
                writer.writeheader()
                writer.writerow({"relative_path": "asset.txt", "size_bytes": 6, "sha256": digest(asset)})
            (freeze / "protocol_freeze_summary.json").write_text(
                json.dumps(
                    {
                        "status": "frozen_before_validation_data_access",
                        "manifest_sha256": digest(manifest),
                    }
                ),
                encoding="utf-8",
            )
            verify_freeze(freeze, root)
            asset.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen asset changed"):
                verify_freeze(freeze, root)


if __name__ == "__main__":
    unittest.main()
