import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_dryad_confirmatory_freeze_chain import verify_csv_manifest, verify_json_manifest


class DryadConfirmatoryFreezeChainTests(unittest.TestCase):
    def test_csv_manifest_verifies_and_detects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "a.txt"
            asset.write_text("frozen", encoding="utf-8")
            manifest = root / "m.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
                writer.writeheader()
                writer.writerow({"relative_path": "a.txt", "size_bytes": 6, "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()})
            self.assertEqual(verify_csv_manifest(manifest, root)["assets_verified"], 1)
            asset.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed frozen asset"):
                verify_csv_manifest(manifest, root)

    def test_json_manifest_verifies_file_list_hash_and_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "a.txt"
            asset.write_text("frozen", encoding="utf-8")
            files = [{"path": "a.txt", "bytes": 6, "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}]
            canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            manifest = root / "m.json"
            manifest.write_text(json.dumps({"files": files, "manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest()}), encoding="utf-8")
            self.assertEqual(verify_json_manifest(manifest, root)["assets_verified"], 1)
            data = json.loads(manifest.read_text())
            data["manifest_sha256"] = "0" * 64
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "file-list hash mismatch"):
                verify_json_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
