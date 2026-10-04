import csv
import tempfile
import unittest
from pathlib import Path

from scripts.refresh_release_manifest import refresh


class RefreshReleaseManifestTests(unittest.TestCase):
    def test_refresh_hashes_all_nonmanifest_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            (package / "a.txt").write_text("a", encoding="utf-8")
            (package / "nested").mkdir()
            (package / "nested/b.txt").write_text("b", encoding="utf-8")
            result = refresh(package)
            self.assertEqual(result["hashed_files"], 2)
            with (package / "release_candidate_manifest.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["relative_path"] for row in rows}, {"a.txt", "nested/b.txt"})

    def test_python_cache_blocks_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            cache = package / "__pycache__"
            cache.mkdir()
            (cache / "module.pyc").write_bytes(b"cache")
            with self.assertRaises(RuntimeError):
                refresh(package)


if __name__ == "__main__":
    unittest.main()
