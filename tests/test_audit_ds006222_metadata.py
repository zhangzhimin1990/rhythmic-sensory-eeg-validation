import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_ds006222_metadata import annex_declared_size, normalize_event_value


class Ds006222MetadataAuditTests(unittest.TestCase):
    def test_biosemi_status_offsets_are_removed(self):
        self.assertEqual(normalize_event_value("61442"), 2)
        self.assertEqual(normalize_event_value("61462"), 22)
        self.assertEqual(normalize_event_value("49175"), 23)

    def test_condition_labels_are_normalized(self):
        self.assertEqual(normalize_event_value("condition 22"), 22)
        self.assertIsNone(normalize_event_value("not-an-event"))

    def test_annex_declared_size_is_parsed_from_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "signal.fdt"
            os.symlink(
                "target/SHA256E-s254689280--abcdef.fdt/SHA256E-s254689280--abcdef.fdt",
                path,
            )
            self.assertEqual(annex_declared_size(path), 254689280)


if __name__ == "__main__":
    unittest.main()
