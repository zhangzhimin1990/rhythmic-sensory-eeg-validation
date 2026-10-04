import tempfile
import unittest
from pathlib import Path

from scripts.qc_ds006222_signal import annex_size, normalize_event


class Ds006222SignalQcTests(unittest.TestCase):
    def test_normalize_event_variants(self):
        self.assertEqual(normalize_event("condition 22"), 22)
        self.assertEqual(normalize_event("61442"), 2)
        self.assertEqual(normalize_event("49154"), 2)
        self.assertIsNone(normalize_event("boundary"))

    def test_annex_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = "../../../.git/annex/objects/x/SHA256E-s251002880--abc.fdt/SHA256E-s251002880--abc.fdt"
            link = Path(temporary) / "signal.fdt"
            link.symlink_to(target)
            self.assertEqual(annex_size(link), 251002880)


if __name__ == "__main__":
    unittest.main()
