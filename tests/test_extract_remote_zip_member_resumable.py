import binascii
import tempfile
import unittest
import zlib
from pathlib import Path

from scripts.extract_remote_zip_member_resumable import inflate_and_verify


class ResumableRemoteZipExtractionTests(unittest.TestCase):
    def test_raw_deflate_inflates_with_size_and_crc_verification(self):
        payload = (b"rhythmic-eeg" * 10000) + bytes(range(256))
        compressor = zlib.compressobj(level=6, wbits=-15)
        encoded = compressor.compress(payload) + compressor.flush()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "member.deflate.partial"
            output = root / "member.bdf"
            source.write_bytes(encoded)
            inflate_and_verify(source, output, {
                "uncompressed_bytes": len(payload),
                "crc32": binascii.crc32(payload) & 0xFFFFFFFF,
            })
            self.assertEqual(output.read_bytes(), payload)

    def test_crc_mismatch_removes_inflated_partial(self):
        payload = b"test" * 100
        compressor = zlib.compressobj(wbits=-15)
        encoded = compressor.compress(payload) + compressor.flush()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "member.deflate.partial"
            output = root / "member.bdf"
            source.write_bytes(encoded)
            with self.assertRaisesRegex(RuntimeError, "verification"):
                inflate_and_verify(source, output, {
                    "uncompressed_bytes": len(payload), "crc32": 0,
                })
            self.assertFalse(output.exists())
            self.assertFalse((root / "member.bdf.partial").exists())


if __name__ == "__main__":
    unittest.main()
