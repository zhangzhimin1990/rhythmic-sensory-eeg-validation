import io
import http.client
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from scripts.stream_remote_zip import (
    HTTPRangeReader,
    RangeRequestError,
    extract_member,
    safe_source_label,
)


class FakeResponse:
    def __init__(self, body, status, headers, url="https://objects.example/archive.zip"):
        self._stream = io.BytesIO(body)
        self.status = status
        self.headers = headers
        self._url = url

    def getcode(self):
        return self.status

    def geturl(self):
        return self._url

    def read(self, size=-1):
        return self._stream.read(size)

    def close(self):
        self._stream.close()


def range_urlopen(payload, ignore_range=False):
    def opener(request, timeout=None):
        header = request.get_header("Range")
        if ignore_range or not header:
            return FakeResponse(payload, 200, {"Content-Length": str(len(payload))})
        start_text, end_text = header[6:].split("-", 1)
        start = int(start_text)
        end = min(int(end_text), len(payload) - 1)
        body = payload[start : end + 1]
        return FakeResponse(
            body,
            206,
            {
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                "Accept-Ranges": "bytes",
            },
        )

    return opener


class RemoteZipTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive_path = self.root / "fixture.zip"
        with zipfile.ZipFile(self.archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("participant-01/session.bdf", b"BDF" + bytes(range(256)) * 100)
            archive.writestr("metadata/readme.txt", "public fixture")
        self.payload = self.archive_path.read_bytes()

    def tearDown(self):
        self.temporary.cleanup()

    def test_zipfile_lists_and_extracts_over_ranges(self):
        url = "https://objects.example/fixture.zip?temporary=secret"
        with mock.patch(
            "scripts.stream_remote_zip.urllib.request.urlopen",
            side_effect=range_urlopen(self.payload),
        ):
            reader = HTTPRangeReader(url, cache_bytes=64 * 1024)
            with zipfile.ZipFile(reader) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["participant-01/session.bdf", "metadata/readme.txt"],
                )
                output = self.root / "extracted.bdf"
                extract_member(
                    archive,
                    "participant-01/session.bdf",
                    output,
                    max_member_bytes=1024 * 1024,
                    minimum_free_after=0,
                )
        self.assertEqual(output.read_bytes(), b"BDF" + bytes(range(256)) * 100)

    def test_server_that_ignores_range_is_rejected(self):
        with mock.patch(
            "scripts.stream_remote_zip.urllib.request.urlopen",
            side_effect=range_urlopen(self.payload, ignore_range=True),
        ):
            url = "https://objects.example/file.zip"
            with self.assertRaisesRegex(RangeRequestError, "refusing possible full download"):
                HTTPRangeReader(url, cache_bytes=64 * 1024)

    def test_transient_timeout_retries_the_same_range(self):
        calls = []
        successful_open = range_urlopen(self.payload)

        def flaky_open(request, timeout=None):
            calls.append(request.get_header("Range"))
            if len(calls) == 1:
                raise TimeoutError("transient read timeout")
            return successful_open(request, timeout=timeout)

        with mock.patch(
            "scripts.stream_remote_zip.urllib.request.urlopen",
            side_effect=flaky_open,
        ), mock.patch("scripts.stream_remote_zip.time.sleep") as sleep:
            reader = HTTPRangeReader(
                "https://objects.example/file.zip",
                cache_bytes=64 * 1024,
                max_retries=2,
                retry_backoff_seconds=0.01,
            )
        self.assertEqual(reader.size, len(self.payload))
        self.assertEqual(calls, ["bytes=0-0", "bytes=0-0"])
        sleep.assert_called_once_with(0.01)

    def test_incomplete_range_body_is_retried_without_accepting_partial_bytes(self):
        successful_open = range_urlopen(self.payload)
        calls = []

        class IncompleteResponse(FakeResponse):
            def read(self, size=-1):
                raise http.client.IncompleteRead(b"partial", 10)

        def incomplete_then_complete(request, timeout=None):
            calls.append(request.get_header("Range"))
            if len(calls) == 1:
                return IncompleteResponse(
                    b"",
                    206,
                    {"Content-Range": f"bytes 0-0/{len(self.payload)}"},
                )
            return successful_open(request, timeout=timeout)

        with mock.patch(
            "scripts.stream_remote_zip.urllib.request.urlopen",
            side_effect=incomplete_then_complete,
        ), mock.patch("scripts.stream_remote_zip.time.sleep"):
            reader = HTTPRangeReader(
                "https://objects.example/file.zip",
                cache_bytes=64 * 1024,
                max_retries=1,
                retry_backoff_seconds=0,
            )
        self.assertEqual(reader.size, len(self.payload))
        self.assertEqual(calls, ["bytes=0-0", "bytes=0-0"])

    def test_signed_query_is_removed_from_source_label(self):
        label = safe_source_label(
            "https://objects.example/archive.zip?X-Amz-Credential=secret&X-Amz-Signature=token"
        )
        self.assertEqual(label, "https://objects.example/archive.zip")
        self.assertNotIn("secret", label)


if __name__ == "__main__":
    unittest.main()
