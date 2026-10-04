#!/usr/bin/env python3
"""Inspect or extract one member from a remote ZIP using HTTP byte ranges.

The tool is intended for large public archives exposed through temporary signed
object-store URLs.  It never accepts a server that ignores Range requests:
that guard prevents an accidental full-archive download.  Signed query strings
are neither printed nor written to the listing output.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import io
import os
import shutil
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


MIB = 1024**2
GIB = 1024**3


class RangeRequestError(RuntimeError):
    """Raised when a remote object cannot be accessed safely by byte range."""


class HTTPRangeReader(io.RawIOBase):
    """Seekable, block-cached reader backed by strict HTTP Range requests."""

    def __init__(
        self,
        url: str,
        cache_bytes: int = 8 * MIB,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ):
        super().__init__()
        if cache_bytes < 64 * 1024:
            raise ValueError("cache_bytes must be at least 64 KiB")
        self._url = url
        self._cache_bytes = cache_bytes
        self._timeout = timeout
        self._max_retries = int(max_retries)
        self._retry_backoff_seconds = float(retry_backoff_seconds)
        if self._max_retries < 0 or self._retry_backoff_seconds < 0:
            raise ValueError("retry settings must be non-negative")
        self._position = 0
        self._cache_start = 0
        self._cache = b""
        probe, total, _ = self._fetch(0, 0)
        if len(probe) != 1:
            raise RangeRequestError("one-byte range probe returned an unexpected body length")
        self._size = total

    @staticmethod
    def _parse_content_range(value: str | None) -> tuple[int, int, int]:
        if not value or not value.startswith("bytes ") or "/" not in value:
            raise RangeRequestError("missing or invalid Content-Range header")
        span, total_text = value[6:].split("/", 1)
        start_text, end_text = span.split("-", 1)
        try:
            start, end, total = int(start_text), int(end_text), int(total_text)
        except ValueError as exc:
            raise RangeRequestError("non-integer Content-Range header") from exc
        if start < 0 or end < start or total <= end:
            raise RangeRequestError("inconsistent Content-Range header")
        return start, end, total

    def _fetch_once(self, start: int, end: int) -> tuple[bytes, int, str]:
        request = urllib.request.Request(
            self._url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "public-eeg-validation/0.2"},
        )
        response = urllib.request.urlopen(request, timeout=self._timeout)
        try:
            status = getattr(response, "status", response.getcode())
            if status != 206:
                raise RangeRequestError(
                    f"server returned HTTP {status}, not 206; refusing possible full download"
                )
            actual_start, actual_end, total = self._parse_content_range(
                response.headers.get("Content-Range")
            )
            if actual_start != start or actual_end > end:
                raise RangeRequestError("server returned a different byte range than requested")
            body = response.read()
            if len(body) != actual_end - actual_start + 1:
                raise RangeRequestError("range body length does not match Content-Range")
            return body, total, response.geturl()
        finally:
            response.close()

    def _fetch(self, start: int, end: int) -> tuple[bytes, int, str]:
        for attempt in range(self._max_retries + 1):
            try:
                return self._fetch_once(start, end)
            except (
                TimeoutError,
                ConnectionError,
                urllib.error.URLError,
                http.client.IncompleteRead,
            ):
                if attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_backoff_seconds * (2**attempt))
        raise AssertionError("unreachable retry loop")

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self._position + offset
        elif whence == io.SEEK_END:
            new_position = self._size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if new_position < 0:
            raise ValueError("negative seek position")
        self._position = new_position
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self._position >= self._size or size == 0:
            return b""
        if size is None or size < 0:
            size = self._size - self._position
        size = min(size, self._size - self._position)
        output = bytearray()
        while len(output) < size:
            cache_end = self._cache_start + len(self._cache)
            if not (self._cache_start <= self._position < cache_end):
                end = min(
                    self._size - 1,
                    self._position + max(self._cache_bytes, size - len(output)) - 1,
                )
                self._cache, total, _ = self._fetch(self._position, end)
                if total != self._size:
                    raise RangeRequestError("remote object size changed during access")
                self._cache_start = self._position
                cache_end = self._cache_start + len(self._cache)
            take = min(size - len(output), cache_end - self._position)
            cache_offset = self._position - self._cache_start
            output.extend(self._cache[cache_offset : cache_offset + take])
            self._position += take
        return bytes(output)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def safe_source_label(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def read_url(args: argparse.Namespace) -> str:
    if args.url_file:
        value = Path(args.url_file).read_text(encoding="utf-8").strip()
    else:
        value = args.url.strip()
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("a valid HTTP(S) URL is required")
    return value


def write_listing(archive: zipfile.ZipFile, output: Path | None, source: str) -> None:
    rows = []
    for member in archive.infolist():
        rows.append(
            {
                "member": member.filename,
                "is_directory": member.is_dir(),
                "compressed_bytes": member.compress_size,
                "uncompressed_bytes": member.file_size,
                "compression": member.compress_type,
                "crc32": f"{member.CRC:08x}",
                "source_without_query": source,
            }
        )
    fieldnames = list(rows[0]) if rows else [
        "member",
        "is_directory",
        "compressed_bytes",
        "uncompressed_bytes",
        "compression",
        "crc32",
        "source_without_query",
    ]
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_member(
    archive: zipfile.ZipFile,
    member_name: str,
    output: Path,
    max_member_bytes: int,
    minimum_free_after: int,
) -> None:
    member = archive.getinfo(member_name)
    if member.is_dir():
        raise ValueError("selected member is a directory")
    if member.file_size > max_member_bytes:
        raise ValueError(
            f"member requires {member.file_size / GIB:.2f} GiB, above the configured limit"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes - member.file_size < minimum_free_after:
        raise OSError(
            "insufficient safe free space after extraction: "
            f"free={free_bytes / GIB:.2f} GiB, member={member.file_size / GIB:.2f} GiB"
        )
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"refusing to reuse partial output: {partial}")
    try:
        with archive.open(member, "r") as source, partial.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=8 * MIB)
        os.replace(partial, output)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="temporary remote URL; --url-file is safer for signed URLs")
    source.add_argument("--url-file", help="UTF-8 file containing the temporary remote URL")
    parser.add_argument("--list-csv", type=Path, help="write the ZIP member inventory to CSV")
    parser.add_argument("--extract-member", help="exact archive member name to extract")
    parser.add_argument("--output", type=Path, help="output path for --extract-member")
    parser.add_argument("--cache-mib", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-member-gib", type=float, default=2.0)
    parser.add_argument("--minimum-free-gib-after", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.extract_member) != bool(args.output):
        raise SystemExit("--extract-member and --output must be supplied together")
    url = read_url(args)
    reader = HTTPRangeReader(url, cache_bytes=args.cache_mib * MIB, timeout=args.timeout)
    with zipfile.ZipFile(reader) as archive:
        if args.list_csv or not args.extract_member:
            write_listing(archive, args.list_csv, safe_source_label(url))
        if args.extract_member:
            extract_member(
                archive,
                args.extract_member,
                args.output,
                max_member_bytes=int(args.max_member_gib * GIB),
                minimum_free_after=int(args.minimum_free_gib_after * GIB),
            )
    print(
        {
            "archive_bytes": reader.size,
            "listing": str(args.list_csv) if args.list_csv else None,
            "extracted": str(args.output) if args.output else None,
            "source": safe_source_label(url),
        },
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
