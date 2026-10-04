#!/usr/bin/env python3
"""Resume-safe extraction of one deflated member from a remote ZIP archive."""

from __future__ import annotations

import argparse
import binascii
import json
import os
import shutil
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

from scripts.stream_remote_zip import GIB, MIB, HTTPRangeReader, read_url


def locate_member(url: str, member_name: str, timeout: float) -> dict[str, int]:
    reader = HTTPRangeReader(
        url, cache_bytes=8 * MIB, timeout=timeout, max_retries=12,
        retry_backoff_seconds=0.5,
    )
    with zipfile.ZipFile(reader) as archive:
        info = archive.getinfo(member_name)
        if info.is_dir() or info.compress_type != zipfile.ZIP_DEFLATED:
            raise ValueError("member must be a deflated regular file")
        reader.seek(info.header_offset)
        header = reader.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("invalid ZIP local header")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        return {
            "data_offset": info.header_offset + 30 + name_length + extra_length,
            "compressed_bytes": info.compress_size,
            "uncompressed_bytes": info.file_size,
            "crc32": info.CRC,
        }


def fetch_range(url: str, start: int, end: int, timeout: float, retries: int) -> bytes:
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "public-eeg-validation/0.3"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if getattr(response, "status", response.getcode()) != 206:
                    raise RuntimeError("range request did not return HTTP 206")
                body = response.read()
            if len(body) != end - start + 1:
                raise RuntimeError("range response length mismatch")
            return body
        except (TimeoutError, ConnectionError, urllib.error.URLError, RuntimeError):
            if attempt >= retries:
                raise
            time.sleep(min(15.0, 0.5 * (2**attempt)))
    raise AssertionError("unreachable retry loop")


def download_compressed(
    url: str, metadata: dict[str, int], partial: Path, chunk_mib: int,
    timeout: float, retries: int,
) -> None:
    partial.parent.mkdir(parents=True, exist_ok=True)
    current = partial.stat().st_size if partial.exists() else 0
    total = metadata["compressed_bytes"]
    if current > total:
        raise ValueError("compressed partial exceeds expected size")
    with partial.open("ab") as handle:
        while current < total:
            count = min(chunk_mib * MIB, total - current)
            absolute_start = metadata["data_offset"] + current
            body = fetch_range(url, absolute_start, absolute_start + count - 1, timeout, retries)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
            current += len(body)


def inflate_and_verify(compressed: Path, output: Path, metadata: dict[str, int]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()
    decompressor = zlib.decompressobj(-15)
    crc = 0
    size = 0
    with compressed.open("rb") as source, partial.open("xb") as destination:
        for block in iter(lambda: source.read(8 * MIB), b""):
            data = decompressor.decompress(block)
            destination.write(data)
            crc = binascii.crc32(data, crc)
            size += len(data)
        tail = decompressor.flush()
        destination.write(tail)
        crc = binascii.crc32(tail, crc)
        size += len(tail)
    if not decompressor.eof or size != metadata["uncompressed_bytes"] or crc & 0xFFFFFFFF != metadata["crc32"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError("inflated member failed EOF, size, or CRC verification")
    os.replace(partial, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-file", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--minimum-free-gib-after", type=float, default=1.5)
    args = parser.parse_args()
    url = read_url(argparse.Namespace(url_file=args.url_file, url=None))
    metadata = locate_member(url, args.member, args.timeout)
    free = shutil.disk_usage(args.output.parent).free
    if free - metadata["uncompressed_bytes"] < args.minimum_free_gib_after * GIB:
        raise OSError("insufficient safe free space after extraction")
    compressed = args.output.with_name(args.output.name + ".deflate.partial")
    download_compressed(url, metadata, compressed, args.chunk_mib, args.timeout, args.retries)
    inflate_and_verify(compressed, args.output, metadata)
    compressed.unlink()
    print(json.dumps({"member": args.member, **metadata, "status": "verified_and_extracted"}))


if __name__ == "__main__":
    main()
