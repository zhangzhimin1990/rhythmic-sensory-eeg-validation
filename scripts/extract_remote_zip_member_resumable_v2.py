#!/usr/bin/env python3
"""Resume-safe ZIP member extraction with automatic IncompleteRead retries."""

from __future__ import annotations

import http.client
import time
import urllib.error
import urllib.request

from scripts import extract_remote_zip_member_resumable as base


RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    urllib.error.URLError,
    RuntimeError,
    http.client.IncompleteRead,
)


def fetch_range(url: str, start: int, end: int, timeout: float, retries: int) -> bytes:
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "public-eeg-validation/0.4"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if getattr(response, "status", response.getcode()) != 206:
                    raise RuntimeError("range request did not return HTTP 206")
                body = response.read()
            if len(body) != end - start + 1:
                raise RuntimeError("range response length mismatch")
            return body
        except RETRYABLE_EXCEPTIONS:
            if attempt >= retries:
                raise
            time.sleep(min(15.0, 0.5 * (2**attempt)))
    raise AssertionError("unreachable retry loop")


base.fetch_range = fetch_range


if __name__ == "__main__":
    base.main()
