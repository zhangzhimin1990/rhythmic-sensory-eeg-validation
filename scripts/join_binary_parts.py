#!/usr/bin/env python3
"""Join ordered HTTP range parts and verify the exact expected byte count."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("expected_bytes", type=int)
    parser.add_argument("parts", nargs="+", type=Path)
    args = parser.parse_args()
    with args.destination.open("wb") as output:
        for part in args.parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    actual = args.destination.stat().st_size
    if actual != args.expected_bytes:
        args.destination.unlink(missing_ok=True)
        raise RuntimeError(f"joined size {actual} != expected {args.expected_bytes}")
    for part in args.parts:
        part.unlink()


if __name__ == "__main__":
    main()
