#!/usr/bin/env python3
"""Refresh hashes after a release candidate regenerates its derived artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh(package: Path) -> dict[str, object]:
    package = package.resolve()
    manifest = package / "release_candidate_manifest.csv"
    cache_files = [
        path for path in package.rglob("*")
        if path.is_file() and ("__pycache__" in path.parts or path.suffix == ".pyc")
    ]
    if cache_files:
        raise RuntimeError("Python cache files must not enter the release: " + ", ".join(map(str, cache_files)))
    files = sorted(
        path for path in package.rglob("*")
        if path.is_file() and path != manifest
    )
    rows = [
        {
            "relative_path": path.relative_to(package).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "source_class": "author_generated_or_public_derived",
            "manual_license_review": "required",
        }
        for path in files
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"package": str(package), "hashed_files": len(rows), "manifest": str(manifest)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(refresh(args.package))


if __name__ == "__main__":
    main()
