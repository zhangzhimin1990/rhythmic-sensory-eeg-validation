#!/usr/bin/env python3
"""Conservative pre-publication audit for a release-candidate directory.

This scanner finds obvious raw EEG files, direct identifiers, credentials, and
local absolute paths. It is a safety net, not an ethics or licensing review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


RAW_EEG_EXTENSIONS = {
    ".edf", ".bdf", ".set", ".fdt", ".vhdr", ".vmrk", ".eeg", ".cnt", ".fif"
}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml",
    ".py", ".r", ".m", ".sh", ".zsh", ".ipynb", ".xml", ".html"
}
MAX_TEXT_BYTES = 20 * 1024 * 1024
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


@dataclass
class Issue:
    severity: str
    check: str
    relative_path: str
    evidence: str
    remediation: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_text(relative_path: str, text: str) -> list[Issue]:
    macos_prefix = "/" + "Users/"
    linux_prefix = "/" + "home/"
    windows_prefix = "C:" + "\\\\Users\\\\"
    # Require plausible field boundaries. This avoids treating decimal output or
    # hexadecimal checksums as phone/identity numbers while still catching
    # standalone identifier-like values in tables and prose.
    mobile_pattern = re.compile(r"(?<![A-Za-z0-9.])1[3-9]\d{9}(?![A-Za-z0-9.])")
    prc_id_pattern = re.compile(
        r"(?<![A-Za-z0-9.])\d{6}(?:18|19|20)\d{2}"
        r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]"
        r"(?![A-Za-z0-9.])"
    )
    checks = [
        ("Critical", "email_identifier", re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I), "Remove or replace personal email addresses."),
        ("Critical", "mainland_mobile_identifier", mobile_pattern, "Remove mobile numbers."),
        ("Critical", "prc_id_identifier", prc_id_pattern, "Remove PRC identity-card-like numbers."),
        ("Critical", "credential", re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"]?[^\s,'\"]{8,}"), "Remove the credential and rotate it if it was real."),
        ("High", "macos_absolute_path", re.compile(re.escape(macos_prefix) + r"[^\s\"']+"), "Replace local absolute paths with relative or configurable paths."),
        ("High", "linux_absolute_path", re.compile(re.escape(linux_prefix) + r"[^\s\"']+"), "Replace local absolute paths with relative or configurable paths."),
        ("High", "windows_absolute_path", re.compile(re.escape(windows_prefix) + r"[^\s\"']+", re.I), "Replace local absolute paths with relative or configurable paths."),
    ]
    issues = []
    for severity, name, pattern, remediation in checks:
        match = pattern.search(text)
        if match:
            excerpt = match.group(0)
            if len(excerpt) > 120:
                excerpt = excerpt[:117] + "..."
            issues.append(Issue(severity, name, relative_path, excerpt, remediation))
    return issues


def audit_release(package: Path, output: Path) -> dict:
    package = package.resolve()
    output = output.resolve()
    if not package.is_dir():
        raise FileNotFoundError(f"Release candidate directory not found: {package}")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    issues: list[Issue] = []
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        try:
            path.resolve().relative_to(output)
            continue
        except ValueError:
            pass
        relative = str(path.relative_to(package))
        suffix = path.suffix.casefold()
        size = path.stat().st_size
        rows.append({
            "relative_path": relative,
            "suffix": suffix,
            "size_bytes": size,
            "sha256": sha256(path),
        })
        if suffix in RAW_EEG_EXTENSIONS:
            issues.append(Issue(
                "Critical", "raw_eeg_file", relative, suffix,
                "Remove the file unless an explicit, documented public-data exception applies."
            ))
        path_issues = scan_text(relative, relative)
        issues.extend(path_issues)
        if suffix in TEXT_EXTENSIONS and size <= MAX_TEXT_BYTES:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                issues.append(Issue(
                    "Medium", "text_decode", relative, "UTF-8 read failed",
                    "Inspect the file manually or convert it to a documented encoding."
                ))
            else:
                issues.extend(scan_text(relative, text))

    with (output / "release_file_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "suffix", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    issue_rows = sorted(
        (asdict(issue) for issue in issues),
        key=lambda row: (SEVERITY_ORDER[row["severity"]], row["relative_path"], row["check"]),
    )
    with (output / "release_issues.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["severity", "check", "relative_path", "evidence", "remediation"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issue_rows)

    counts = {severity: 0 for severity in SEVERITY_ORDER}
    for issue in issues:
        counts[issue.severity] += 1
    summary = {
        "release_candidate": str(package),
        "file_count": len(rows),
        "issue_counts": counts,
        "machine_audit_passed": counts["Critical"] == 0 and counts["High"] == 0,
        "ethics_license_manual_review_required": True,
    }
    (output / "release_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fail-on-high", action="store_true")
    args = parser.parse_args()
    summary = audit_release(args.release_candidate, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_high and not summary["machine_audit_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
