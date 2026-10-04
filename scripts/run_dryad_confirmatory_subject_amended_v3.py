#!/usr/bin/env python3
"""Result-blind Dryad runner using resume-safe remote member extraction."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts import run_dryad_confirmatory_subject as frozen
from scripts.run_dryad_confirmatory_subject_amended import load_subject


AMENDMENT_IDS = [
    "dryad_confirmatory_runner_input_guard_v1",
    "dryad_confirmatory_resumable_transport_v1",
]


def value_after(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def resumable_arguments(arguments: list[str]) -> list[str]:
    if "scripts/stream_remote_zip.py" not in arguments:
        return list(arguments)
    return [
        arguments[0], "-m", "scripts.extract_remote_zip_member_resumable",
        "--url-file", value_after(arguments, "--url-file"),
        "--member", value_after(arguments, "--extract-member"),
        "--output", value_after(arguments, "--output"),
        "--chunk-mib", "8", "--timeout", "60", "--retries", "12",
        "--minimum-free-gib-after", value_after(arguments, "--minimum-free-gib-after"),
    ]


def run_command(arguments: list[str]) -> None:
    subprocess.run(resumable_arguments(arguments), cwd=frozen.ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int)
    parser.add_argument("--members", type=Path, default=frozen.DEFAULT_MEMBERS)
    parser.add_argument("--url-file", type=Path, default=frozen.DEFAULT_URL_FILE)
    parser.add_argument("--scratch-root", type=Path, default=frozen.DEFAULT_SCRATCH)
    parser.add_argument("--output-root", type=Path, default=frozen.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--freeze-dir", type=Path, default=frozen.DEFAULT_FREEZE)
    parser.add_argument("--protocol", type=Path, default=frozen.DEFAULT_PROTOCOL)
    parser.add_argument("--metadata", type=Path, default=frozen.DEFAULT_METADATA)
    parser.add_argument("--technical-root", type=Path, default=frozen.DEFAULT_TECHNICAL)
    args = parser.parse_args()
    original_loader, original_runner = frozen.load_subject, frozen.run_command
    frozen.load_subject, frozen.run_command = load_subject, run_command
    try:
        result = frozen.run_subject(
            args.subject, args.members, args.url_file, args.scratch_root,
            args.output_root, args.freeze_dir, args.protocol, args.metadata,
            args.technical_root,
        )
    except Exception as error:
        raise SystemExit(f"amended-v3 confirmatory run stopped: {error}") from error
    finally:
        frozen.load_subject, frozen.run_command = original_loader, original_runner
    result["post_freeze_amendment_ids"] = AMENDMENT_IDS
    result["transport"] = "persistent compressed-member ranges; CRC, size, and BDF SHA-256 verified"
    output_dir = args.output_root / f"S{args.subject}"
    (output_dir / "confirmatory_run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
