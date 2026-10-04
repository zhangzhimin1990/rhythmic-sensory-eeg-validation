#!/usr/bin/env python3
"""Result-blind Dryad runner with corrected frequency guard and resilient transport."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts import run_dryad_confirmatory_subject as frozen
from scripts.run_dryad_confirmatory_subject_amended import load_subject


AMENDMENT_IDS = [
    "dryad_confirmatory_runner_input_guard_v1",
    "dryad_confirmatory_transport_resilience_v1",
]


def resilient_arguments(arguments: list[str]) -> list[str]:
    """Increase range-read cache and timeout without changing extracted bytes."""
    adjusted = list(arguments)
    if "scripts/stream_remote_zip.py" not in adjusted:
        return adjusted
    cache_index = adjusted.index("--cache-mib") + 1
    timeout_index = adjusted.index("--timeout") + 1
    adjusted[cache_index] = "8"
    adjusted[timeout_index] = "60"
    return adjusted


def run_command(arguments: list[str]) -> None:
    subprocess.run(resilient_arguments(arguments), cwd=frozen.ROOT, check=True)


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

    original_loader = frozen.load_subject
    original_runner = frozen.run_command
    frozen.load_subject = load_subject
    frozen.run_command = run_command
    try:
        result = frozen.run_subject(
            args.subject, args.members, args.url_file, args.scratch_root,
            args.output_root, args.freeze_dir, args.protocol, args.metadata,
            args.technical_root,
        )
    except Exception as error:
        raise SystemExit(
            f"amended-v2 confirmatory run stopped: {error}. Any extracted BDF was retained."
        ) from error
    finally:
        frozen.load_subject = original_loader
        frozen.run_command = original_runner

    result["post_freeze_amendment_ids"] = AMENDMENT_IDS
    result["remote_range_cache_mib"] = 8
    result["remote_timeout_seconds"] = 60
    output_dir = args.output_root / f"S{args.subject}"
    (output_dir / "confirmatory_run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
