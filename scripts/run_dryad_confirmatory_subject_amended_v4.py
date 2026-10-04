#!/usr/bin/env python3
"""Result-blind Dryad runner with resume-safe IncompleteRead retries."""

from __future__ import annotations

import sys

from scripts import run_dryad_confirmatory_subject_amended_v3 as v3


def resumable_arguments_v2(arguments: list[str]) -> list[str]:
    adjusted = v3.resumable_arguments(arguments)
    if "scripts.extract_remote_zip_member_resumable" in adjusted:
        adjusted[adjusted.index("scripts.extract_remote_zip_member_resumable")] = (
            "scripts.extract_remote_zip_member_resumable_v2"
        )
    return adjusted


def run_command(arguments: list[str]) -> None:
    import subprocess
    subprocess.run(resumable_arguments_v2(arguments), cwd=v3.frozen.ROOT, check=True)


v3.run_command = run_command
v3.AMENDMENT_IDS = [
    "dryad_confirmatory_runner_input_guard_v1",
    "dryad_confirmatory_resumable_transport_v1",
    "dryad_confirmatory_incomplete_read_retry_v1",
]


if __name__ == "__main__":
    v3.main()
