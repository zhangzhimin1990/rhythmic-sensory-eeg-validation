#!/usr/bin/env python3
"""Result-blind amendment to the frozen Dryad confirmatory subject runner.

The frozen v1 runner incorrectly validates the released ``individual_freq``
field against 3--8 Hz.  The actual confirmatory analysis is performed at
``1.33 * individual_freq``.  This wrapper preserves every frozen extraction,
hash, derivation, disk-reserve, and deletion step, changing only the input
guard so that the *analysis frequency* must fall within 3--8 Hz.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from scripts import run_dryad_confirmatory_subject as frozen


AMENDMENT_ID = "dryad_confirmatory_runner_input_guard_v1"
ANALYSIS_FREQUENCY_MULTIPLIER = 1.33
MINIMUM_ANALYSIS_FREQUENCY_HZ = 3.0
MAXIMUM_ANALYSIS_FREQUENCY_HZ = 8.0


def load_subject(
    subject: int, members_path: Path, protocol_path: Path, metadata_path: Path
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if subject not in set(map(int, protocol["confirmatory_subjects"])):
        raise ValueError("subject is not in the locked confirmatory set")
    members = pd.read_csv(members_path)
    extracted = members["member"].astype(str).str.extract(r"(?:^|/)S(\d+)_Stim\.bdf$")[0]
    inventory = members.loc[extracted.notna()].copy()
    inventory["subject"] = extracted.dropna().astype(int).to_numpy()
    if set(inventory["subject"]) != set(range(1, 45)) or inventory["subject"].duplicated().any():
        raise ValueError("stimulus ZIP inventory is not the exact 44-person set")
    source = inventory.loc[inventory["subject"].eq(subject)].iloc[0]
    metadata = pd.read_excel(metadata_path, sheet_name="Dataset")
    participant = metadata.loc[metadata["subject"].eq(subject)]
    if len(participant) != 1:
        raise ValueError("metadata must contain exactly one participant row")
    theta = float(participant.iloc[0]["individual_freq"])
    analysis_frequency = ANALYSIS_FREQUENCY_MULTIPLIER * theta
    if not MINIMUM_ANALYSIS_FREQUENCY_HZ <= analysis_frequency <= MAXIMUM_ANALYSIS_FREQUENCY_HZ:
        raise ValueError("derived confirmatory analysis frequency is outside 3--8 Hz")
    return {
        "subject": subject,
        "expected_uncompressed_bytes": int(source["uncompressed_bytes"]),
        "individual_theta_hz": theta,
    }


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
    frozen.load_subject = load_subject
    try:
        result = frozen.run_subject(
            args.subject, args.members, args.url_file, args.scratch_root,
            args.output_root, args.freeze_dir, args.protocol, args.metadata,
            args.technical_root,
        )
    except Exception as error:
        raise SystemExit(
            f"amended confirmatory run stopped: {error}. Any extracted BDF was retained."
        ) from error
    finally:
        frozen.load_subject = original_loader

    result["post_freeze_amendment_id"] = AMENDMENT_ID
    output_dir = args.output_root / f"S{args.subject}"
    (output_dir / "confirmatory_run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
