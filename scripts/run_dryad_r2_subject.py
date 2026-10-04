#!/usr/bin/env python3
"""Safely extract and analyse one frozen Dryad R2 participant at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    ROOT / "outputs/qc/dryad_raw_stream/R2_selection_frozen_v1/r2_selected_subjects.csv"
)
DEFAULT_URL_FILE = Path("/tmp/dryad_stim_signed_url.txt")
DEFAULT_SCRATCH = Path("/tmp/dryad_r2_streaming")
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/qc/dryad_raw_stream/R2_subjects"
REMAINING_R2_SUBJECTS = (13, 27, 23, 37)
MINIMUM_SCRATCH_RESERVE_BYTES = 64 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_subject(selection_path: Path, subject: int) -> dict[str, object]:
    selection = pd.read_csv(selection_path).sort_values("selection_order")
    observed_order = selection["subject"].astype(int).tolist()
    if observed_order != [2, 13, 27, 23, 37]:
        raise ValueError(f"selection file is not the frozen R2 order: {observed_order}")
    if subject not in REMAINING_R2_SUBJECTS:
        raise ValueError(
            f"subject must be one of the remaining frozen R2 participants: "
            f"{REMAINING_R2_SUBJECTS}"
        )
    row = selection.loc[selection["subject"].eq(subject)]
    if len(row) != 1:
        raise ValueError(f"expected exactly one selection row for S{subject}")
    values = row.iloc[0]
    return {
        "subject": int(subject),
        "individual_theta_hz": float(values["individual_freq"]),
        "expected_uncompressed_bytes": int(values["uncompressed_bytes"]),
        "selection_order": int(values["selection_order"]),
    }


def require_fresh_output(base: Path) -> tuple[Path, Path]:
    audit_dir = base / "audit"
    signal_dir = base / "signal"
    for directory in (audit_dir, signal_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"refusing non-empty output directory: {directory}")
    return audit_dir, signal_dir


def require_scratch_capacity(
    scratch_root: Path,
    expected_bdf_bytes: int,
    reserve_bytes: int = MINIMUM_SCRATCH_RESERVE_BYTES,
) -> dict[str, int]:
    """Fail before extraction unless the BDF plus a fixed reserve fits."""
    usage = shutil.disk_usage(scratch_root)
    required = int(expected_bdf_bytes) + int(reserve_bytes)
    if usage.free < required:
        raise OSError(
            "insufficient scratch space; "
            f"free={usage.free}, required={required}, "
            f"expected_bdf={expected_bdf_bytes}, reserve={reserve_bytes}"
        )
    return {
        "scratch_free_bytes_before_extraction": int(usage.free),
        "scratch_required_bytes": required,
        "scratch_reserve_bytes": int(reserve_bytes),
    }


def run_command(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def verify_outputs(
    subject: int,
    expected_bytes: int,
    bdf: Path,
    audit_dir: Path,
    signal_dir: Path,
) -> dict[str, object]:
    required = [
        audit_dir / "audit_summary.json",
        audit_dir / "trials.csv",
        signal_dir / "qc_summary.json",
        signal_dir / "condition_target_engagement.csv",
        signal_dir / "trial_signal_qc.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"S{subject}: missing outputs: {missing}")
    with (audit_dir / "audit_summary.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    with (signal_dir / "qc_summary.json").open(encoding="utf-8") as handle:
        signal = json.load(handle)
    if int(audit["file_size_bytes"]) != expected_bytes or bdf.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"S{subject}: BDF size mismatch; expected={expected_bytes}, "
            f"audit={audit['file_size_bytes']}, local={bdf.stat().st_size}"
        )
    local_hash = sha256(bdf)
    if audit["source_sha256"] != local_hash:
        raise RuntimeError(f"S{subject}: audit/local BDF SHA-256 mismatch")
    if int(audit["n_scalp_eeg_channels"]) != 64:
        raise RuntimeError(f"S{subject}: expected 64 scalp channels")
    if int(signal["n_trials"]) != int(audit["n_trials"]):
        raise RuntimeError(f"S{subject}: audit/signal trial-count mismatch")
    return {
        "source_bdf_sha256": local_hash,
        "source_bdf_size_bytes": expected_bytes,
        "n_trials": int(audit["n_trials"]),
        "n_valid_sequences": int(audit["n_valid_sequences"]),
        "n_bad_channels": int(signal["n_bad_channels"]),
        "n_usable_trials": int(signal["n_usable_trials"]),
    }


def verify_retained_bdf_and_audit(
    subject: int,
    expected_bytes: int,
    bdf: Path,
    audit_dir: Path,
) -> None:
    """Authorize a resume only for the exact BDF already covered by the audit."""
    required = [audit_dir / "audit_summary.json", audit_dir / "trials.csv"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"S{subject}: retained audit is incomplete: {missing}")
    if not bdf.is_file() or bdf.stat().st_size != expected_bytes:
        observed = bdf.stat().st_size if bdf.is_file() else None
        raise RuntimeError(
            f"S{subject}: retained BDF size mismatch; expected={expected_bytes}, "
            f"observed={observed}"
        )
    with (audit_dir / "audit_summary.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if int(audit["file_size_bytes"]) != expected_bytes:
        raise RuntimeError(f"S{subject}: retained audit size does not match freeze")
    if audit["source_sha256"] != sha256(bdf):
        raise RuntimeError(f"S{subject}: retained BDF SHA-256 does not match audit")
    if int(audit["n_scalp_eeg_channels"]) != 64:
        raise RuntimeError(f"S{subject}: retained audit does not contain 64 scalp channels")


def run_subject(
    subject: int,
    selection_path: Path,
    url_file: Path,
    scratch_root: Path,
    output_root: Path,
    resume_retained_bdf: bool = False,
) -> dict[str, object]:
    frozen = load_frozen_subject(selection_path, subject)
    if not url_file.is_file():
        raise FileNotFoundError(f"temporary signed URL file not found: {url_file}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    capacity = require_scratch_capacity(
        scratch_root, int(frozen["expected_uncompressed_bytes"])
    )
    output_base = output_root / f"S{subject}"
    audit_dir = output_base / "audit"
    signal_dir = output_base / "signal"
    bdf = scratch_root / f"S{subject}_Stim.bdf"
    if resume_retained_bdf:
        if signal_dir.exists() and any(signal_dir.iterdir()):
            raise FileExistsError(f"refusing non-empty signal directory: {signal_dir}")
        verify_retained_bdf_and_audit(
            subject,
            int(frozen["expected_uncompressed_bytes"]),
            bdf,
            audit_dir,
        )
    else:
        audit_dir, signal_dir = require_fresh_output(output_base)
        if bdf.exists():
            raise FileExistsError(f"refusing to overwrite scratch BDF: {bdf}")
        member = f"Raw data stim/S{subject}_Stim.bdf"
        run_command(
            [
                sys.executable,
                "scripts/stream_remote_zip.py",
                "--url-file",
                str(url_file),
                "--extract-member",
                member,
                "--output",
                str(bdf),
            ]
        )
        if bdf.stat().st_size != frozen["expected_uncompressed_bytes"]:
            raise RuntimeError(
                f"S{subject}: extracted size mismatch; expected="
                f"{frozen['expected_uncompressed_bytes']}, observed={bdf.stat().st_size}"
            )
        run_command(
            [sys.executable, "scripts/audit_dryad_bdf_pilot.py", str(bdf), str(audit_dir)]
        )
    run_command(
        [
            sys.executable,
            "scripts/qc_dryad_signal_pilot.py",
            str(bdf),
            str(audit_dir / "trials.csv"),
            str(signal_dir),
            "--individual-theta-hz",
            str(frozen["individual_theta_hz"]),
        ]
    )
    verified = verify_outputs(
        subject,
        int(frozen["expected_uncompressed_bytes"]),
        bdf,
        audit_dir,
        signal_dir,
    )
    # The exact target is fixed above, its size and hash have been verified, and
    # all required derived outputs now exist. No directory or wildcard is removed.
    bdf.unlink()
    summary = {
        **frozen,
        **capacity,
        **verified,
        "selection_file_sha256": sha256(selection_path),
        "temporary_bdf_deleted_after_verification": True,
        "resumed_from_verified_retained_bdf": bool(resume_retained_bdf),
        "status": "complete",
    }
    output_base.mkdir(parents=True, exist_ok=True)
    with (output_base / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=int)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--url-file", type=Path, default=DEFAULT_URL_FILE)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume-retained-bdf",
        action="store_true",
        help="resume only after the retained BDF matches the completed audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = run_subject(
            args.subject,
            args.selection,
            args.url_file,
            args.scratch_root,
            args.output_root,
            args.resume_retained_bdf,
        )
    except Exception as error:
        raise SystemExit(
            f"R2 subject run stopped: {error}. Any extracted BDF was retained for inspection."
        ) from error
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
