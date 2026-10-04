#!/usr/bin/env python3
"""Metadata-only audit of ds007648 and ds006780 before signal inspection."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def annex_size(path: Path) -> int | None:
    if not path.is_symlink():
        return path.stat().st_size if path.exists() else None
    match = re.search(r"SHA256E-s(\d+)--", str(path.readlink()))
    return int(match.group(1)) if match else None


def audit_ds007648(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    participants = pd.read_csv(root / "participants.tsv", sep="\t", keep_default_na=False)
    rows = []
    for events_path in sorted(root.glob("sub-*/eeg/*_events.tsv")):
        subject = events_path.parts[-3]
        events = pd.read_csv(events_path, sep="\t")
        signal_path = next(events_path.parent.glob(f"{subject}_task-CrossModal_eeg.eeg"))
        rows.append(
            {
                "subject": subject,
                "trials": len(events),
                "accuracy": events["Correct"].mean(),
                "median_correct_rt": events.loc[events["Correct"].eq(1), "response_time"].median(),
                "missing_behavior_rows": events[["Correct", "response_time"]].isna().any(axis=1).sum(),
                "signal_bytes": annex_size(signal_path),
            }
        )
    frame = pd.DataFrame(rows).merge(participants, left_on="subject", right_on="participant_id", how="left")
    summary = {
        "participants": int(frame.subject.nunique()),
        "trials": int(frame.trials.sum()),
        "trial_range": [int(frame.trials.min()), int(frame.trials.max())],
        "accuracy_range": [float(frame.accuracy.min()), float(frame.accuracy.max())],
        "behavior_complete": bool((frame.missing_behavior_rows == 0).all()),
        "signal_gib": float(frame.signal_bytes.sum() / 1024**3),
        "age_range": [float(frame.age.min()), float(frame.age.max())],
    }
    return frame, summary


def audit_ds006780(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    participants = pd.read_csv(root / "participants.tsv", sep="\t", keep_default_na=False)
    rows = []
    for signal_path in sorted(root.glob("sub-*/eeg/*task-ASSR*_eeg.bdf")):
        subject = signal_path.parts[-3]
        stem = signal_path.name.removesuffix("_eeg.bdf")
        events_path = signal_path.parent / f"{stem}_events.tsv"
        rows.append(
            {
                "subject": subject,
                "run": re.search(r"run-(\d+)", stem).group(1),
                "signal_bytes": annex_size(signal_path),
                "events_present": events_path.exists(),
                "events": len(pd.read_csv(events_path, sep="\t")) if events_path.exists() else np.nan,
            }
        )
    frame = pd.DataFrame(rows).merge(participants, left_on="subject", right_on="participant_id", how="left")
    subject_frame = frame.drop_duplicates("subject")
    assr_yes = participants.loc[participants.completed_ASSR.eq("yes")]
    summary = {
        "participants_released": int(len(participants)),
        "participants_marked_assr_complete": int(len(assr_yes)),
        "participants_with_assr_signal": int(frame.subject.nunique()),
        "assr_runs": int(len(frame)),
        "runs_with_events": int(frame.events_present.sum()),
        "signal_gib": float(frame.signal_bytes.sum() / 1024**3),
        "groups_with_signal": subject_frame.group.value_counts().to_dict(),
        "fsiq_available_with_signal": int(pd.to_numeric(subject_frame.fsiq, errors="coerce").notna().sum()),
        "srs2_available_with_signal": int(pd.to_numeric(subject_frame.srs2_total_t, errors="coerce").notna().sum()),
        "age_available_with_signal": int(pd.to_numeric(subject_frame.age, errors="coerce").notna().sum()),
    }
    return frame, summary


def main() -> None:
    out = Path("outputs/planning/behavioral_validation_candidates")
    out.mkdir(parents=True, exist_ok=True)
    crossmodal, crossmodal_summary = audit_ds007648(Path("data/public/ds007648/v1.1.0"))
    sfari, sfari_summary = audit_ds006780(Path("data/public/ds006780/v1.0.0"))
    crossmodal.to_csv(out / "ds007648_inventory.csv", index=False)
    sfari.to_csv(out / "ds006780_assr_inventory.csv", index=False)
    summary = {"ds007648": crossmodal_summary, "ds006780": sfari_summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
