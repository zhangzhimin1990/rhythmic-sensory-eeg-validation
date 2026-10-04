#!/usr/bin/env python3
"""Run a non-destructive structural, event, behavior, and lightweight signal audit on one Dryad BDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd


CONDITION_NAMES = {1: "f_theta", 2: "2_hz", 3: "f_theta_plus", 4: "non_rhythmic"}
TARGET_CODES = {11, 12, 21, 22, 31, 32, 41, 42}
CONGRUENT_CODES = {11, 22, 31, 42}
RESPONSE_CODES = {
    11: {50: "incorrect", 51: "correct", 55: "omission"},
    12: {50: "incorrect", 52: "correct", 55: "omission"},
    21: {60: "incorrect", 61: "correct", 65: "omission"},
    22: {60: "incorrect", 62: "correct", 65: "omission"},
    31: {70: "incorrect", 71: "correct", 75: "omission"},
    32: {70: "incorrect", 72: "correct", 75: "omission"},
    41: {80: "incorrect", 81: "correct", 85: "omission"},
    42: {80: "incorrect", 82: "correct", 85: "omission"},
}
ALL_RESPONSE_CODES = set().union(*(mapping.keys() for mapping in RESPONSE_CODES.values()))
ALLOWED_CODES = set(CONDITION_NAMES) | TARGET_CODES | ALL_RESPONSE_CODES
EXTERNAL_CHANNELS = {"UP", "DOWN", "LEFT", "RIGHT", "EXG5", "EXG6", "EXG7", "EXG8"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_trigger(value: int) -> int:
    """Keep the low status byte used by the published BioSemi event map."""
    return int(value) & 0xFF


def collapse_overlapping_status_edges(
    events: np.ndarray, max_transition_samples: int = 2
) -> np.ndarray:
    """Recover semantic markers from one-sample bitwise-OR transition states.

    In the released BDF, a new TTL marker can briefly overlap the preceding
    condition, target, or response marker. BioSemi then records their bitwise
    OR for one sample before the intended marker remains alone. The published
    EEGLAB workflow removes these extra states manually. Here we deterministically
    replace a short transition state with the following allowed marker while
    retaining the onset sample of the transition.
    """
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError("events must have shape (n_events, 3)")
    rows: list[list[int]] = []
    index = 0
    while index < len(events):
        sample, previous, value = (int(item) for item in events[index])
        code = normalize_trigger(value)
        if index + 1 < len(events):
            next_sample, _, next_value = (int(item) for item in events[index + 1])
            next_code = normalize_trigger(next_value)
            if (
                0 < next_sample - sample <= max_transition_samples
                and next_code in ALLOWED_CODES
            ):
                rows.append([sample, normalize_trigger(previous), next_code])
                index += 2
                continue
        if code in ALLOWED_CODES:
            rows.append([sample, normalize_trigger(previous), code])
        index += 1
    return np.asarray(rows, dtype=int).reshape(-1, 3)


def reconstruct_trials(events: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Reconstruct condition-target-response triples without deleting anomalies."""
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError("events must have shape (n_events, 3)")
    rows: list[dict] = []
    current_condition: int | None = None
    condition_sample: int | None = None
    codes = [normalize_trigger(value) for value in events[:, 2]]
    samples = [int(value) for value in events[:, 0]]
    for index, (sample, code) in enumerate(zip(samples, codes)):
        if code in CONDITION_NAMES:
            current_condition = code
            condition_sample = sample
            continue
        if code not in TARGET_CODES:
            continue
        next_code = codes[index + 1] if index + 1 < len(codes) else None
        next_sample = samples[index + 1] if index + 1 < len(samples) else None
        response_class = RESPONSE_CODES[code].get(next_code, "missing_or_unexpected")
        expected_response = next(
            (candidate for candidate, label in RESPONSE_CODES[code].items() if label == "correct"),
            None,
        )
        has_expected_response_family = next_code in RESPONSE_CODES[code]
        response_sample = next_sample if has_expected_response_family else None
        response_code = next_code if has_expected_response_family else None
        response_latency_ms = (
            (response_sample - sample) * 1000.0 / sfreq if response_sample is not None else np.nan
        )
        rt_ms = response_latency_ms if response_class in {"correct", "incorrect"} else np.nan
        stimulation_to_target_ms = (
            (sample - condition_sample) * 1000.0 / sfreq
            if condition_sample is not None
            else np.nan
        )
        rows.append(
            {
                "trial_index": len(rows) + 1,
                "condition_code": current_condition,
                "condition": CONDITION_NAMES.get(current_condition, "missing"),
                "stimulation_sample": condition_sample,
                "target_code": code,
                "target_sample": sample,
                "congruency": "congruent" if code in CONGRUENT_CODES else "incongruent",
                "response_code": response_code,
                "response_sample": response_sample,
                "response_class": response_class,
                "expected_correct_code": expected_response,
                "stimulation_to_target_ms": stimulation_to_target_ms,
                "response_latency_ms": response_latency_ms,
                "rt_ms": rt_ms,
                "rt_plausible_100_2000_ms": bool(np.isfinite(rt_ms) and 100 <= rt_ms <= 2000),
                "sequence_valid": bool(current_condition is not None and has_expected_response_family),
            }
        )
        current_condition = None
        condition_sample = None
    return pd.DataFrame(rows)


def find_status_events(raw: mne.io.BaseRaw) -> tuple[np.ndarray, str, np.ndarray]:
    preferred = [name for name in ("Status", "STI 014") if name in raw.ch_names]
    stim_names = [
        raw.ch_names[index]
        for index in mne.pick_types(raw.info, stim=True, exclude=[])
        if raw.ch_names[index] not in preferred
    ]
    candidates = preferred + stim_names
    if not candidates:
        raise RuntimeError("no BioSemi status/stim channel found")
    errors = []
    for name in candidates:
        try:
            events = mne.find_events(
                raw,
                stim_channel=name,
                shortest_event=1,
                initial_event=True,
                uint_cast=True,
                mask=255,
                mask_type="and",
                consecutive=True,
                output="step",
                verbose="error",
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        if len(events):
            return collapse_overlapping_status_edges(events), name, events
    raise RuntimeError("no events found on candidate status channels: " + "; ".join(errors))


def channel_qc(raw: mne.io.BaseRaw, window_seconds: float = 10.0, n_windows: int = 9) -> pd.DataFrame:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise RuntimeError("no EEG channels found")
    window = max(1, int(round(window_seconds * raw.info["sfreq"])))
    latest_start = max(0, raw.n_times - window)
    starts = np.unique(np.linspace(0, latest_start, n_windows, dtype=int))
    chunks = [raw.get_data(picks=picks, start=int(start), stop=int(start + window)) for start in starts]
    data = np.concatenate(chunks, axis=1) * 1e6
    rows = []
    for channel, values in zip((raw.ch_names[index] for index in picks), data):
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "channel": channel,
                "channel_role": "external" if channel in EXTERNAL_CHANNELS else "scalp_eeg",
                "sampled_seconds": len(finite) / raw.info["sfreq"],
                "finite_fraction": float(np.mean(np.isfinite(values))),
                "median_uv": float(np.median(finite)) if len(finite) else np.nan,
                "mad_uv": float(np.median(np.abs(finite - np.median(finite)))) if len(finite) else np.nan,
                "p01_uv": float(np.quantile(finite, 0.01)) if len(finite) else np.nan,
                "p99_uv": float(np.quantile(finite, 0.99)) if len(finite) else np.nan,
                "std_uv": float(np.std(finite)) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def trial_summary(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    summary = (
        trials.groupby(["condition", "congruency", "response_class"], dropna=False)
        .agg(
            n_trials=("trial_index", "size"),
            median_rt_ms=("rt_ms", "median"),
            plausible_rt_fraction=("rt_plausible_100_2000_ms", "mean"),
            valid_sequence_fraction=("sequence_valid", "mean"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if not args.bdf.is_file():
        raise SystemExit(f"BDF not found: {args.bdf}")
    if args.output_dir.exists():
        if not args.output_dir.is_dir():
            raise SystemExit(f"output path exists and is not a directory: {args.output_dir}")
        if any(args.output_dir.iterdir()):
            raise SystemExit(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_bdf(args.bdf, preload=False, verbose="error")
    events, status_channel, raw_edges = find_status_events(raw)
    trials = reconstruct_trials(events, float(raw.info["sfreq"]))
    normalized_codes = [normalize_trigger(value) for value in events[:, 2]]
    counts = pd.DataFrame(
        sorted(Counter(normalized_codes).items()), columns=["event_code", "count"]
    )
    counts["allowed_by_published_map"] = counts["event_code"].isin(ALLOWED_CODES)
    qc = channel_qc(raw)
    summary = trial_summary(trials)

    trials.to_csv(args.output_dir / "trials.csv", index=False)
    summary.to_csv(args.output_dir / "trial_summary.csv", index=False)
    counts.to_csv(args.output_dir / "event_counts.csv", index=False)
    raw_counts = pd.DataFrame(
        sorted(Counter(normalize_trigger(value) for value in raw_edges[:, 2]).items()),
        columns=["raw_status_code", "count"],
    )
    raw_counts["allowed_without_overlap_decoding"] = raw_counts["raw_status_code"].isin(
        ALLOWED_CODES
    )
    raw_counts.to_csv(args.output_dir / "raw_status_edge_counts.csv", index=False)
    qc.to_csv(args.output_dir / "channel_qc.csv", index=False)
    scalp_channels = [name for name in raw.ch_names if name not in EXTERNAL_CHANNELS | {status_channel}]
    external_channels = [name for name in raw.ch_names if name in EXTERNAL_CHANNELS]
    metadata = {
        "source_name": args.bdf.name,
        "source_sha256": sha256(args.bdf),
        "file_size_bytes": args.bdf.stat().st_size,
        "n_channels": len(raw.ch_names),
        "n_mne_eeg_labeled_channels": int(len(mne.pick_types(raw.info, eeg=True, exclude=[]))),
        "n_scalp_eeg_channels": len(scalp_channels),
        "n_external_channels": len(external_channels),
        "external_channels": external_channels,
        "channel_names": raw.ch_names,
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "n_samples": int(raw.n_times),
        "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
        "status_channel": status_channel,
        "n_raw_status_edges": int(len(raw_edges)),
        "n_semantic_events_after_overlap_decoding": int(len(events)),
        "n_events": int(len(events)),
        "n_trials": int(len(trials)),
        "n_valid_sequences": int(trials["sequence_valid"].sum()) if len(trials) else 0,
        "unexpected_event_codes_after_overlap_decoding": sorted(
            set(normalized_codes) - ALLOWED_CODES
        ),
        "raw_transition_codes_requiring_review": sorted(
            set(normalize_trigger(value) for value in raw_edges[:, 2]) - ALLOWED_CODES
        ),
        "mne_version": mne.__version__,
        "interpretation": "pilot structural/event audit; not a final preprocessing or mechanism analysis",
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
