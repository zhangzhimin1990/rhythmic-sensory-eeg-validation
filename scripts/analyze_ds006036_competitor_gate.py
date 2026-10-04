#!/usr/bin/env python3
"""Reproduce the 2026 direct competitor's closed-eye photic-driving analysis.

This is a sensitivity/competition gate, not an exact independent replication.
It uses the dataset-provided SET files and a transparent event reconstruction,
then contrasts closed-eye driving-index results with the existing eyes-open SNR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import welch
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bids_events import PHOTO_RE, contiguous_pulse_segment, first_event_between  # noqa: E402


CORE_FREQUENCIES = (5.0, 10.0, 15.0, 20.0)
POSTERIOR_ROI = ("O1", "O2", "P3", "P4", "Pz", "T5", "T6", "P7", "P8")
GROUP_LABELS = {"A": "AD", "C": "CN", "F": "FTD"}


def driving_index(freqs: np.ndarray, psd: np.ndarray, target_hz: float) -> float:
    """Target-bin PSD divided by median local background, as in the preprint."""
    target = np.abs(freqs - target_hz) <= 0.26
    background = (
        (np.abs(freqs - target_hz) <= 2.0)
        & (np.abs(freqs - target_hz) > 0.5)
    )
    target_power = float(np.mean(psd[target]))
    background_power = float(np.median(psd[background]))
    return target_power / background_power


def spearman_brown(reliability: float) -> float:
    return 2.0 * reliability / (1.0 + reliability) if reliability > -1.0 else np.nan


def closed_eye_blocks(raw: mne.io.BaseRaw, subject: str) -> list[dict]:
    rows = sorted(
        (float(onset), str(description).strip())
        for onset, description in zip(raw.annotations.onset, raw.annotations.description)
    )
    labels = [
        (index, onset, float(match.group(1)))
        for index, (onset, value) in enumerate(rows)
        if (match := PHOTO_RE.match(value))
    ]
    pulses = [onset for onset, value in rows if value == "Photo/HV mark"]
    blocks = []
    for block_index, (_, label_onset, frequency) in enumerate(labels):
        if frequency not in CORE_FREQUENCIES:
            continue
        next_label = labels[block_index + 1][1] if block_index + 1 < len(labels) else float("inf")
        pulse_segment = contiguous_pulse_segment(pulses, label_onset, frequency)
        if not pulse_segment:
            continue
        pulse_stop = min(next_label, pulse_segment[-1] + max(0.1, 1.5 / frequency))
        open_onset = first_event_between(rows, "open eyes", label_onset, pulse_stop)
        closed_onset = (
            first_event_between(rows, "closed eyes", open_onset, pulse_stop)
            if open_onset is not None
            else None
        )
        if closed_onset is None:
            continue
        # The direct competitor used the eyes-closed portion inside each pulse train.
        start = closed_onset + 0.05
        stop = pulse_stop - 0.05
        if stop - start >= 2.0:
            blocks.append(
                {
                    "subject": subject,
                    "block_index": block_index,
                    "frequency_hz": frequency,
                    "start_s": start,
                    "stop_s": stop,
                    "duration_s": stop - start,
                }
            )
    return blocks


def epoch_metrics(
    data: np.ndarray,
    sfreq: float,
    target_hz: float,
    epoch_s: float = 2.0,
    peak_to_peak_limit_uv: float = 150.0,
) -> list[dict]:
    epoch_samples = int(round(epoch_s * sfreq))
    rows = []
    for start in range(0, data.shape[1] - epoch_samples + 1, epoch_samples):
        epoch = data[:, start : start + epoch_samples]
        peak_to_peak_uv = float(np.max(np.ptp(epoch, axis=1)) * 1e6)
        if peak_to_peak_uv > peak_to_peak_limit_uv:
            continue
        freqs, psd = welch(
            epoch,
            fs=sfreq,
            window="hann",
            nperseg=epoch_samples,
            noverlap=0,
            axis=-1,
        )
        # Aggregate PSD within the posterior ROI before taking the ratio. This
        # matches the preprint's ROI-level definition and avoids inflating the
        # ratio by averaging channel-wise denominators.
        roi_psd = np.mean(psd, axis=0)
        roi_di = driving_index(freqs, roi_psd, target_hz)
        rows.append(
            {
                "driving_index": float(roi_di),
                "snr_db": float(10.0 * np.log10(roi_di)),
                "peak_to_peak_uv": peak_to_peak_uv,
            }
        )
    return rows


def fit_group_models(features: pd.DataFrame, outcome: str, condition: str) -> pd.DataFrame:
    rows = []
    for frequency, frequency_data in features.groupby("frequency_hz"):
        data = frequency_data.dropna(subset=[outcome, "age", "gender", "n_epochs"]).copy()
        data["group"] = pd.Categorical(data["group"], categories=["CN", "AD", "FTD"])
        epoch_term = " + n_epochs" if data["n_epochs"].nunique() > 1 else ""
        model = smf.ols(
            f"{outcome} ~ C(group, Treatment(reference='CN')) + age + C(gender){epoch_term}",
            data=data,
        ).fit(cov_type="HC3")
        for group in ("AD", "FTD"):
            term = f"C(group, Treatment(reference='CN'))[T.{group}]"
            rows.append(
                {
                    "condition": condition,
                    "outcome": outcome,
                    "frequency_hz": frequency,
                    "contrast": f"{group}-CN",
                    "n": int(model.nobs),
                    "coefficient": float(model.params[term]),
                    "ci_low": float(model.conf_int().loc[term, 0]),
                    "ci_high": float(model.conf_int().loc[term, 1]),
                    "p_value": float(model.pvalues[term]),
                    "r_squared": float(model.rsquared),
                }
            )
    result = pd.DataFrame(rows)
    result["q_bh"] = np.nan
    for _, indices in result.groupby(["condition", "outcome", "contrast"]).groups.items():
        result.loc[indices, "q_bh"] = multipletests(result.loc[indices, "p_value"], method="fdr_bh")[1]
    return result


def reliability_rows(epoch_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frequency, frequency_data in epoch_frame.groupby("frequency_hz"):
        halves = []
        for subject, subject_data in frequency_data.sort_values("epoch_index").groupby("subject"):
            values = subject_data["driving_index"].to_numpy()
            if len(values) < 2:
                continue
            midpoint = len(values) // 2
            if midpoint == 0 or midpoint == len(values):
                continue
            halves.append((subject, len(values), float(values[:midpoint].mean()), float(values[midpoint:].mean())))
        halves_frame = pd.DataFrame(halves, columns=["subject", "n_epochs", "first_half", "second_half"])
        if len(halves_frame) >= 3:
            r, p = pearsonr(halves_frame["first_half"], halves_frame["second_half"])
        else:
            r, p = np.nan, np.nan
        rows.append(
            {
                "frequency_hz": frequency,
                "n_subjects": len(halves_frame),
                "pearson_r": r,
                "p_value": p,
                "spearman_brown": spearman_brown(r) if np.isfinite(r) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("eyes_open_features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(args.root / "participants.tsv", sep="\t", encoding="utf-8-sig")
    participants.columns = [column.strip() for column in participants.columns]
    participant_rows = participants.set_index("participant_id").to_dict("index")
    epoch_rows = []
    block_rows = []
    recording_rows = []

    for set_path in sorted(args.root.glob("sub-*/eeg/*_eeg.set")):
        subject = set_path.parts[-3]
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        posterior_channels = [channel for channel in POSTERIOR_ROI if channel in raw.ch_names]
        data = raw.get_data(picks=posterior_channels)
        data = data - raw.get_data().mean(axis=0, keepdims=True)
        subject_epoch_count = 0
        for block in closed_eye_blocks(raw, subject):
            start = int(round(block["start_s"] * sfreq))
            stop = int(round(block["stop_s"] * sfreq))
            rows = epoch_metrics(data[:, start:stop], sfreq, block["frequency_hz"])
            for epoch_index, row in enumerate(rows):
                epoch_rows.append({**block, "epoch_index": epoch_index, **row})
            subject_epoch_count += len(rows)
            if rows:
                block_rows.append(
                    {
                        **block,
                        "n_epochs": len(rows),
                        "driving_index": float(np.mean([row["driving_index"] for row in rows])),
                        "snr_db": float(np.mean([row["snr_db"] for row in rows])),
                    }
                )
        recording_rows.append(
            {
                "subject": subject,
                "n_posterior_channels": len(posterior_channels),
                "n_clean_closed_eye_epochs": subject_epoch_count,
            }
        )

    epoch_frame = pd.DataFrame(epoch_rows)
    block_frame = pd.DataFrame(block_rows)
    subject_frequency = (
        block_frame.groupby(["subject", "frequency_hz"], as_index=False)
        .agg(
            n_epochs=("n_epochs", "sum"),
            closed_eye_duration_s=("duration_s", "sum"),
            driving_index=("driving_index", "mean"),
            snr_db=("snr_db", "mean"),
        )
    )
    demographics = []
    for subject, row in participant_rows.items():
        demographics.append(
            {
                "subject": subject,
                "gender": str(row["Gender"]).strip(),
                "age": float(row["Age"]),
                "group": GROUP_LABELS.get(str(row["Group"]).strip(), str(row["Group"]).strip()),
                "mmse": float(row[[column for column in row if column.strip() == "MMSE"][0]]),
            }
        )
    subject_frequency = subject_frequency.merge(pd.DataFrame(demographics), on="subject", validate="many_to_one")
    subject_frequency["log1p_driving_index"] = np.log1p(subject_frequency["driving_index"])

    model_parts = [
        fit_group_models(subject_frequency, "driving_index", "closed_eye"),
        fit_group_models(subject_frequency, "log1p_driving_index", "closed_eye"),
        fit_group_models(subject_frequency, "snr_db", "closed_eye"),
    ]

    eyes_open = pd.read_csv(args.eyes_open_features).merge(
        pd.DataFrame(demographics)[["subject", "gender", "age", "group"]],
        on="subject",
        suffixes=("_source", ""),
        validate="many_to_one",
    )
    eyes_open["n_epochs"] = 1
    model_parts.append(fit_group_models(eyes_open, "local_snr_db", "eyes_open"))
    models = pd.concat(model_parts, ignore_index=True)
    reliability = reliability_rows(epoch_frame)

    epoch_frame.to_csv(args.output / "closed_eye_epoch_features.csv", index=False)
    subject_frequency.to_csv(args.output / "closed_eye_subject_frequency.csv", index=False)
    pd.DataFrame(recording_rows).to_csv(args.output / "recording_qc.csv", index=False)
    reliability.to_csv(args.output / "split_half_reliability.csv", index=False)
    models.to_csv(args.output / "group_models.csv", index=False)

    ten_hz = models[models.frequency_hz == 10.0]
    summary = {
        "n_subjects": int(subject_frequency.subject.nunique()),
        "n_subject_frequency_cells": int(len(subject_frequency)),
        "median_clean_epochs_per_cell": float(subject_frequency.n_epochs.median()),
        "ten_hz_results": ten_hz[
            ["condition", "outcome", "contrast", "coefficient", "ci_low", "ci_high", "p_value", "q_bh"]
        ].to_dict("records"),
        "boundary": "Sensitivity analysis of dataset-provided files; not an exact reproduction of the preprint ICA pipeline.",
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
