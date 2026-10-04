#!/usr/bin/env python3
"""Predefined temporal and non-zero-lag features for ds006036.

The unit of analysis is subject × stimulation frequency. Sensor-level
connectivity is treated as a propagation candidate, not proof of cortical
communication. The script uses only reconstructed eyes-open pulse windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch
from scipy.signal.windows import dpss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_ds006036 import (  # noqa: E402
    CORE_FREQUENCIES,
    blocks_from_annotations,
    local_snr_db,
)


POSTERIOR_ROI = ("P3", "Pz", "P4", "O1", "O2")
FRONTAL_ROI = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def debiased_squared_wpli(imag_cross: np.ndarray) -> float:
    """Debiased squared weighted phase-lag index across observations."""
    values = np.asarray(imag_cross, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.nan
    summed = values.sum()
    summed_sq = np.square(values).sum()
    denominator = np.square(np.abs(values).sum()) - summed_sq
    if denominator <= np.finfo(float).eps:
        return np.nan
    return float((summed * summed - summed_sq) / denominator)


def target_fourier_observations(
    segment: np.ndarray,
    sfreq: float,
    target_hz: float,
    epoch_seconds: float = 1.0,
) -> np.ndarray:
    """Return epoch×taper complex target coefficients for all channels."""
    epoch_samples = int(round(epoch_seconds * sfreq))
    n_epochs = segment.shape[1] // epoch_samples
    if n_epochs < 2:
        return np.empty((0, segment.shape[0]), dtype=complex)
    epochs = segment[:, : n_epochs * epoch_samples].reshape(
        segment.shape[0], n_epochs, epoch_samples
    ).transpose(1, 0, 2)
    epochs = detrend(epochs, axis=-1, type="linear")
    tapers = dpss(epoch_samples, NW=2.0, Kmax=3, sym=False)
    time = np.arange(epoch_samples) / sfreq
    carrier = np.exp(-2j * np.pi * target_hz * time)
    observations = []
    for epoch in epochs:
        for taper in tapers:
            observations.append((epoch * taper) @ carrier)
    return np.asarray(observations)


def pair_connectivity(
    observations: np.ndarray, first: int, second: int
) -> tuple[float, float, float]:
    """Magnitude coherence, absolute imaginary coherence and dwPLI."""
    x = observations[:, first]
    y = observations[:, second]
    cross = x * np.conj(y)
    scale = np.sqrt(np.mean(np.abs(x) ** 2) * np.mean(np.abs(y) ** 2))
    if scale <= np.finfo(float).eps:
        return np.nan, np.nan, np.nan
    normalized_cross = cross / scale
    coherency = np.mean(normalized_cross)
    return (
        float(np.abs(coherency)),
        float(np.abs(np.imag(coherency))),
        debiased_squared_wpli(np.imag(normalized_cross)),
    )


def snr_for_channels(
    data: np.ndarray, sfreq: float, target_hz: float, indices: list[int]
) -> float:
    nperseg = min(int(round(2.0 * sfreq)), data.shape[1])
    freqs, psd = welch(
        data[indices],
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        axis=-1,
    )
    return float(np.mean([local_snr_db(freqs, row, target_hz) for row in psd]))


def connectivity_summary(
    segment: np.ndarray,
    sfreq: float,
    target_hz: float,
    channel_names: list[str],
    first_roi: tuple[str, ...] = POSTERIOR_ROI,
    second_roi: tuple[str, ...] = FRONTAL_ROI,
    epoch_seconds: float = 1.0,
) -> dict[str, float]:
    observations = target_fourier_observations(
        segment, sfreq, target_hz, epoch_seconds=epoch_seconds
    )
    posterior = [channel_names.index(channel) for channel in first_roi]
    frontal = [channel_names.index(channel) for channel in second_roi]
    values = [
        pair_connectivity(observations, posterior_index, frontal_index)
        for posterior_index in posterior
        for frontal_index in frontal
    ]
    array = np.asarray(values, dtype=float)
    return {
        "n_fourier_observations": int(observations.shape[0]),
        "posterior_frontal_coherence": float(np.nanmedian(array[:, 0])),
        "posterior_frontal_abs_imcoh": float(np.nanmedian(array[:, 1])),
        "posterior_frontal_dwpli2": float(np.nanmedian(array[:, 2])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(
        args.root / "participants.tsv", sep="\t", encoding="utf-8-sig"
    )
    participants.columns = [column.strip() for column in participants.columns]
    participant_rows = participants.set_index("participant_id").to_dict("index")
    block_rows, qc_rows, manifests = [], [], []

    for set_path in sorted(args.root.glob("sub-*/eeg/*_eeg.set")):
        sid = set_path.parts[-3]
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        native = raw.get_data()
        average = native - native.mean(axis=0, keepdims=True)
        csd_raw = raw.copy()
        csd_raw.set_montage("standard_1020", on_missing="raise", verbose="ERROR")
        csd = mne.preprocessing.compute_current_source_density(
            csd_raw, lambda2=1e-5, stiffness=4, copy=True, verbose="ERROR"
        ).get_data()
        blocks = [
            block
            for block in blocks_from_annotations(raw, sid)
            if block["frequency_hz"] in CORE_FREQUENCIES
            and block["open_duration_s"] >= 2.5
        ]
        usable = 0
        for block in blocks:
            start = int(round((block["open_onset_s"] + 0.5) * sfreq))
            stop = int(round((block["closed_onset_s"] - 0.1) * sfreq))
            if stop - start < int(round(2.0 * sfreq)):
                continue
            target = float(block["frequency_hz"])
            segment = average[:, start:stop]
            native_segment = native[:, start:stop]
            csd_segment = csd[:, start:stop]
            posterior = [raw.ch_names.index(channel) for channel in POSTERIOR_ROI]
            frontal = [raw.ch_names.index(channel) for channel in FRONTAL_ROI]
            full_posterior_snr = snr_for_channels(segment, sfreq, target, posterior)
            full_frontal_snr = snr_for_channels(segment, sfreq, target, frontal)
            two_seconds = int(round(2.0 * sfreq))
            early_snr = snr_for_channels(
                segment[:, :two_seconds], sfreq, target, posterior
            )
            late_snr = snr_for_channels(
                segment[:, -two_seconds:], sfreq, target, posterior
            )
            average_connectivity = connectivity_summary(
                segment, sfreq, target, raw.ch_names
            )
            native_connectivity = connectivity_summary(
                native_segment, sfreq, target, raw.ch_names
            )
            csd_connectivity = connectivity_summary(
                csd_segment, sfreq, target, raw.ch_names
            )
            p2p = float(np.ptp(segment[posterior], axis=-1).max() * 1e6)
            overlap_s = max(0.0, 4.0 - segment.shape[1] / sfreq)
            participant = participant_rows[sid]
            block_rows.append(
                {
                    "subject": sid,
                    "block_index": block["block_index"],
                    "group": str(participant["Group"]).strip(),
                    "gender": str(participant["Gender"]).strip(),
                    "age": float(participant["Age"]),
                    "mmse": float(participant["MMSE"]),
                    "frequency_hz": target,
                    "analyzed_duration_s": segment.shape[1] / sfreq,
                    "early_late_overlap_s": overlap_s,
                    "posterior_peak_to_peak_uv": p2p,
                    "posterior_snr_db": full_posterior_snr,
                    "frontal_snr_db": full_frontal_snr,
                    "posterior_minus_frontal_snr_db": full_posterior_snr
                    - full_frontal_snr,
                    "early_posterior_snr_db": early_snr,
                    "late_posterior_snr_db": late_snr,
                    "late_minus_early_snr_db": late_snr - early_snr,
                    **average_connectivity,
                    **{
                        f"native_{key}": value
                        for key, value in native_connectivity.items()
                        if key != "n_fourier_observations"
                    },
                    **{
                        f"csd_{key}": value
                        for key, value in csd_connectivity.items()
                        if key != "n_fourier_observations"
                    },
                }
            )
            usable += 1
        qc_rows.append(
            {
                "subject": sid,
                "n_eligible_blocks": len(blocks),
                "n_analyzed_blocks": usable,
                "all_required_channels": all(
                    channel in raw.ch_names
                    for channel in POSTERIOR_ROI + FRONTAL_ROI
                ),
            }
        )
        manifests.append(
            {
                "subject": sid,
                "path": str(set_path),
                "bytes": set_path.stat().st_size,
                "sha256": sha256(set_path),
            }
        )

    blocks = pd.DataFrame(block_rows)
    qc = pd.DataFrame(qc_rows)
    subject_frequency = (
        blocks.groupby(
            ["subject", "group", "gender", "age", "mmse", "frequency_hz"],
            as_index=False,
        )
        .agg(
            n_blocks=("block_index", "nunique"),
            analyzed_duration_s=("analyzed_duration_s", "mean"),
            early_late_overlap_s=("early_late_overlap_s", "mean"),
            posterior_peak_to_peak_uv=("posterior_peak_to_peak_uv", "max"),
            posterior_snr_db=("posterior_snr_db", "mean"),
            frontal_snr_db=("frontal_snr_db", "mean"),
            posterior_minus_frontal_snr_db=(
                "posterior_minus_frontal_snr_db",
                "mean",
            ),
            late_minus_early_snr_db=("late_minus_early_snr_db", "mean"),
            posterior_frontal_coherence=("posterior_frontal_coherence", "mean"),
            posterior_frontal_abs_imcoh=("posterior_frontal_abs_imcoh", "mean"),
            posterior_frontal_dwpli2=("posterior_frontal_dwpli2", "mean"),
            native_posterior_frontal_coherence=(
                "native_posterior_frontal_coherence",
                "mean",
            ),
            native_posterior_frontal_abs_imcoh=(
                "native_posterior_frontal_abs_imcoh",
                "mean",
            ),
            native_posterior_frontal_dwpli2=(
                "native_posterior_frontal_dwpli2",
                "mean",
            ),
            csd_posterior_frontal_coherence=(
                "csd_posterior_frontal_coherence",
                "mean",
            ),
            csd_posterior_frontal_abs_imcoh=(
                "csd_posterior_frontal_abs_imcoh",
                "mean",
            ),
            csd_posterior_frontal_dwpli2=(
                "csd_posterior_frontal_dwpli2",
                "mean",
            ),
        )
    )
    blocks.to_csv(args.output / "block_features.csv", index=False)
    subject_frequency.to_csv(args.output / "subject_frequency_features.csv", index=False)
    qc.to_csv(args.output / "recording_qc.csv", index=False)
    pd.DataFrame(manifests).to_csv(
        args.output / "file_manifest_sha256.csv", index=False
    )
    summary = {
        "n_recordings": int(qc.subject.nunique()),
        "n_subjects_with_eligible_signal_windows": int(
            subject_frequency.subject.nunique()
        ),
        "n_blocks": int(blocks.shape[0]),
        "n_subject_frequency_cells": int(subject_frequency.shape[0]),
        "all_required_channels": bool(qc.all_required_channels.all()),
        "median_fourier_observations": float(
            blocks.n_fourier_observations.median()
        ),
        "median_early_late_overlap_s": float(blocks.early_late_overlap_s.median()),
        "interpretation_boundary": "Sensor-level propagation candidates only; no cortical directionality claim.",
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(
        subject_frequency.groupby("frequency_hz")[
            [
                "posterior_minus_frontal_snr_db",
                "late_minus_early_snr_db",
                "posterior_frontal_abs_imcoh",
                "posterior_frontal_dwpli2",
            ]
        ]
        .median()
        .to_string()
    )


if __name__ == "__main__":
    main()
