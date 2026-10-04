#!/usr/bin/env python3
"""Temporal and non-zero-lag foundation gate for ds005048 auditory 40 Hz EEG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ds006036_network import connectivity_summary, sha256  # noqa: E402
from qc_ds005048 import (  # noqa: E402
    NARROW_NOISE_BANDS,
    TARGET_HZ,
    normalize_marker,
    parse_subject,
    psd_for_segment,
    read_participants,
    read_raw,
    snr_db,
)


CENTRAL_TEMPORAL_ROI = ("T7", "T8", "C3", "Cz", "C4")
FRONTAL_ROI = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8")


def roi_snr_db(data: np.ndarray, sfreq: float, indices: list[int]) -> float:
    frequencies, psd = psd_for_segment(data[indices], sfreq)
    return float(
        np.mean(snr_db(psd, frequencies, TARGET_HZ, NARROW_NOISE_BANDS))
    )


def linear_slope(values: list[float]) -> float:
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    participants = read_participants(args.root)
    participant_rows = participants.set_index("participant_id").to_dict("index")
    block_rows, recording_rows, manifests = [], [], []

    for set_path in sorted(args.root.glob("sub-*/eeg/*_eeg.set")):
        sid = parse_subject(set_path)
        raw = read_raw(set_path, preload=True)
        events = pd.read_csv(
            set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv")),
            sep="\t",
        )
        stimulus = events[
            events["value"].map(normalize_marker).eq(2)
            & (events["duration"].astype(float) >= 39.0)
        ].head(6)
        sfreq = float(raw.info["sfreq"])
        native = raw.get_data()
        average = native - native.mean(axis=0, keepdims=True)
        csd_raw = raw.copy()
        csd_raw.set_montage("standard_1020", on_missing="raise", verbose="ERROR")
        csd = mne.preprocessing.compute_current_source_density(
            csd_raw,
            lambda2=1e-5,
            stiffness=4,
            copy=True,
            verbose="ERROR",
        ).get_data()
        central = [raw.ch_names.index(channel) for channel in CENTRAL_TEMPORAL_ROI]
        frontal = [raw.ch_names.index(channel) for channel in FRONTAL_ROI]
        participant = participant_rows[sid]

        for block_number, row in enumerate(stimulus.itertuples(index=False), start=1):
            # Analyze seconds 1–39 of each 40 s block to reduce onset/offset edge effects.
            start = int(round((float(row.onset) + 1.0) * sfreq))
            stop = int(round((float(row.onset) + float(row.duration) - 1.0) * sfreq))
            segment = average[:, start:stop]
            native_segment = native[:, start:stop]
            csd_segment = csd[:, start:stop]
            if segment.shape[1] < int(36 * sfreq):
                continue

            boundaries = np.linspace(0, segment.shape[1], 5, dtype=int)
            quartile_snr = [
                roi_snr_db(segment[:, boundaries[i] : boundaries[i + 1]], sfreq, central)
                for i in range(4)
            ]
            average_connectivity = connectivity_summary(
                segment,
                sfreq,
                TARGET_HZ,
                raw.ch_names,
                first_roi=CENTRAL_TEMPORAL_ROI,
                second_roi=FRONTAL_ROI,
                epoch_seconds=2.0,
            )
            native_connectivity = connectivity_summary(
                native_segment,
                sfreq,
                TARGET_HZ,
                raw.ch_names,
                first_roi=CENTRAL_TEMPORAL_ROI,
                second_roi=FRONTAL_ROI,
                epoch_seconds=2.0,
            )
            csd_connectivity = connectivity_summary(
                csd_segment,
                sfreq,
                TARGET_HZ,
                raw.ch_names,
                first_roi=CENTRAL_TEMPORAL_ROI,
                second_roi=FRONTAL_ROI,
                epoch_seconds=2.0,
            )
            central_snr = roi_snr_db(segment, sfreq, central)
            frontal_snr = roi_snr_db(segment, sfreq, frontal)
            block_rows.append(
                {
                    "participant_id": sid,
                    "sex": participant["sex"],
                    "age": participant["age"],
                    "group": participant["group"],
                    "mmse": participant["mmse"],
                    "block_index": block_number,
                    "analyzed_duration_s": segment.shape[1] / sfreq,
                    "central_peak_to_peak_uv": float(
                        np.ptp(segment[central], axis=-1).max() * 1e6
                    ),
                    "central_snr_db": central_snr,
                    "frontal_snr_db": frontal_snr,
                    "central_minus_frontal_snr_db": central_snr - frontal_snr,
                    "q1_snr_db": quartile_snr[0],
                    "q2_snr_db": quartile_snr[1],
                    "q3_snr_db": quartile_snr[2],
                    "q4_snr_db": quartile_snr[3],
                    "within_block_slope_db_per_quartile": linear_slope(quartile_snr),
                    "q4_minus_q1_snr_db": quartile_snr[3] - quartile_snr[0],
                    **{
                        key.replace("posterior_frontal", "central_frontal"): value
                        for key, value in average_connectivity.items()
                    },
                    **{
                        f"native_{key.replace('posterior_frontal', 'central_frontal')}": value
                        for key, value in native_connectivity.items()
                        if key != "n_fourier_observations"
                    },
                    **{
                        f"csd_{key.replace('posterior_frontal', 'central_frontal')}": value
                        for key, value in csd_connectivity.items()
                        if key != "n_fourier_observations"
                    },
                }
            )

        recording_rows.append(
            {
                "participant_id": sid,
                "n_first_six_stimulus_blocks": len(stimulus),
                "all_required_channels": all(
                    channel in raw.ch_names
                    for channel in CENTRAL_TEMPORAL_ROI + FRONTAL_ROI
                ),
            }
        )
        manifests.append(
            {
                "participant_id": sid,
                "path": str(set_path),
                "bytes_set": set_path.stat().st_size,
                "sha256_set": sha256(set_path),
                "sha256_fdt": sha256(set_path.with_suffix(".fdt")),
            }
        )

    blocks = pd.DataFrame(block_rows)
    subject_rows = []
    split_rows = []
    feature_columns = [
        "central_snr_db",
        "central_minus_frontal_snr_db",
        "within_block_slope_db_per_quartile",
        "q4_minus_q1_snr_db",
        "central_frontal_abs_imcoh",
        "central_frontal_dwpli2",
        "native_central_frontal_abs_imcoh",
        "native_central_frontal_dwpli2",
        "csd_central_frontal_abs_imcoh",
        "csd_central_frontal_dwpli2",
    ]
    for sid, data in blocks.groupby("participant_id"):
        data = data.sort_values("block_index")
        metadata = data.iloc[0]
        row = {
            "participant_id": sid,
            "sex": metadata.sex,
            "age": metadata.age,
            "group": metadata.group,
            "mmse": metadata.mmse,
            "n_blocks": data.block_index.nunique(),
            "block_adaptation_slope_db_per_block": linear_slope(
                data.central_snr_db.tolist()
            ),
            "max_central_peak_to_peak_uv": data.central_peak_to_peak_uv.max(),
        }
        row.update({column: data[column].mean() for column in feature_columns})
        subject_rows.append(row)
        for half_name, subset in (("first3", data.head(3)), ("last3", data.tail(3))):
            split = {"participant_id": sid, "half": half_name}
            split.update({column: subset[column].mean() for column in feature_columns})
            split_rows.append(split)

    subjects = pd.DataFrame(subject_rows)
    split = pd.DataFrame(split_rows).pivot(index="participant_id", columns="half")
    reliability_rows = []
    for feature in feature_columns:
        first = split[(feature, "first3")]
        last = split[(feature, "last3")]
        result = spearmanr(first, last, nan_policy="omit")
        reliability_rows.append(
            {
                "feature": feature,
                "n": int(pd.concat([first, last], axis=1).dropna().shape[0]),
                "split_half_spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )

    blocks.to_csv(args.output / "block_features.csv", index=False)
    subjects.to_csv(args.output / "subject_features.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(
        args.output / "split_half_reliability.csv", index=False
    )
    pd.DataFrame(recording_rows).to_csv(args.output / "recording_qc.csv", index=False)
    pd.DataFrame(manifests).to_csv(args.output / "file_manifest_sha256.csv", index=False)
    summary = {
        "n_subjects": int(subjects.participant_id.nunique()),
        "n_blocks": int(blocks.shape[0]),
        "all_have_six_blocks": bool((subjects.n_blocks == 6).all()),
        "median_fourier_observations_per_block": float(
            blocks.n_fourier_observations.median()
        ),
        "median_analyzed_duration_s": float(blocks.analyzed_duration_s.median()),
        "interpretation_boundary": "Sensor-level temporal and non-zero-lag candidates; no source-level propagation claim.",
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(reliability_rows).to_string(index=False))


if __name__ == "__main__":
    main()
