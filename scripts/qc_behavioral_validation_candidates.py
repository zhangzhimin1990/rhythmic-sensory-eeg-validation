#!/usr/bin/env python3
"""Signal/event/data-quality gate for ds007648 and ds006780 QC samples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import norm


CROSSMODAL = ("03", "04", "10", "13", "14", "23")
SFARI = ("10025", "10764", "11038", "11325", "1501", "1589")


def annex_size(path: Path) -> int | None:
    if not path.is_symlink():
        return path.stat().st_size if path.exists() else None
    match = re.search(r"SHA256E-s(\d+)--", str(path.readlink()))
    return int(match.group(1)) if match else None


def local_snr_db(freq: np.ndarray, psd: np.ndarray, target: float) -> float:
    target_power = psd[np.argmin(np.abs(freq - target))]
    flank = ((freq >= target - 2) & (freq <= target - 0.5)) | (
        (freq >= target + 0.5) & (freq <= target + 2)
    )
    return float(10 * np.log10(target_power / np.mean(psd[flank])))


def epoch_psd(
    raw: mne.io.BaseRaw,
    samples: np.ndarray,
    picks: list[str],
    tmin: float,
    tmax: float,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    sfreq = float(raw.info["sfreq"])
    segments = []
    for sample in samples.astype(int):
        start = sample + int(round(tmin * sfreq))
        stop = sample + int(round(tmax * sfreq))
        if start < 0 or stop > raw.n_times:
            continue
        data = raw.get_data(picks=picks, start=start, stop=stop)
        if np.isfinite(data).all():
            segments.append(data)
    if not segments:
        return np.array([]), np.array([]), np.array([]), 0
    stacked = np.asarray(segments)
    nperseg = min(int(round((tmax - tmin) * sfreq)), int(sfreq))
    freq, psd = welch(
        stacked,
        fs=sfreq,
        window="hamming",
        nperseg=nperseg,
        noverlap=nperseg // 2 if stacked.shape[-1] > nperseg else 0,
        nfft=nfft,
        axis=-1,
    )
    # Preserve two distinct constructs. Mean single-trial power captures both
    # phase-locked and non-phase-locked activity, whereas the spectrum of the
    # trial-averaged waveform isolates the phase-locked (evoked) component.
    total_psd = psd.mean(axis=(0, 1))
    freq_evoked, evoked_psd = welch(
        stacked.mean(axis=0),
        fs=sfreq,
        window="hamming",
        nperseg=nperseg,
        noverlap=nperseg // 2 if stacked.shape[-1] > nperseg else 0,
        nfft=nfft,
        axis=-1,
    )
    assert np.allclose(freq, freq_evoked)
    return freq, total_psd, evoked_psd.mean(axis=0), int(stacked.shape[0])


def audit_crossmodal(metadata: Path, signals: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = []
    spectra = []
    for subject in CROSSMODAL:
        subject_id = f"sub-{subject}"
        stem = f"{subject_id}_task-CrossModal"
        signal_dir = signals / subject_id / "eeg"
        metadata_dir = metadata / subject_id / "eeg"
        eeg_path = signal_dir / f"{stem}_eeg.eeg"
        vhdr_path = signal_dir / f"{stem}_eeg.vhdr"
        events_path = signal_dir / f"{stem}_events.tsv"
        expected = annex_size(metadata_dir / f"{stem}_eeg.eeg")
        complete = eeg_path.exists() and eeg_path.stat().st_size == expected and vhdr_path.exists()
        row: dict[str, object] = {
            "dataset": "ds007648",
            "subject": subject_id,
            "complete": complete,
            "signal_bytes": eeg_path.stat().st_size if eeg_path.exists() else None,
            "expected_bytes": expected,
        }
        if not complete:
            inventory.append(row)
            continue
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="error")
        events = pd.read_csv(events_path, sep="\t")
        max_event_sample = int(events["sample"].max())
        row.update(
            {
                "n_channels": len(raw.ch_names),
                "n_eeg": sum(x == "eeg" for x in raw.get_channel_types()),
                "sfreq": float(raw.info["sfreq"]),
                "n_times": int(raw.n_times),
                "duration_seconds": float(raw.times[-1]),
                "n_trials": len(events),
                "duplicate_onsets": int(events.onset.duplicated().sum()),
                "behavior_missing_rows": int(events[["Correct", "response_time"]].isna().any(axis=1).sum()),
                "invalid_correct_rows": int((~events.Correct.isin([0, 1])).sum()),
                "nonpositive_rt_rows": int((events.response_time <= 0).sum()),
                "max_event_within_signal": bool(max_event_sample < raw.n_times),
                "accuracy": float(events.Correct.mean()),
                "median_correct_rt": float(events.loc[events.Correct.eq(1), "response_time"].median()),
            }
        )
        inventory.append(row)
        for target, picks in [
            (40.0, ["Cz", "FC1", "FC2", "C1", "C2", "FCz"]),
            (36.0, ["POz", "O1", "O2", "Oz"]),
        ]:
            freq, total_psd, evoked_psd, n_epochs = epoch_psd(
                raw, events["sample"].to_numpy(), picks, 0.5, 3.0, 2000
            )
            spectra.append(
                {
                    "dataset": "ds007648",
                    "subject": subject_id,
                    "target_hz": target,
                    "n_epochs": n_epochs,
                    "total_local_snr_db": local_snr_db(freq, total_psd, target),
                    "evoked_local_snr_db": local_snr_db(freq, evoked_psd, target),
                }
            )
    return pd.DataFrame(inventory), pd.DataFrame(spectra)


def reconstruct_oddball(events: pd.DataFrame) -> dict[str, float | int]:
    events = events.sort_values("onset").reset_index(drop=True)
    stimuli = events.loc[events.trial_type.str.contains("Standard|Oddball", na=False)].copy()
    responses = events.loc[events.trial_type.eq("Response_button"), "onset"].to_numpy()
    assigned = np.zeros(len(stimuli), dtype=bool)
    rt = np.full(len(stimuli), np.nan)
    stimulus_onsets = stimuli.onset.to_numpy()
    for response in responses:
        candidates = np.where((stimulus_onsets <= response) & ((response - stimulus_onsets) <= 1.0))[0]
        if len(candidates):
            index = candidates[-1]
            if not assigned[index]:
                assigned[index] = True
                rt[index] = response - stimulus_onsets[index]
    oddball = stimuli.trial_type.str.contains("Oddball").to_numpy()
    standard = stimuli.trial_type.str.contains("Standard").to_numpy()
    hits = int((oddball & assigned).sum())
    misses = int((oddball & ~assigned).sum())
    false_alarms = int((standard & assigned).sum())
    correct_rejections = int((standard & ~assigned).sum())
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_rejections": correct_rejections,
        "dprime": float(norm.ppf(hit_rate) - norm.ppf(fa_rate)),
        "median_hit_rt": float(np.nanmedian(rt[oddball])) if hits else np.nan,
        "unassigned_responses": int(len(responses) - assigned.sum()),
    }


def reconstruct_oddball_by_frequency(events: pd.DataFrame) -> list[dict[str, float | int]]:
    """Reconstruct performance separately for the 40- and 27-Hz standard blocks."""
    events = events.sort_values("onset").reset_index(drop=True)
    stimuli = events.loc[
        events.trial_type.str.match(r"^(27|40)_Hz_(Standard|Oddball)$", na=False)
    ].copy()
    responses = events.loc[events.trial_type.eq("Response_button"), "onset"].to_numpy()
    assigned = np.zeros(len(stimuli), dtype=bool)
    rt = np.full(len(stimuli), np.nan)
    stimulus_onsets = stimuli.onset.to_numpy()
    for response in responses:
        candidates = np.where((stimulus_onsets <= response) & ((response - stimulus_onsets) <= 1.0))[0]
        if len(candidates):
            index = candidates[-1]
            if not assigned[index]:
                assigned[index] = True
                rt[index] = response - stimulus_onsets[index]
    stimuli["assigned"] = assigned
    stimuli["rt"] = rt
    stimuli["standard_frequency_hz"] = np.where(
        stimuli.trial_type.isin(["40_Hz_Standard", "27_Hz_Oddball"]), 40, 27
    )

    rows = []
    for standard_frequency in (40, 27):
        subset = stimuli.loc[stimuli.standard_frequency_hz.eq(standard_frequency)]
        oddball = subset.trial_type.str.contains("Oddball").to_numpy()
        standard = subset.trial_type.str.contains("Standard").to_numpy()
        subset_assigned = subset.assigned.to_numpy()
        hits = int((oddball & subset_assigned).sum())
        misses = int((oddball & ~subset_assigned).sum())
        false_alarms = int((standard & subset_assigned).sum())
        correct_rejections = int((standard & ~subset_assigned).sum())
        hit_rate = (hits + 0.5) / (hits + misses + 1)
        fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
        rows.append(
            {
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_rejections": correct_rejections,
                "dprime": float(norm.ppf(hit_rate) - norm.ppf(fa_rate)),
                "median_hit_rt": float(subset.loc[oddball & subset_assigned, "rt"].median())
                if hits
                else np.nan,
                "unassigned_responses_total": int(len(responses) - assigned.sum()),
                "standard_frequency_hz": standard_frequency,
            }
        )
    return rows


def audit_sfari(metadata: Path, signals: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = []
    spectra = []
    behavior = []
    participants = pd.read_csv(metadata / "participants.tsv", sep="\t", keep_default_na=False)
    group_map = dict(zip(participants.participant_id, participants.group))
    for subject in SFARI:
        subject_id = f"sub-{subject}"
        signal_dir = signals / subject_id / "eeg"
        metadata_dir = metadata / subject_id / "eeg"
        for bdf_path in sorted(signal_dir.glob(f"{subject_id}_task-ASSR_run-*_eeg.bdf")):
            stem = bdf_path.name.removesuffix("_eeg.bdf")
            source_bdf = metadata_dir / bdf_path.name
            expected = annex_size(source_bdf)
            events_path = signal_dir / f"{stem}_events.tsv"
            complete = bdf_path.stat().st_size == expected and events_path.exists()
            row: dict[str, object] = {
                "dataset": "ds006780",
                "subject": subject_id,
                "group": group_map.get(subject_id),
                "run": re.search(r"run-(\d+)", stem).group(1),
                "complete": complete,
                "signal_bytes": bdf_path.stat().st_size,
                "expected_bytes": expected,
                "events_present": events_path.exists(),
            }
            if not complete:
                inventory.append(row)
                continue
            raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="error")
            events = pd.read_csv(events_path, sep="\t")
            row.update(
                {
                    "n_channels": len(raw.ch_names),
                    "n_eeg": sum(x == "eeg" for x in raw.get_channel_types()),
                    "sfreq": float(raw.info["sfreq"]),
                    "n_times": int(raw.n_times),
                    "duration_seconds": float(raw.times[-1]),
                    "n_events": len(events),
                    "duplicate_event_samples": int(events["sample"].duplicated().sum()),
                    "max_event_within_signal": bool(int(events["sample"].max()) < raw.n_times),
                }
            )
            inventory.append(row)
            for beh in reconstruct_oddball_by_frequency(events):
                beh.update(
                    {
                        "dataset": "ds006780",
                        "subject": subject_id,
                        "group": group_map.get(subject_id),
                        "run": row["run"],
                    }
                )
                behavior.append(beh)
            for target in (40.0, 27.0):
                label = f"{int(target)}_Hz_Standard"
                samples = events.loc[events.trial_type.eq(label), "sample"].to_numpy()
                if not len(samples):
                    continue
                freq, total_psd, evoked_psd, n_epochs = epoch_psd(
                    raw, samples, ["FCz", "FC3", "FC4"], 0.2, 0.5, 1024
                )
                spectra.append(
                    {
                        "dataset": "ds006780",
                        "subject": subject_id,
                        "group": group_map.get(subject_id),
                        "run": row["run"],
                        "target_hz": target,
                        "n_epochs": n_epochs,
                        "total_local_snr_db": local_snr_db(freq, total_psd, target),
                        "evoked_local_snr_db": local_snr_db(freq, evoked_psd, target),
                    }
                )
    return pd.DataFrame(inventory), pd.DataFrame(spectra), pd.DataFrame(behavior)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossmodal-metadata", type=Path, default=Path("data/public/ds007648/v1.1.0"))
    parser.add_argument("--sfari-metadata", type=Path, default=Path("data/public/ds006780/v1.0.0"))
    parser.add_argument("--signals", type=Path, default=Path("data/public/behavioral_validation_qc"))
    parser.add_argument("--output", type=Path, default=Path("outputs/planning/behavioral_validation_signal_qc"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cross_inv, cross_spectra = audit_crossmodal(args.crossmodal_metadata, args.signals / "ds007648")
    sfari_inv, sfari_spectra, sfari_behavior = audit_sfari(args.sfari_metadata, args.signals / "ds006780")
    inventory = pd.concat([cross_inv, sfari_inv], ignore_index=True)
    spectra = pd.concat([cross_spectra, sfari_spectra], ignore_index=True)
    inventory.to_csv(args.output / "recording_qc.csv", index=False)
    spectra.to_csv(args.output / "target_frequency_qc.csv", index=False)
    sfari_behavior.to_csv(args.output / "ds006780_behavior_reconstruction_qc.csv", index=False)
    summary = {
        "recordings_found": inventory.groupby("dataset").size().to_dict() if len(inventory) else {},
        "recordings_complete": inventory.loc[inventory.complete].groupby("dataset").size().to_dict() if len(inventory) else {},
        "event_alignment_failures": int((~inventory.loc[inventory.complete, "max_event_within_signal"]).sum()) if len(inventory) else 0,
        "crossmodal_behavior_missing_rows": int(cross_inv.get("behavior_missing_rows", pd.Series(dtype=float)).fillna(0).sum()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
