#!/usr/bin/env python3
"""Estimate behavioral split-half reliability in the two public validity cohorts.

The split is stratified within subject and trial type so that odd/even halves retain
the experimental composition.  Reliability is evaluated across subjects and then
Spearman--Brown corrected to the full task length.  This analysis addresses a
specific validity threat: weak EEG--behavior correspondence can be interpreted
only after the behavioral outcome's own measurement reliability is known.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

COUNT_COLUMNS = ("hits", "misses", "false_alarms", "correct_rejections")


def assign_responses(events: pd.DataFrame, response_window: float = 1.0) -> pd.DataFrame:
    """Reproduce the frozen response-assignment rule without loading EEG dependencies."""
    stimuli = events.loc[
        events.trial_type.str.match(r"^(27|40)_Hz_(Standard|Oddball)$", na=False)
    ].copy()
    stimuli = stimuli.sort_values("onset").reset_index(drop=True)
    responses = events.loc[events.trial_type.eq("Response_button"), "onset"].to_numpy()
    assigned = np.zeros(len(stimuli), dtype=bool)
    reaction_time = np.full(len(stimuli), np.nan)
    onsets = stimuli.onset.to_numpy()
    for response in responses:
        candidate = np.where((onsets <= response) & ((response - onsets) <= response_window))[0]
        if len(candidate):
            index = candidate[-1]
            if not assigned[index]:
                assigned[index] = True
                reaction_time[index] = response - onsets[index]
    stimuli["response_assigned"] = assigned
    stimuli["reaction_time"] = reaction_time
    # The label is the standard frequency defining each oddball block; therefore
    # 27-Hz oddballs belong to the 40-Hz-standard block and vice versa.
    stimuli["standard_frequency_hz"] = np.where(
        stimuli.trial_type.isin(["40_Hz_Standard", "27_Hz_Oddball"]), 40, 27
    )
    stimuli["is_oddball"] = stimuli.trial_type.str.contains("Oddball")
    return stimuli


def corrected_dprime(hits: int, misses: int, false_alarms: int, correct_rejections: int) -> float:
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    false_alarm_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    return float(norm.ppf(hit_rate) - norm.ppf(false_alarm_rate))


def spearman_brown(rho: float) -> float:
    if not np.isfinite(rho) or np.isclose(1 + rho, 0):
        return np.nan
    return float(2 * rho / (1 + rho))


def reliability_row(
    wide: pd.DataFrame,
    cohort: str,
    measure: str,
    frequency_hz: float | None,
    seed: int,
    n_boot: int,
) -> dict[str, object]:
    used = wide[[0, 1]].dropna()
    rho = float(spearmanr(used[0], used[1]).statistic)
    values = used.to_numpy()
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        candidate = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        corrected = spearman_brown(candidate)
        if np.isfinite(corrected):
            boot.append(corrected)
    return {
        "cohort": cohort,
        "measure": measure,
        "frequency_hz": frequency_hz,
        "n_subjects": int(len(used)),
        "odd_even_spearman_rho": rho,
        "spearman_brown_full_length": spearman_brown(rho),
        "sb_bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "sb_bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "split_rule": "alternating trials within subject and trial type",
    }


def ds007648_halves(trials: pd.DataFrame) -> pd.DataFrame:
    used = trials.loc[trials.eeg_valid.fillna(False).astype(bool)].copy()
    used = used.sort_values(["subject", "trial_type", "trial_index"])
    used["half"] = used.groupby(["subject", "trial_type"]).cumcount().mod(2)
    rows = []
    for (subject, half), group in used.groupby(["subject", "half"]):
        correct_rt = group.loc[group.Correct.eq(1) & group.response_time.gt(0), "response_time"]
        rows.append(
            {
                "subject": subject,
                "half": int(half),
                "n_trials": int(len(group)),
                "accuracy": float(group.Correct.mean()),
                "median_correct_rt": float(correct_rt.median()),
            }
        )
    return pd.DataFrame(rows)


def locate_events(root: Path, subject: str, run: object) -> Path:
    run_number = int(run)
    candidate = root / subject / "eeg" / f"{subject}_task-ASSR_run-{run_number:02d}_events.tsv"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def split_ds006780_events(events: pd.DataFrame) -> pd.DataFrame:
    stimuli = assign_responses(events)
    stimuli["is_oddball"] = stimuli.trial_type.str.contains("Oddball", regex=False)
    stimuli = stimuli.sort_values("onset")
    stimuli["half"] = stimuli.groupby(["standard_frequency_hz", "is_oddball"]).cumcount().mod(2)
    rows = []
    for (frequency, half), group in stimuli.groupby(["standard_frequency_hz", "half"]):
        oddball = group.is_oddball.to_numpy()
        response = group.response_assigned.to_numpy()
        rows.append(
            {
                "frequency_hz": int(frequency),
                "half": int(half),
                "hits": int((oddball & response).sum()),
                "misses": int((oddball & ~response).sum()),
                "false_alarms": int((~oddball & response).sum()),
                "correct_rejections": int((~oddball & ~response).sum()),
            }
        )
    return pd.DataFrame(rows)


def ds006780_halves(events_root: Path, runs: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    usable = set(subjects.loc[subjects.technical_usable.astype(bool), "subject"])
    run_rows = []
    for subject, run in runs[["subject", "run"]].drop_duplicates().itertuples(index=False):
        if subject not in usable:
            continue
        events = pd.read_csv(locate_events(events_root, subject, run), sep="\t")
        split = split_ds006780_events(events)
        split.insert(0, "run", int(run))
        split.insert(0, "subject", subject)
        run_rows.append(split)
    all_runs = pd.concat(run_rows, ignore_index=True)
    halves = all_runs.groupby(["subject", "frequency_hz", "half"], as_index=False)[list(COUNT_COLUMNS)].sum()
    halves["dprime"] = [
        corrected_dprime(*row) for row in halves[list(COUNT_COLUMNS)].itertuples(index=False, name=None)
    ]

    # Confirm that splitting preserves the exact full-task behavioral counts used
    # in the primary analysis.
    totals = halves.groupby(["subject", "frequency_hz"], as_index=False)[list(COUNT_COLUMNS)].sum()
    for frequency in (27, 40):
        expected = subjects.loc[subjects.subject.isin(usable), ["subject"] + [f"{c}_{frequency}" for c in COUNT_COLUMNS]].copy()
        expected["frequency_hz"] = frequency
        expected = expected.rename(columns={f"{c}_{frequency}": c for c in COUNT_COLUMNS})
        check = totals.loc[totals.frequency_hz.eq(frequency)].merge(
            expected, on=["subject", "frequency_hz"], suffixes=("_split", "_primary"), validate="one_to_one"
        )
        if len(check) != len(expected):
            raise RuntimeError(f"ds006780 {frequency} Hz subject count mismatch")
        mismatched = [c for c in COUNT_COLUMNS if not check[f"{c}_split"].eq(check[f"{c}_primary"]).all()]
        if mismatched:
            raise RuntimeError(f"ds006780 {frequency} Hz count mismatch: {mismatched}")
    return halves


def build_reliability(ds7: pd.DataFrame, ds6: pd.DataFrame, seed: int, n_boot: int) -> pd.DataFrame:
    rows = []
    for offset, measure in enumerate(("accuracy", "median_correct_rt")):
        wide = ds7.pivot(index="subject", columns="half", values=measure)
        rows.append(reliability_row(wide, "ds007648", measure, None, seed + offset, n_boot))
    for offset, frequency in enumerate((27, 40), start=10):
        used = ds6.loc[ds6.frequency_hz.eq(frequency)]
        wide = used.pivot(index="subject", columns="half", values="dprime")
        rows.append(reliability_row(wide, "ds006780", "dprime", frequency, seed + offset, n_boot))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds007648-trials", type=Path, default=Path("outputs/derived/ds007648_trial_features/all_trial_features.csv"))
    parser.add_argument("--ds006780-events-root", type=Path, default=Path("data/public/ds006780/v1.0.0"))
    parser.add_argument("--ds006780-runs", type=Path, default=Path("outputs/derived/ds006780_assr_features/all_run_features.csv"))
    parser.add_argument("--ds006780-subjects", type=Path, default=Path("outputs/models/ds006780_behavioral_validity/subject_level_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/models/behavior_measurement_reliability"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    ds7 = ds007648_halves(pd.read_csv(args.ds007648_trials, low_memory=False))
    ds6 = ds006780_halves(
        args.ds006780_events_root,
        pd.read_csv(args.ds006780_runs),
        pd.read_csv(args.ds006780_subjects),
    )
    reliability = build_reliability(ds7, ds6, args.seed, args.bootstrap)
    args.output.mkdir(parents=True, exist_ok=True)
    ds7.to_csv(args.output / "ds007648_behavior_halves.csv", index=False)
    ds6.to_csv(args.output / "ds006780_behavior_halves.csv", index=False)
    reliability.to_csv(args.output / "behavior_reliability.csv", index=False)
    print(reliability.to_string(index=False))


if __name__ == "__main__":
    main()
