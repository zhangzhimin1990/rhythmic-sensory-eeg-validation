#!/usr/bin/env python3
"""Evaluate the frozen Dryad five-participant R2 technical and measurement gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data/public/dryad_t76hdr8dm/v5/metadata/Dataset.xlsx"
DEFAULT_SELECTION = (
    ROOT / "outputs/qc/dryad_raw_stream/R2_selection_frozen_v1/r2_selected_subjects.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/qc/dryad_raw_stream"
DEFAULT_OUTPUT = ROOT / "outputs/qc/dryad_raw_stream/R2_gate_frozen_v1"
CONDITION_PREFIX = {
    "2_hz": "doshz",
    "f_theta": "fθ",
    "f_theta_plus": "fθ+",
    "non_rhythmic": "nr",
}
CONGRUENCY_SUFFIX = {"congruent": "con", "incongruent": "incon"}
EXPECTED_TRIALS_PER_PARTICIPANT = 384
ANALYSIS_WINDOW_END_MS = 1750.0
MINIMUM_PRE_TARGET_MARGIN_MS = 100.0


def behaviour_alignment(trials: pd.DataFrame, author_row: pd.Series) -> pd.DataFrame:
    """Compare reconstructed accuracy and correct-trial RT with the released table."""
    rows = []
    for condition, prefix in CONDITION_PREFIX.items():
        for congruency, suffix in CONGRUENCY_SUFFIX.items():
            cell = trials.loc[
                trials["condition"].eq(condition)
                & trials["congruency"].eq(congruency)
                & trials["sequence_valid"].astype(bool)
            ]
            scored = cell.loc[cell["response_class"].isin(["correct", "incorrect"])]
            correct = cell.loc[cell["response_class"].eq("correct")]
            reconstructed_accuracy = (
                float(len(correct) / len(scored)) if len(scored) else np.nan
            )
            reconstructed_rt = float(correct["rt_ms"].mean()) if len(correct) else np.nan
            author_accuracy = float(author_row[f"{prefix}_acc_{suffix}"])
            author_rt = float(author_row[f"{prefix}_rt_{suffix}"])
            rows.append(
                {
                    "condition": condition,
                    "congruency": congruency,
                    "n_total_trials": int(len(cell)),
                    "n_scored_trials": int(len(scored)),
                    "n_correct_trials": int(len(correct)),
                    "reconstructed_accuracy": reconstructed_accuracy,
                    "author_accuracy": author_accuracy,
                    "accuracy_absolute_difference": abs(
                        reconstructed_accuracy - author_accuracy
                    ),
                    "reconstructed_correct_rt_ms": reconstructed_rt,
                    "author_correct_rt_ms": author_rt,
                    "rt_absolute_difference_ms": abs(reconstructed_rt - author_rt),
                }
            )
    return pd.DataFrame(rows)


def evaluate_participant(
    subject: int,
    audit_summary: dict[str, object],
    signal_summary: dict[str, object],
    condition_metrics: pd.DataFrame,
    alignment: pd.DataFrame,
    minimum_stimulation_to_target_ms: float,
) -> tuple[dict[str, object], dict[str, object]]:
    expected_conditions = set(CONDITION_PREFIX)
    if set(condition_metrics["condition"]) != expected_conditions:
        raise ValueError(f"S{subject}: signal table does not contain exactly four conditions")
    by_condition = condition_metrics.set_index("condition")
    theta_plus = by_condition.loc["f_theta_plus"]
    non_rhythmic = by_condition.loc["non_rhythmic"]

    n_trials = int(audit_summary["n_trials"])
    n_valid = int(audit_summary["n_valid_sequences"])
    n_scalp = int(audit_summary["n_scalp_eeg_channels"])
    n_bad = int(signal_summary["n_bad_channels"])
    n_good = int(signal_summary["n_good_channels"])
    minimum_roi_channels = min(
        len([name for name in str(value).split(";") if name])
        for value in condition_metrics["frontocentral_roi"]
    )
    total_usable_fraction = float(signal_summary["n_usable_trials"]) / float(
        signal_summary["n_trials"]
    )
    per_condition_usable = (
        condition_metrics["n_usable_trials"] / condition_metrics["n_trials"]
    )
    finite_metrics = np.isfinite(
        condition_metrics[
            ["mean_itpc_in_stimulation_window", "evoked_local_log_snr_db"]
        ].to_numpy(dtype=float)
    ).all()
    accuracy_exact = bool(
        (alignment["accuracy_absolute_difference"] <= 1e-6).all()
    )
    max_rt_difference = float(alignment["rt_absolute_difference_ms"].max())
    pre_target_margin_ms = minimum_stimulation_to_target_ms - ANALYSIS_WINDOW_END_MS

    technical_components = {
        "scalp_and_good_channel_gate": (
            n_scalp == 64 and n_bad <= 12 and n_good >= 52 and minimum_roi_channels >= 3
        ),
        "valid_event_sequence_gate": (
            n_valid / EXPECTED_TRIALS_PER_PARTICIPANT >= 0.95
        ),
        "behaviour_accuracy_exact_gate": accuracy_exact,
        "behaviour_rt_within_2ms_gate": max_rt_difference <= 2.0,
        "overall_usable_trial_gate": total_usable_fraction >= 0.90,
        "per_condition_usable_trial_gate": bool((per_condition_usable >= 0.80).all()),
        "finite_primary_metrics_gate": bool(finite_metrics),
        "analysis_window_precedes_target_by_at_least_100ms_gate": (
            pre_target_margin_ms >= MINIMUM_PRE_TARGET_MARGIN_MS
        ),
    }
    participant_row: dict[str, object] = {
        "subject": int(subject),
        "n_trials": n_trials,
        "n_valid_sequences": n_valid,
        "valid_sequence_fraction": n_valid / n_trials if n_trials else np.nan,
        "n_scalp_channels": n_scalp,
        "n_bad_channels": n_bad,
        "n_good_channels": n_good,
        "minimum_frontocentral_roi_channels": minimum_roi_channels,
        "expected_trial_coverage_fraction": n_valid / EXPECTED_TRIALS_PER_PARTICIPANT,
        "overall_usable_trial_fraction": total_usable_fraction,
        "minimum_condition_usable_trial_fraction": float(per_condition_usable.min()),
        "maximum_accuracy_absolute_difference": float(
            alignment["accuracy_absolute_difference"].max()
        ),
        "maximum_rt_absolute_difference_ms": max_rt_difference,
        "minimum_stimulation_to_target_ms": minimum_stimulation_to_target_ms,
        "minimum_analysis_window_pre_target_margin_ms": pre_target_margin_ms,
        **technical_components,
        "technical_pass": bool(all(technical_components.values())),
    }
    contrast_row = {
        "subject": int(subject),
        "itpc_theta_plus_minus_non_rhythmic": float(
            theta_plus["mean_itpc_in_stimulation_window"]
            - non_rhythmic["mean_itpc_in_stimulation_window"]
        ),
        "odd_itpc_theta_plus_minus_non_rhythmic": float(
            theta_plus["odd_trial_itpc"] - non_rhythmic["odd_trial_itpc"]
        ),
        "even_itpc_theta_plus_minus_non_rhythmic": float(
            theta_plus["even_trial_itpc"] - non_rhythmic["even_trial_itpc"]
        ),
        "evoked_snr_theta_plus_minus_non_rhythmic_db": float(
            theta_plus["evoked_local_log_snr_db"]
            - non_rhythmic["evoked_local_log_snr_db"]
        ),
    }
    return participant_row, contrast_row


def evaluate_group_gate(
    participant_gate: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> dict[str, object]:
    if len(participant_gate) != 5 or len(contrasts) != 5:
        raise ValueError("the frozen R2 gate requires exactly five participants")
    contrast_columns = {
        "full_itpc": "itpc_theta_plus_minus_non_rhythmic",
        "odd_itpc": "odd_itpc_theta_plus_minus_non_rhythmic",
        "even_itpc": "even_itpc_theta_plus_minus_non_rhythmic",
        "evoked_snr": "evoked_snr_theta_plus_minus_non_rhythmic_db",
    }
    counts = {
        name: int((contrasts[column] > 0).sum())
        for name, column in contrast_columns.items()
    }
    medians = {
        name: float(contrasts[column].median())
        for name, column in contrast_columns.items()
    }
    measurement_components = {
        "full_itpc_median_positive": medians["full_itpc"] > 0,
        "full_itpc_positive_in_at_least_4_of_5": counts["full_itpc"] >= 4,
        "odd_itpc_positive_in_at_least_3_of_5": counts["odd_itpc"] >= 3,
        "even_itpc_positive_in_at_least_3_of_5": counts["even_itpc"] >= 3,
        "evoked_snr_median_positive": medians["evoked_snr"] > 0,
        "evoked_snr_positive_in_at_least_3_of_5": counts["evoked_snr"] >= 3,
    }
    technical_pass_count = int(participant_gate["technical_pass"].sum())
    technical_go = technical_pass_count >= 4
    measurement_go = bool(all(measurement_components.values()))
    return {
        "n_participants": 5,
        "technical_pass_count": technical_pass_count,
        "technical_go_requires_at_least_4_of_5": technical_go,
        "positive_contrast_counts": counts,
        "median_contrasts": medians,
        "measurement_components": measurement_components,
        "measurement_go": measurement_go,
        "overall_r2_go": technical_go and measurement_go,
        "scope": (
            "R2 is a pipeline and measurement heterogeneity gate. The theta-plus "
            "versus non-rhythmic contrast is evaluated at the same analysis "
            "frequency and functions as a temporal-regularity/pipeline-sensitivity "
            "check; it is not a matched-stimulus-frequency comparison, group "
            "significance test, or entrainment mechanism claim."
        ),
    }


def participant_paths(output_root: Path, subject: int) -> tuple[Path, Path]:
    if subject == 2:
        return output_root / "S2_pilot_v3", output_root / "S2_signal_qc_frozen_v1"
    base = output_root / "R2_subjects" / f"S{subject}"
    return base / "audit", base / "signal"


def run(
    metadata_path: Path,
    selection_path: Path,
    output_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    metadata = pd.read_excel(metadata_path).set_index("subject")
    selection = pd.read_csv(selection_path).sort_values("selection_order")
    subjects = selection["subject"].astype(int).tolist()
    if subjects != [2, 13, 27, 23, 37]:
        raise ValueError(f"selection is not the frozen R2 order: {subjects}")

    participant_rows = []
    contrast_rows = []
    alignment_rows = []
    for subject in subjects:
        audit_dir, signal_dir = participant_paths(output_root, subject)
        with (audit_dir / "audit_summary.json").open(encoding="utf-8") as handle:
            audit_summary = json.load(handle)
        with (signal_dir / "qc_summary.json").open(encoding="utf-8") as handle:
            signal_summary = json.load(handle)
        trials = pd.read_csv(audit_dir / "trials.csv")
        metrics = pd.read_csv(signal_dir / "condition_target_engagement.csv")
        alignment = behaviour_alignment(trials, metadata.loc[subject])
        alignment.insert(0, "subject", subject)
        valid_timing = trials.loc[
            trials["sequence_valid"].astype(bool), "stimulation_to_target_ms"
        ]
        minimum_stimulation_to_target_ms = float(valid_timing.min())
        participant_row, contrast_row = evaluate_participant(
            subject,
            audit_summary,
            signal_summary,
            metrics,
            alignment,
            minimum_stimulation_to_target_ms,
        )
        participant_rows.append(participant_row)
        contrast_rows.append(contrast_row)
        alignment_rows.append(alignment)

    participant_gate = pd.DataFrame(participant_rows)
    contrasts = pd.DataFrame(contrast_rows)
    alignments = pd.concat(alignment_rows, ignore_index=True)
    summary = evaluate_group_gate(participant_gate, contrasts)
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_gate.to_csv(output_dir / "participant_technical_gate.csv", index=False)
    contrasts.to_csv(
        output_dir / "same_analysis_frequency_condition_contrasts.csv", index=False
    )
    alignments.to_csv(output_dir / "behaviour_reconstruction_alignment.csv", index=False)
    with (output_dir / "r2_gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(args.metadata, args.selection, args.output_root, args.output_dir),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
