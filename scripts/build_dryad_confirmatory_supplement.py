#!/usr/bin/env python3
"""Build Dryad confirmatory supplementary figures S5-S6 and tables S6-S7."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "outputs/models/dryad_confirmatory_group_v1"
AUDIT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_v1"
SUBJECTS = ROOT / "outputs/models/dryad_confirmatory_subjects_v1"
DEFAULT_OUTPUT = ROOT / "outputs/manuscript_supplement_v03"
BLUE = "#245B8A"
ORANGE = "#C77818"
GREY = "#667085"
LIGHT_GREY = "#E5E7EB"
INK = "#202733"
PDF_METADATA = {"Creator": "public-eeg-validation", "Producer": "Matplotlib", "CreationDate": None, "ModDate": None}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=1.0, zorder=1)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def subject_flow_table(fallback_path: Path) -> tuple[pd.DataFrame, list[Path]]:
    summary_paths = sorted(SUBJECTS.glob("S*/confirmatory_subject_summary.json"))
    metric_paths = sorted(SUBJECTS.glob("S*/condition_neural_metrics.csv"))
    if len(summary_paths) != 35 or len(metric_paths) != 35:
        if not fallback_path.is_file():
            raise FileNotFoundError(
                "Supplementary Table S6 requires either 35 subject outputs or the "
                "deidentified participant-flow table"
            )
        table = pd.read_csv(fallback_path)
        if len(table) != 35 or "analytic_id" not in table.columns:
            raise ValueError("Deidentified Supplementary Table S6 is incomplete")
        return table, [fallback_path]
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    rows = []
    for index, summary in enumerate(sorted(summaries, key=lambda item: int(item["subject"])), start=1):
        subject = int(summary["subject"])
        metrics = pd.read_csv(SUBJECTS / f"S{subject}" / "condition_neural_metrics.csv").set_index("condition")
        rows.append(
            {
                "analytic_id": f"C{index:02d}",
                "individual_theta_hz": summary["individual_theta_hz"],
                "analysis_frequency_hz": summary["analysis_frequency_hz"],
                "n_good_frozen_qc_channels": summary["n_good_frozen_qc_channels"],
                "n_good_frontocentral_roi_channels": len(str(metrics.loc["f_theta_plus", "frontocentral_roi"]).split(";")),
                "n_technical_usable_epochs_reproduced": summary["n_technical_usable_epochs_reproduced"],
                "theta_plus_event_valid_trials": int(metrics.loc["f_theta_plus", "n_event_valid_trials"]),
                "theta_plus_eeg_usable_trials": int(metrics.loc["f_theta_plus", "n_usable_trials"]),
                "non_rhythmic_event_valid_trials": int(metrics.loc["non_rhythmic", "n_event_valid_trials"]),
                "non_rhythmic_eeg_usable_trials": int(metrics.loc["non_rhythmic", "n_usable_trials"]),
                "derivation_status": summary["status"],
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != 35:
        raise ValueError("Supplementary Table S6 requires 35 confirmatory participants")
    return table, summary_paths + metric_paths


def complete_estimate_table() -> pd.DataFrame:
    neural = pd.read_csv(GROUP / "paired_neural_effects.csv")
    behavior = pd.read_csv(GROUP / "paired_behaviour_effects.csv")
    coupling = pd.read_csv(AUDIT / "within_condition_coupling_simple_slopes.csv")
    prediction = pd.read_csv(AUDIT / "prediction_increment_bootstrap.csv")
    rows: list[dict[str, object]] = []
    for _, row in neural.iterrows():
        family = "primary_neural" if row.metric == "mean_itpc_in_stimulation_window" else "key_secondary_neural"
        rows.append({
            "family": family, "endpoint": row.metric, "comparison": "theta_plus_minus_non_rhythmic",
            "n_subjects": int(row.n), "n_trials": "", "estimate": row.mean_difference_theta_plus_minus_non_rhythmic,
            "ci95_low": row.ci95_low, "ci95_high": row.ci95_high, "standardized_estimate": row.paired_dz,
            "p_value_raw": row.p_value_t_two_sided,
            "p_value_adjusted": "" if pd.isna(row.p_value_holm_key_secondary) else row.p_value_holm_key_secondary,
            "multiplicity": "single primary" if family == "primary_neural" else "Holm across 3 neural secondary endpoints",
        })
    for _, row in behavior.iterrows():
        rows.append({
            "family": "behaviour", "endpoint": row.metric, "comparison": "theta_plus_minus_non_rhythmic",
            "n_subjects": int(row.n), "n_trials": "", "estimate": row.mean_difference_theta_plus_minus_non_rhythmic,
            "ci95_low": row.ci95_low, "ci95_high": row.ci95_high, "standardized_estimate": row.paired_dz,
            "p_value_raw": row.p_value_t_two_sided, "p_value_adjusted": row.p_value_holm_behaviour,
            "multiplicity": "Holm across 3 behavioural endpoints",
        })
    for _, row in coupling.iterrows():
        rows.append({
            "family": "within_condition_coupling", "endpoint": row.model, "comparison": row.condition,
            "n_subjects": int(row.n_subjects), "n_trials": int(row.n_trials), "estimate": row.simple_slope_phase_alignment_within,
            "ci95_low": row.ci95_low, "ci95_high": row.ci95_high, "standardized_estimate": "",
            "p_value_raw": row.p_value_two_sided, "p_value_adjusted": row.p_value_holm_simple_slope_family,
            "multiplicity": "Holm across 4 simple slopes",
        })
    for _, row in prediction.iterrows():
        rows.append({
            "family": "incremental_prediction", "endpoint": "MAE improvement", "comparison": row.comparator,
            "n_subjects": int(row.n), "n_trials": "", "estimate": row.mae_improvement_comparator_minus_neural,
            "ci95_low": row.mae_improvement_bootstrap_ci95_low, "ci95_high": row.mae_improvement_bootstrap_ci95_high,
            "standardized_estimate": "", "p_value_raw": "", "p_value_adjusted": "",
            "multiplicity": "estimation-only participant bootstrap",
        })
    return pd.DataFrame(rows)


def build(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    neural_path = GROUP / "paired_neural_effects.csv"
    consistency_path = AUDIT / "internal_consistency_bootstrap.csv"
    neural = pd.read_csv(neural_path)
    consistency = pd.read_csv(consistency_path)

    # Supplementary Figure S5: component decomposition.
    s5 = neural.loc[neural.metric.isin([
        "evoked_local_log_snr_db", "total_power_local_log_snr_db", "induced_local_log_snr_db"
    ])].copy()
    s5["label"] = s5.metric.map({
        "evoked_local_log_snr_db": "Evoked local log-SNR",
        "total_power_local_log_snr_db": "Total-power local log-SNR",
        "induced_local_log_snr_db": "Induced local log-SNR",
    })
    fig, ax = plt.subplots(figsize=(8.1, 4.4), constrained_layout=True)
    positions = np.arange(len(s5))
    for pos, (_, row) in zip(positions, s5.iterrows()):
        color = BLUE if row.metric == "evoked_local_log_snr_db" else GREY
        ax.errorbar(row.mean_difference_theta_plus_minus_non_rhythmic, pos,
                    xerr=[[row.mean_difference_theta_plus_minus_non_rhythmic - row.ci95_low],
                          [row.ci95_high - row.mean_difference_theta_plus_minus_non_rhythmic]],
                    fmt="o", color=color, ecolor=color, markersize=7, elinewidth=2.0, capsize=3)
        ax.text(row.ci95_high + 0.35, pos,
                f"Δ={row.mean_difference_theta_plus_minus_non_rhythmic:.3f}; Holm p={row.p_value_holm_key_secondary:.3g}",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(positions, s5.label)
    ax.invert_yaxis()
    ax.set_xlabel("Theta-plus − non-rhythmic difference (dB; 95% CI)")
    ax.set_title("Supplementary Figure S5. Phase-locked and non-phase-locked spectral components", loc="left", fontweight="bold")
    style_axis(ax)
    s5_png = output_dir / "FigureS5_dryad_component_decomposition.png"
    s5_pdf = output_dir / "FigureS5_dryad_component_decomposition.pdf"
    fig.savefig(s5_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(s5_pdf, bbox_inches="tight", facecolor="white", metadata=PDF_METADATA)
    plt.close(fig)

    # Supplementary Figure S6: internal consistency with bootstrap intervals.
    fig, ax = plt.subplots(figsize=(9.2, 6.1), constrained_layout=True)
    plot = consistency.copy()
    plot["label"] = (
        plot.condition.map({"f_theta_plus": "Theta-plus", "non_rhythmic": "Non-rhythmic"}) + " · " +
        plot.metric.map({"itpc": "ITPC", "evoked_local_log_snr_db": "Evoked SNR"}) + " · " +
        plot.split.map({"odd_even": "odd–even", "time_half": "time-half"})
    )
    positions = np.arange(len(plot))
    for pos, (_, row) in zip(positions, plot.iterrows()):
        color = BLUE if row.metric == "itpc" else ORANGE
        ax.errorbar(row.spearman_brown, pos,
                    xerr=[[row.spearman_brown - row.spearman_brown_bootstrap_ci95_low],
                          [row.spearman_brown_bootstrap_ci95_high - row.spearman_brown]],
                    fmt="o", color=color, ecolor=color, markersize=6.5, elinewidth=1.8, capsize=3)
    ax.set_yticks(positions, plot.label)
    ax.invert_yaxis()
    ax.set_xlim(-0.85, 1.05)
    ax.set_xlabel("Spearman–Brown within-recording coefficient (95% bootstrap CI)")
    ax.set_title("Supplementary Figure S6. Internal consistency depends on condition and metric", loc="left", fontweight="bold")
    ax.text(0.99, 0.98, "Within-session internal consistency; not test–retest reliability", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=GREY)
    style_axis(ax)
    s6_png = output_dir / "FigureS6_dryad_internal_consistency.png"
    s6_pdf = output_dir / "FigureS6_dryad_internal_consistency.pdf"
    fig.savefig(s6_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(s6_pdf, bbox_inches="tight", facecolor="white", metadata=PDF_METADATA)
    plt.close(fig)

    table_s6 = output_dir / "TableS6_dryad_participant_flow.csv"
    flow, subject_inputs = subject_flow_table(table_s6)
    flow.to_csv(table_s6, index=False)
    estimates = complete_estimate_table()
    table_s7 = output_dir / "TableS7_dryad_confirmatory_estimates.csv"
    estimates.to_csv(table_s7, index=False)

    inputs = [
        neural_path, consistency_path, GROUP / "paired_behaviour_effects.csv",
        AUDIT / "within_condition_coupling_simple_slopes.csv", AUDIT / "prediction_increment_bootstrap.csv",
        *subject_inputs,
    ]
    manifest = {
        "artifacts": [path.name for path in (s5_png, s5_pdf, s6_png, s6_pdf, table_s6, table_s7)],
        "n_confirmatory": 35,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "privacy": "Table S6 uses sequential analytic IDs and omits source filenames and checksums",
        "claim_boundary": "internal consistency is not test-retest reliability; coupling is not mediation",
    }
    manifest_path = output_dir / "dryad_confirmatory_supplement_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return [s5_png, s5_pdf, s6_png, s6_pdf, table_s6, table_s7, manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in build(args.output):
        print(path)


if __name__ == "__main__":
    main()
