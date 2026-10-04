#!/usr/bin/env python3
"""Build the locked four-panel Dryad confirmatory Figure 7."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "outputs/models/dryad_confirmatory_group_v1"
AUDIT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_v1"
SUBJECTS = ROOT / "outputs/models/dryad_confirmatory_subjects_v1"
DEFAULT_OUTPUT = ROOT / "outputs/manuscript_figures_v03"

INK = "#202733"
BLUE = "#245B8A"
ORANGE = "#C77818"
LIGHT_BLUE = "#9AB8D0"
GREY = "#667085"
LIGHT_GREY = "#E5E7EB"
PALE = "#F6F7F9"
PDF_METADATA = {
    "Creator": "public-eeg-validation",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_subject_neural(fallback_path: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted(SUBJECTS.glob("S*/condition_neural_metrics.csv"))
    if len(paths) != 35:
        if not fallback_path.is_file():
            raise FileNotFoundError(
                "Figure 7 requires either 35 subject metric files or the deidentified "
                "participant-contrast table"
            )
        out = pd.read_csv(fallback_path)
        required = {"subject", "itpc_theta_plus_minus_non_rhythmic"}
        if len(out) != 35 or not required.issubset(out.columns):
            raise ValueError("Deidentified Figure 7 participant contrasts are incomplete")
        return out, [fallback_path]
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    wide = data.pivot(index="subject", columns="condition", values="mean_itpc_in_stimulation_window")
    out = pd.DataFrame(
        {
            "subject": wide.index.astype(int),
            "itpc_theta_plus_minus_non_rhythmic": (
                wide["f_theta_plus"] - wide["non_rhythmic"]
            ).to_numpy(),
        }
    ).reset_index(drop=True).sort_values("subject")
    if len(out) != 35 or out.itpc_theta_plus_minus_non_rhythmic.isna().any():
        raise ValueError("Figure 7 requires 35 complete participant ITPC contrasts")
    return out, paths


def style_axis(ax, *, xgrid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8.5)
    if xgrid:
        ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def interval(ax, estimate: float, low: float, high: float, *, color: str = BLUE) -> None:
    ax.errorbar(
        estimate,
        0,
        xerr=[[estimate - low], [high - estimate]],
        fmt="o",
        markersize=5.5,
        color=color,
        ecolor=color,
        elinewidth=1.8,
        capsize=2.5,
        zorder=4,
    )
    ax.axvline(0, color=GREY, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks([])
    style_axis(ax)


def build_figure(output_dir: Path) -> list[Path]:
    neural_path = GROUP / "paired_neural_effects.csv"
    behaviour_path = GROUP / "paired_behaviour_effects.csv"
    coupling_path = AUDIT / "within_condition_coupling_simple_slopes.csv"
    prediction_path = AUDIT / "prediction_increment_bootstrap.csv"
    sensitivity_path = AUDIT / "primary_roi_channel_sensitivity.csv"

    neural = pd.read_csv(neural_path)
    behaviour = pd.read_csv(behaviour_path)
    coupling = pd.read_csv(coupling_path)
    prediction = pd.read_csv(prediction_path)
    sensitivity = pd.read_csv(sensitivity_path)
    subject_delta, subject_paths = load_subject_neural(
        output_dir / "Figure7B_participant_itpc_contrasts.csv"
    )

    primary = neural.loc[neural.metric.eq("mean_itpc_in_stimulation_window")].iloc[0]
    sens = sensitivity.iloc[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_delta.to_csv(output_dir / "Figure7B_participant_itpc_contrasts.csv", index=False)
    behaviour.to_csv(output_dir / "Figure7C_behaviour.csv", index=False)
    coupling.to_csv(output_dir / "Figure7C_coupling.csv", index=False)
    prediction.to_csv(output_dir / "Figure7D_prediction_increment.csv", index=False)

    fig = plt.figure(figsize=(14.2, 10.5), facecolor="white")
    outer = fig.add_gridspec(2, 2, width_ratios=[0.94, 1.06], height_ratios=[0.86, 1.14], hspace=0.32, wspace=0.25)

    # Panel A: locked design and sample flow.
    ax = fig.add_subplot(outer[0, 0])
    ax.axis("off")
    ax.set_title("A  Frozen design and outcome-blind sample flow", loc="left", fontsize=12, fontweight="bold", color=INK)
    xs = [0.12, 0.38, 0.64, 0.88]
    labels = [
        ("Public\nrelease", "N = 44"),
        ("Development\nset", "n = 5"),
        ("Untouched\nvalidation", "n = 39"),
        ("Confirmatory\ncohort", "n = 35"),
    ]
    for i, (x, (title, count)) in enumerate(zip(xs, labels)):
        edge = ORANGE if i == 1 else BLUE
        width = 0.205
        ax.add_patch(FancyBboxPatch(
            (x - width / 2, 0.56), width, 0.17,
            boxstyle="round,pad=0.015", facecolor="white", edgecolor=edge,
            linewidth=1.5, transform=ax.transAxes, clip_on=False,
        ))
        ax.text(
            x,
            0.645,
            f"{title}\n{count}",
            ha="center",
            va="center",
            fontsize=8.25,
            color=INK,
            transform=ax.transAxes,
        )
        if i < 3:
            ax.annotate("", xy=(xs[i + 1] - 0.112, 0.645), xytext=(x + 0.112, 0.645), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(0.47, 0.43, "Frozen technical rules applied condition-blind\n4 technical failures; no condition effects accessed", ha="center", va="center", fontsize=8.7, color=GREY, transform=ax.transAxes)
    ax.text(0.5, 0.18, "Primary contrast: theta-plus − non-rhythmic", ha="center", fontsize=10.2, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(0.5, 0.08, "Common analysis frequency: 1.33 × individual theta", ha="center", fontsize=9.2, color=GREY, transform=ax.transAxes)

    # Panel B: participant ITPC contrasts and primary estimate.
    ax = fig.add_subplot(outer[0, 1])
    vals = subject_delta.itpc_theta_plus_minus_non_rhythmic.to_numpy()
    rng = np.random.default_rng(20260930)
    jitter = rng.normal(0, 0.028, len(vals))
    ax.scatter(vals, jitter, s=27, facecolor="white", edgecolor=BLUE, linewidth=1.0, alpha=0.9, zorder=3)
    ax.errorbar(primary.mean_difference_theta_plus_minus_non_rhythmic, 0.18,
                xerr=[[primary.mean_difference_theta_plus_minus_non_rhythmic - primary.ci95_low],
                      [primary.ci95_high - primary.mean_difference_theta_plus_minus_non_rhythmic]],
                fmt="o", color=BLUE, ecolor=BLUE, markersize=7, elinewidth=2.4, capsize=3, label="Primary 95% CI")
    ax.errorbar(primary.mean_difference_theta_plus_minus_non_rhythmic, 0.12,
                xerr=[[primary.mean_difference_theta_plus_minus_non_rhythmic - primary.bootstrap_ci95_low],
                      [primary.bootstrap_ci95_high - primary.mean_difference_theta_plus_minus_non_rhythmic]],
                fmt="s", mfc="white", mec=ORANGE, color=ORANGE, ecolor=ORANGE, markersize=5, elinewidth=1.6, capsize=2, label="Bootstrap 95% CI")
    ax.errorbar(sens["mean_difference_theta_plus_minus_non_rhythmic"], -0.16,
                xerr=[[sens["mean_difference_theta_plus_minus_non_rhythmic"] - sens["ci95_low"]],
                      [sens["ci95_high"] - sens["mean_difference_theta_plus_minus_non_rhythmic"]]],
                fmt="D", color=GREY, ecolor=GREY, markersize=4.8, elinewidth=1.4, capsize=2)
    bound = float(primary.equivalence_bound_raw_plus_minus)
    ax.axvspan(-bound, bound, color=LIGHT_GREY, alpha=0.55, zorder=0, label="±0.30 dz bounds")
    ax.axvline(0, color=GREY, linestyle="--", linewidth=1.0)
    ax.set_ylim(-0.23, 0.25)
    ax.set_yticks([0.18, 0.12, 0, -0.16], ["Mean, t CI", "Mean, bootstrap CI", "Participants", "≥4 ROI channels"])
    ax.set_xlabel("Frontocentral ITPC difference", fontsize=9.5)
    ax.set_title("B  Primary target-engagement estimand", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.text(0.99, 0.96, "Δ = 0.088 [0.051, 0.124]\npaired dz = 0.830; p = 2.23×10⁻⁵",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.9, color=INK)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")

    # Panel C: independent raw-unit mini-axes for behavior and coupling.
    cgrid = GridSpecFromSubplotSpec(3, 6, subplot_spec=outer[1, 0], height_ratios=[0.15, 0.75, 1.0], hspace=0.74, wspace=0.75)
    title_ax = fig.add_subplot(cgrid[0, :])
    title_ax.axis("off")
    title_ax.text(0, 0.92, "C  Acute behavior and condition-specific coupling", fontsize=12, fontweight="bold", color=INK, va="top")
    beh_order = ["accuracy", "median_correct_rt_ms", "inverse_efficiency_ms"]
    beh_titles = ["Accuracy", "Correct RT", "Inverse efficiency"]
    beh_units = ["proportion", "ms", "ms"]
    for j, (metric, title, unit) in enumerate(zip(beh_order, beh_titles, beh_units)):
        sub = fig.add_subplot(cgrid[1, j * 2:(j + 1) * 2])
        row = behaviour.loc[behaviour.metric.eq(metric)].iloc[0]
        interval(sub, row.mean_difference_theta_plus_minus_non_rhythmic, row.ci95_low, row.ci95_high, color=BLUE)
        sub.set_title(title, fontsize=9.2, color=INK)
        sub.set_xlabel(unit, fontsize=8)
        sub.text(0.5, 0.94, f"{row.mean_difference_theta_plus_minus_non_rhythmic:.3f}\n[{row.ci95_low:.3f}, {row.ci95_high:.3f}]",
                 transform=sub.transAxes, ha="center", va="top", fontsize=7.2, color=GREY)
        sub.text(0.97, 0.06, f"Holm p={row.p_value_holm_behaviour:.3g}", transform=sub.transAxes,
                 ha="right", va="bottom", fontsize=7.0, color=GREY)
    coupling_specs = [
        ("correctness_binomial_gee", "Correctness slope", "log-odds / phase SD"),
        ("correct_rt_gaussian_gee", "Correct log-RT slope", "log-ms / phase SD"),
    ]
    for j, (model, title, unit) in enumerate(coupling_specs):
        sub = fig.add_subplot(cgrid[2, j * 3:(j + 1) * 3])
        rows = coupling.loc[coupling.model.eq(model)].copy()
        positions = np.arange(len(rows))
        for pos, (_, row) in zip(positions, rows.iterrows()):
            color = BLUE if row.condition == "f_theta_plus" else ORANGE
            sub.errorbar(row.simple_slope_phase_alignment_within, pos,
                         xerr=[[row.simple_slope_phase_alignment_within - row.ci95_low],
                               [row.ci95_high - row.simple_slope_phase_alignment_within]],
                         fmt="o", color=color, ecolor=color, markersize=5, elinewidth=1.5, capsize=2)
        sub.axvline(0, color=GREY, linestyle="--", linewidth=0.9)
        sub.set_yticks(positions, ["Non-rhythmic", "Theta-plus"])
        sub.invert_yaxis()
        sub.set_title(title, fontsize=9.2, color=INK)
        sub.set_xlabel(unit, fontsize=8)
        style_axis(sub)
        sub.text(0.99, 0.05, "all Holm p = 1.000", transform=sub.transAxes, ha="right", va="bottom", fontsize=7.4, color=GREY)

    # Panel D: held-out incremental prediction.
    ax = fig.add_subplot(outer[1, 1])
    positions = np.arange(len(prediction))
    labels = ["vs training-fold mean", "vs baseline features"]
    for pos, (_, row) in zip(positions, prediction.iterrows()):
        color = BLUE if row.comparator == "training_fold_mean" else ORANGE
        ax.errorbar(row.mae_improvement_comparator_minus_neural, pos,
                    xerr=[[row.mae_improvement_comparator_minus_neural - row.mae_improvement_bootstrap_ci95_low],
                          [row.mae_improvement_bootstrap_ci95_high - row.mae_improvement_comparator_minus_neural]],
                    fmt="o", color=color, ecolor=color, markersize=7, elinewidth=2.1, capsize=3)
        ax.text(row.mae_improvement_bootstrap_ci95_high + 0.28, pos,
                f"RMSE Δ {row.rmse_improvement_comparator_minus_neural:+.2f} ms", va="center", fontsize=8.3, color=GREY)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=1.0)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("MAE improvement (ms; 95% participant-bootstrap CI)\nPositive values favor the neural model", fontsize=9.5)
    ax.set_title("D  Nested held-out incremental prediction", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.text(0.98, 0.08, "Prediction gate: DID NOT PASS", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, fontweight="bold", color=INK, bbox=dict(boxstyle="round,pad=0.3", facecolor=PALE, edgecolor=GREY))
    style_axis(ax)

    fig.suptitle("Temporal regularity engages a phase-locked response but does not validate a behavioral mechanism proxy", fontsize=15.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.012, "Healthy older adults; theta-plus versus non-rhythmic stimulation. Intervals are 95% confidence intervals unless stated otherwise.", ha="center", fontsize=9, color=GREY)

    png = output_dir / "Figure7_dryad_confirmatory.png"
    pdf = output_dir / "Figure7_dryad_confirmatory.pdf"
    svg = output_dir / "Figure7_dryad_confirmatory.svg"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata=PDF_METADATA)
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    input_paths = [neural_path, behaviour_path, coupling_path, prediction_path, sensitivity_path, *subject_paths]
    manifest = {
        "figure": "Figure7_dryad_confirmatory",
        "generated_from_locked_structure": "89_Dryad确认性Figure7与正文回填方案.md",
        "n_confirmatory": 35,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths},
        "panels": {
            "A": "frozen design and outcome-blind sample flow",
            "B": "participant ITPC contrasts, primary interval, equivalence bounds, and ROI sensitivity",
            "C": "raw-unit behavioral contrasts and within-condition phase-alignment slopes",
            "D": "nested held-out MAE increment with RMSE guard labels",
        },
        "claim_ceiling": "target_engagement_and_behaviour_without_specific_coupling",
    }
    manifest_path = output_dir / "Figure7_dryad_confirmatory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return [png, pdf, svg, manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in build_figure(args.output):
        print(path)


if __name__ == "__main__":
    main()
