#!/usr/bin/env python3
"""Build the candidate main figure for the personalised-theta validation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/models/dryad_theta_validation_gate"
DEFAULT_OUTPUT = ROOT / "outputs/manuscript_figures_v02"
BLUE = "#2B6CB0"
ORANGE = "#DD971A"
PURPLE = "#7A52A1"
RED = "#C6532E"
GREY = "#666666"
PDF_METADATA = {
    "Creator": "public-eeg-validation",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def errorbar_forest(ax, frame, *, y_col, estimate, low, high, color_col=None):
    positions = np.arange(len(frame))
    colors = frame[color_col].tolist() if color_col else [BLUE] * len(frame)
    for position, (_, row), color in zip(positions, frame.iterrows(), colors):
        ax.errorbar(
            row[estimate],
            position,
            xerr=[[row[estimate] - row[low]], [row[high] - row[estimate]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=0,
            markersize=7,
            zorder=3,
        )
    ax.set_yticks(positions, frame[y_col])
    ax.invert_yaxis()
    ax.axvline(0, color=GREY, linestyle="--", linewidth=1.2)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def build_figure(input_dir: Path, output_dir: Path) -> list[Path]:
    paired_path = input_dir / "paired_condition_effects.csv"
    assoc_path = input_dir / "association_models.csv"
    moderator_path = input_dir / "moderator_checks.csv"
    prediction_path = input_dir / "pretreatment_subject_predictions.csv"
    summary_path = input_dir / "pretreatment_prediction_summary.csv"
    paired = pd.read_csv(paired_path)
    assoc = pd.read_csv(assoc_path)
    moderators = pd.read_csv(moderator_path)
    predictions = pd.read_csv(prediction_path)
    prediction_summary = pd.read_csv(summary_path).iloc[0]

    panel_a = paired.loc[
        paired.contrast.isin(
            [
                "f_theta_minus_2Hz",
                "f_theta_plus_minus_2Hz",
                "personalized_mean_minus_2Hz",
                "personalized_mean_minus_2Hz_inverse_efficiency",
            ]
        )
    ].copy()
    panel_a["label"] = panel_a.contrast.map(
        {
            "f_theta_minus_2Hz": "fθ RT",
            "f_theta_plus_minus_2Hz": "fθ+ RT",
            "personalized_mean_minus_2Hz": "Personalized mean RT",
            "personalized_mean_minus_2Hz_inverse_efficiency": "Personalized inverse efficiency",
        }
    )
    panel_a["color"] = [BLUE, ORANGE, PURPLE, RED]

    panel_b = assoc.loc[
        assoc.model.eq("continuous_post_rt_ms")
        & assoc.term.isin(["entrainment_between_z", "entrainment_within_z"])
    ].copy()
    panel_b["label"] = panel_b.term.map(
        {
            "entrainment_between_z": "Between-person mean",
            "entrainment_within_z": "Within-person condition deviation",
        }
    )
    panel_b["color"] = [PURPLE, BLUE]

    panel_c = moderators.loc[moderators.predictor.eq("bl_rt")].copy()
    panel_c["label"] = panel_c.outcome.map(
        {
            "personalized_improvement_percent": "Improvement vs own baseline",
            "personalized_advantage_vs_2Hz_ms": "Personalized advantage vs 2 Hz",
            "personalized_advantage_vs_NR_ms": "Personalized advantage vs NR",
        }
    )
    panel_c["color"] = [ORANGE, BLUE, PURPLE]

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_a.to_csv(output_dir / "Figure7A_active_control_effects.csv", index=False)
    panel_b.to_csv(output_dir / "Figure7B_between_within.csv", index=False)
    panel_c.to_csv(output_dir / "Figure7C_moderator_definition.csv", index=False)
    predictions.to_csv(output_dir / "Figure7D_pretreatment_prediction.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(
        "Average benefit, neural coupling, and individual prediction are distinct",
        fontsize=18,
        fontweight="bold",
    )

    ax = axes[0, 0]
    errorbar_forest(
        ax,
        panel_a,
        y_col="label",
        estimate="estimate",
        low="ci95_low",
        high="ci95_high",
        color_col="color",
    )
    ax.set_title("A  Personalized theta versus active 2-Hz control", loc="left", fontweight="bold")
    accuracy = paired.loc[
        paired.contrast.eq("personalized_mean_minus_2Hz_accuracy")
    ].iloc[0]
    ax.set_xlabel(
        "Paired difference (ms; 95% CI)\n"
        "Negative values favor personalized stimulation\n"
        f"Accuracy difference: {accuracy.estimate * 100:.2f} percentage points "
        f"[{accuracy.ci95_low * 100:.2f}, {accuracy.ci95_high * 100:.2f}]"
    )

    ax = axes[0, 1]
    errorbar_forest(
        ax,
        panel_b,
        y_col="label",
        estimate="estimate",
        low="ci95_low",
        high="ci95_high",
        color_col="color",
    )
    ax.set_title("B  Entrainment association decomposed", loc="left", fontweight="bold")
    ax.set_xlabel(
        "Post-RT difference per entrainment SD (ms; 95% CI)\n"
        "Within-person permutation p=0.402"
    )

    ax = axes[1, 0]
    errorbar_forest(
        ax,
        panel_c,
        y_col="label",
        estimate="pearson_r",
        low="ci95_low",
        high="ci95_high",
        color_col="color",
    )
    ax.set_title("C  Baseline-performance moderation depends on comparator", loc="left", fontweight="bold")
    ax.set_xlabel("Pearson r with baseline RT (95% CI)")
    ax.set_xlim(-0.8, 0.8)

    ax = axes[1, 1]
    x = predictions["observed_personalized_advantage_ms"]
    y = predictions["pretreatment_ridge_prediction_ms"]
    limits = [min(x.min(), y.min()) - 4, max(x.max(), y.max()) + 4]
    ax.scatter(x, y, s=42, color=BLUE, alpha=0.78, edgecolor="white", linewidth=0.5)
    ax.plot(limits, limits, color=GREY, linestyle="--", linewidth=1.2)
    ax.axhline(predictions.base_prediction_ms.mean(), color=ORANGE, linewidth=1.4)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.grid(color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("D  Pretreatment individual-benefit prediction", loc="left", fontweight="bold")
    ax.set_xlabel("Observed personalized advantage vs 2 Hz (ms)")
    ax.set_ylabel("Nested-LOSO prediction (ms)")
    ax.text(
        0.03,
        0.97,
        f"MAE: {prediction_summary.base_mae_ms:.2f} → {prediction_summary.model_mae_ms:.2f} ms\n"
        f"Prediction–outcome r={prediction_summary.prediction_outcome_r:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
    )

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)

    png = output_dir / "Figure7_personalized_theta_validation.png"
    pdf = output_dir / "Figure7_personalized_theta_validation.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata=PDF_METADATA)
    plt.close(fig)

    manifest = {
        "figure": "Figure7_personalized_theta_validation",
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (paired_path, assoc_path, moderator_path, prediction_path, summary_path)
        },
        "panels": {
            "A": "paired active-control RT and inverse-efficiency effects",
            "B": "between-person versus within-person entrainment association",
            "C": "baseline moderation across three comparator definitions",
            "D": "nested leave-one-subject-out pretreatment prediction",
        },
    }
    (output_dir / "Figure7_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return [png, pdf]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in build_figure(args.input, args.output):
        print(path)


if __name__ == "__main__":
    main()
