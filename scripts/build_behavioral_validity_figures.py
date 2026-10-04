#!/usr/bin/env python3
"""Build source-backed manuscript Figures 5 and 6 for the public-data v0.2 route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2864A8"
GOLD = "#D99A24"
ORANGE = "#C85A32"
PURPLE = "#7B5AA6"
CHARCOAL = "#333333"
LIGHT = "#D9DEE6"
PDF_METADATA = {
    "Creator": "public-eeg-validation",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.14, 1.08, label, transform=axis.transAxes, fontsize=13, fontweight="bold")


def interval_plot(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    label_column: str,
    estimate_column: str = "estimate",
    low_column: str = "ci_low",
    high_column: str = "ci_high",
    colors: list[str] | None = None,
    xlabel: str,
    title: str,
) -> None:
    positions = np.arange(len(frame))[::-1]
    palette = colors or [BLUE] * len(frame)
    for position, (_, row), color in zip(positions, frame.iterrows(), palette):
        axis.plot([row[low_column], row[high_column]], [position, position], color=color, lw=2)
        axis.scatter(row[estimate_column], position, s=42, color=color, edgecolor="white", zorder=3)
    axis.axvline(0, color="#666666", lw=1, ls="--")
    axis.set_yticks(positions, frame[label_column])
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left")
    axis.grid(axis="x", color="#E8E8E8", lw=0.7)


def load_figure5_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(
        ROOT / "outputs/models/ds007648_behavioral_validity/clustered_models.csv"
    )
    selected = models.loc[
        models.specification.eq("main")
        & models.feature_family.eq("phase_projected")
        & models.term.isin(["hz36_average_contrast", "hz40_average_contrast"])
    ].copy()
    accuracy = selected.loc[selected.analysis.eq("accuracy")].copy()
    rt = selected.loc[selected.analysis.eq("rt")].copy()
    label_map = {"hz36_average_contrast": "36 Hz", "hz40_average_contrast": "40 Hz"}
    accuracy["label"] = accuracy.term.map(label_map)
    rt["label"] = rt.term.map(label_map)

    reliability = pd.read_csv(
        ROOT / "outputs/models/ds007648_behavioral_validity/split_half_reliability.csv"
    )
    reliability = reliability.loc[reliability.measure.isin(["hz36_itpc", "hz40_itpc"])].copy()
    reliability["frequency_hz"] = reliability.measure.str.extract(r"hz(\d+)").astype(int)
    validity = accuracy.copy()
    validity["frequency_hz"] = validity.term.str.extract(r"hz(\d+)").astype(int)
    scatter = reliability.merge(
        validity[["frequency_hz", "estimate", "ci_low", "ci_high"]],
        on="frequency_hz",
        validate="one_to_one",
    )

    dprime = pd.concat(
        [
            pd.read_csv(ROOT / "outputs/models/ds006780_behavioral_validity/primary_tests.csv"),
            pd.read_csv(ROOT / "outputs/models/ds006780_behavioral_validity/sensitivity_tests.csv"),
        ],
        ignore_index=True,
    )
    dprime = dprime.loc[dprime.outcome.eq("dprime_40")].copy()
    family_labels = {
        "local_log_snr": "Local log-SNR",
        "morlet_power": "Morlet power",
        "itpc": "ITPC",
    }
    dprime["label"] = dprime.feature_family.map(family_labels)
    return accuracy, rt, scatter, dprime


def build_figure5(output: Path) -> list[Path]:
    accuracy, rt, scatter, dprime = load_figure5_data()
    accuracy.to_csv(output / "Figure5A_ds007648_accuracy.csv", index=False)
    rt.to_csv(output / "Figure5B_ds007648_rt.csv", index=False)
    scatter.to_csv(output / "Figure5C_reliability_validity.csv", index=False)
    dprime.to_csv(output / "Figure5D_ds006780_dprime.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    interval_plot(
        axes[0, 0], accuracy.sort_values("label"), label_column="label",
        colors=[GOLD, BLUE], xlabel="Log-odds per within-subject SD (95% CI)",
        title="ds007648: accuracy association",
    )
    panel_label(axes[0, 0], "A")
    axes[0, 0].text(
        0.03, 0.50, "36 Hz: BH q=0.0039\n40 Hz: BH q=0.353",
        transform=axes[0, 0].transAxes, va="center", fontsize=8,
    )

    interval_plot(
        axes[0, 1], rt.sort_values("label"), label_column="label",
        colors=[GOLD, BLUE], xlabel="Log-RT per within-subject SD (95% CI)",
        title="ds007648: correct-trial RT association",
    )
    panel_label(axes[0, 1], "B")

    axis = axes[1, 0]
    for _, row in scatter.sort_values("frequency_hz").iterrows():
        color = GOLD if row.frequency_hz == 36 else BLUE
        axis.plot(
            [row.spearman_brown_full_length, row.spearman_brown_full_length],
            [row.ci_low, row.ci_high], color=color, lw=2,
        )
        axis.scatter(
            row.spearman_brown_full_length, row.estimate, s=65, color=color,
            edgecolor="white", zorder=3,
        )
        axis.annotate(
            f"{int(row.frequency_hz)} Hz", (row.spearman_brown_full_length, row.estimate),
            xytext=(6, 5), textcoords="offset points", color=color, fontweight="bold",
        )
    axis.axhline(0, color="#666666", lw=1, ls="--")
    axis.set(
        xlim=(0.83, 1.0), xlabel="Full-length split-half reliability",
        ylabel="Accuracy log-odds / within-subject SD",
        title="Reliability and behavioral validity are distinct",
    )
    axis.grid(color="#E8E8E8", lw=0.7)
    panel_label(axis, "C")

    interval_plot(
        axes[1, 1], dprime, label_column="label",
        colors=[BLUE, GOLD, PURPLE], xlabel="d-prime units per neural-feature SD (95% CI)",
        title="ds006780: adjusted 40-Hz behavioral validity",
    )
    axes[1, 1].text(
        0.03, 0.73, "Adjusted for 27-Hz response, age, sex, and group\nn=111",
        transform=axes[1, 1].transAxes, va="center", fontsize=8,
    )
    panel_label(axes[1, 1], "D")
    fig.suptitle("Concurrent behavioral validity of rhythmic sensory EEG responses", fontsize=13, fontweight="bold")
    paths = [output / "Figure5_behavioral_validity.png", output / "Figure5_behavioral_validity.pdf"]
    fig.savefig(paths[0], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", metadata=PDF_METADATA)
    plt.close(fig)
    return paths


def validation_matrix() -> pd.DataFrame:
    stages = ["Detectable", "Reliable", "Sensitive", "Concurrent valid", "Predictive", "Causal/clinical"]
    values = {
        "ds006036 visual": ["Supported", "Limited", "Supported", "Limited", "No increment", "Not tested"],
        "ds005048 auditory": ["Supported", "Supported", "Supported", "Limited", "No increment", "Not tested"],
        "ds007648 audiovisual": ["Supported", "Supported", "Conditional", "Conditional", "No increment", "Not tested"],
        "ds006780 ASSR": ["Supported", "Supported", "Supported", "Limited", "No increment", "Not tested"],
    }
    rows = []
    for dataset, states in values.items():
        for stage, state in zip(stages, states):
            rows.append({"dataset": dataset, "stage": stage, "state": state})
    return pd.DataFrame(rows)


def load_figure6_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = pd.read_csv(ROOT / "outputs/models/ds006780_behavioral_validity/primary_tests.csv")
    sensitivity = pd.read_csv(ROOT / "outputs/models/ds006780_behavioral_validity/sensitivity_tests.csv")
    fsiq = pd.concat([primary, sensitivity], ignore_index=True)
    fsiq = fsiq.loc[fsiq.outcome.eq("fsiq")].copy()
    fsiq["label"] = fsiq.feature_family.map(
        {"local_log_snr": "Local log-SNR", "morlet_power": "Morlet power", "itpc": "ITPC"}
    )

    robustness = pd.read_csv(
        ROOT / "outputs/models/ds006780_behavioral_validity/fsiq_audit/robustness_models.csv"
    )
    robustness["label"] = robustness.variant.map(
        {
            "main": "Full release (main)",
            "without_27hz_control": "No 27-Hz control",
            "quadratic_age": "Quadratic age",
            "signal_quality_adjusted": "Signal-quality adjusted",
            "winsorized_2.5_percent": "Winsorized 2.5%",
            "published_age_8_to_12": "Age 8 to <13",
            "published_fsiq_above_80": "FSIQ >80",
            "published_age_and_fsiq": "Age + FSIQ eligible",
            "published_behavior_threshold": "Published behavior rule",
        }
    )
    robustness = robustness.rename(columns={"estimate_40_per_sd": "estimate"})

    prediction = pd.read_csv(
        ROOT / "outputs/models/validation_strength/prediction_increment_precision.csv"
    )
    prediction["relative_delta_percent"] = 100 * prediction.delta_neural_minus_base / prediction.mean_base
    prediction["relative_ci_low_percent"] = 100 * prediction.ci_low / prediction.mean_base
    prediction["relative_ci_high_percent"] = 100 * prediction.ci_high / prediction.mean_base
    prediction["label"] = (
        prediction.dataset + " · " + prediction.outcome + " · " + prediction.metric
    )
    matrix = validation_matrix()
    return fsiq, robustness, prediction, matrix


def build_figure6(output: Path) -> list[Path]:
    fsiq, robustness, prediction, matrix = load_figure6_data()
    fsiq.to_csv(output / "Figure6A_fsiq_metrics.csv", index=False)
    robustness.to_csv(output / "Figure6B_fsiq_eligibility.csv", index=False)
    prediction.to_csv(output / "Figure6C_prediction_increment.csv", index=False)
    matrix.to_csv(output / "Figure6D_validation_matrix.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    interval_plot(
        axes[0, 0], fsiq, label_column="label", colors=[BLUE, GOLD, PURPLE],
        xlabel="FSIQ points per neural-feature SD (95% CI)",
        title="ds006780: frequency-conditioned FSIQ association",
    )
    axes[0, 0].text(
        0.97, 0.73, "Adjusted for 27-Hz response, age, sex, and group\nn=109",
        transform=axes[0, 0].transAxes, ha="right", va="center", fontsize=8,
    )
    panel_label(axes[0, 0], "A")

    ordered = robustness.reset_index(drop=True)
    eligibility = ordered.variant.isin(["published_fsiq_above_80", "published_age_and_fsiq"])
    interval_plot(
        axes[0, 1], ordered, label_column="label", estimate_column="estimate",
        colors=[ORANGE if flag else BLUE for flag in eligibility],
        xlabel="FSIQ points per 40-Hz log-SNR SD (95% CI)",
        title="Eligibility and model sensitivity",
    )
    panel_label(axes[0, 1], "B")

    pred = prediction.sort_values(["dataset", "outcome", "metric"]).reset_index(drop=True)
    interval_plot(
        axes[1, 0], pred, label_column="label", estimate_column="relative_delta_percent",
        low_column="relative_ci_low_percent", high_column="relative_ci_high_percent",
        colors=[BLUE if dataset == "ds007648" else GOLD for dataset in pred.dataset],
        xlabel="Neural − base prediction error (% of base; 95% CI)",
        title="Out-of-sample prediction increment\n(Negative values favor the neural model)",
    )
    panel_label(axes[1, 0], "C")

    state_order = ["Not tested", "No increment", "Limited", "Conditional", "Supported"]
    state_code = {state: index for index, state in enumerate(state_order)}
    stage_order = ["Detectable", "Reliable", "Sensitive", "Concurrent valid", "Predictive", "Causal/clinical"]
    dataset_order = ["ds006036 visual", "ds005048 auditory", "ds007648 audiovisual", "ds006780 ASSR"]
    pivot = matrix.pivot(index="dataset", columns="stage", values="state").loc[dataset_order, stage_order]
    encoded = pivot.apply(lambda column: column.map(state_code)).astype(int)
    cmap = ListedColormap(["#F2F2F2", "#D8DCE2", "#F4D6A3", "#E8B86A", "#5B83B4"])
    axes[1, 1].imshow(encoded.to_numpy(), cmap=cmap, vmin=-0.5, vmax=4.5, aspect="auto")
    axes[1, 1].set_xticks(np.arange(len(stage_order)), stage_order, rotation=35, ha="right")
    axes[1, 1].set_yticks(np.arange(len(dataset_order)), dataset_order)
    for row in range(len(dataset_order)):
        for column in range(len(stage_order)):
            state = pivot.iloc[row, column]
            axes[1, 1].text(
                column, row, state.replace("No increment", "No\nincrement").replace("Not tested", "Not\ntested"),
                ha="center", va="center", fontsize=7,
                color="white" if state == "Supported" else CHARCOAL,
            )
    axes[1, 1].set_title("Validation level is context-specific", loc="left")
    axes[1, 1].tick_params(length=0)
    for spine in axes[1, 1].spines.values():
        spine.set_visible(False)
    panel_label(axes[1, 1], "D")

    fig.suptitle("Association, eligibility, and predictive value", fontsize=13, fontweight="bold")
    paths = [output / "Figure6_transportability_prediction.png", output / "Figure6_transportability_prediction.pdf"]
    fig.savefig(paths[0], bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white", metadata=PDF_METADATA)
    plt.close(fig)
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/manuscript_figures_v02"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_style()
    paths = build_figure5(args.output) + build_figure6(args.output)

    source_paths = [
        ROOT / "outputs/models/ds007648_behavioral_validity/clustered_models.csv",
        ROOT / "outputs/models/ds007648_behavioral_validity/split_half_reliability.csv",
        ROOT / "outputs/models/ds006780_behavioral_validity/primary_tests.csv",
        ROOT / "outputs/models/ds006780_behavioral_validity/sensitivity_tests.csv",
        ROOT / "outputs/models/ds006780_behavioral_validity/fsiq_audit/robustness_models.csv",
        ROOT / "outputs/models/validation_strength/prediction_increment_precision.csv",
    ]
    manifest = {
        "figures": [str(path.relative_to(ROOT)) for path in paths],
        "sources": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in source_paths
        ],
        "notes": [
            "All intervals are 95% confidence intervals unless stated otherwise.",
            "Cross-dataset raw EEG amplitudes are not pooled.",
            "Validation matrix states are claim-level summaries, not effect sizes.",
        ],
    }
    (args.output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
