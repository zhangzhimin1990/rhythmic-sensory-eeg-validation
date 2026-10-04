#!/usr/bin/env python3
"""Build source-backed Supplementary Figures S1--S4.

These figures document event reconstruction, scalp-wide response distributions,
measurement sensitivity, and the complete advanced-model gate. They are
measurement-validation assets and do not support an intervention-efficacy claim.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scripts.audit_bids_events import audit_visual
except ModuleNotFoundError:  # Direct execution: python scripts/build_*.py
    from audit_bids_events import audit_visual


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_figures"
VISUAL_ROOT = ROOT / "data" / "public" / "ds006036" / "v1.0.6"

CHANNEL_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
FREQUENCIES = [5.0, 10.0, 15.0, 20.0]
BLUE = "#0072B2"
GOLD = "#E69F00"
RED = "#D55E00"
GREEN = "#009E73"
GREY = "#666666"


SOURCE_PATHS = [
    VISUAL_ROOT / "sub-001/eeg/sub-001_task-photomark_events.tsv",
    ROOT / "outputs/qc/ds006036_full_foundation/occipital_block_metrics.csv",
    ROOT / "outputs/qc/ds006036_full_foundation/channel_frequency_summary.csv",
    ROOT / "outputs/qc/ds005048_full_foundation/channel_response_summary.csv",
    ROOT / "outputs/competition/ds006036_competitor_models/direct_10hz_models.csv",
    ROOT / "outputs/network/ds006036_models/reference_sensitivity.csv",
    ROOT / "outputs/network/ds005048_models/reference_sensitivity.csv",
    ROOT / "outputs/network/ds005048_models/leave_one_out_stability.csv",
    ROOT / "outputs/network/ds006036_models/diagnosis_group_models.csv",
    ROOT / "outputs/network/ds006036_models/patient_severity_models.csv",
    ROOT / "outputs/network/ds006036_models/rest_to_advanced_visual_models.csv",
    ROOT / "outputs/network/ds005048_models/mmse_models.csv",
]


def set_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.2,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12, 1.08, label, transform=axis.transAxes, fontsize=12,
        fontweight="bold", va="top",
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, pdf]


def deterministic_jitter(n: int, width: float = 0.13) -> np.ndarray:
    return np.zeros(n) if n <= 1 else np.linspace(-width, width, n)


def visual_block_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = audit_visual(VISUAL_ROOT)
    blocks = pd.DataFrame(audit["blocks"])
    main = blocks.loc[blocks["frequency_hz"].isin(FREQUENCIES)].copy()
    main["reconstructable"] = main["open_duration_s"].notna() & (main["open_duration_s"] > 0)
    strict = pd.read_csv(SOURCE_PATHS[1])
    strict_n = strict[["subject", "block_index"]].drop_duplicates().shape[0]
    flow = pd.DataFrame(
        [
            {"stage": "Nominal frequency-labelled blocks", "n_blocks": len(blocks)},
            {"stage": "Reconstructable open-eye blocks", "n_blocks": int(blocks["open_duration_s"].notna().sum())},
            {"stage": "Strict 5/10/15/20-Hz signal-analysis blocks", "n_blocks": int(strict_n)},
        ]
    )
    return main, flow


def representative_timeline() -> tuple[pd.DataFrame, dict[str, float]]:
    path = SOURCE_PATHS[0]
    events = pd.read_csv(path, sep="\t")
    events["value"] = events["value"].astype(str).str.strip()
    labels = events.loc[events["value"].str.match(r"PHOTO\s+\d+(?:\.\d+)?Hz")]
    target = labels.loc[labels["value"].eq("PHOTO 10Hz")].iloc[0]
    label_onset = float(target["onset"])
    later_labels = labels.loc[labels["onset"] > label_onset, "onset"]
    next_label = float(later_labels.min()) if len(later_labels) else float(events["onset"].max() + 1)
    window = events.loc[(events["onset"] >= label_onset - 0.05) & (events["onset"] < next_label)].copy()
    open_onset = float(window.loc[window["value"].eq("open eyes"), "onset"].iloc[0])
    closed_onset = float(window.loc[window["value"].eq("closed eyes"), "onset"].iloc[0])
    window["relative_onset_s"] = window["onset"] - label_onset
    values = {
        "label": 0.0,
        "open": open_onset - label_onset,
        "analysis_start": open_onset - label_onset + 0.5,
        "analysis_stop": closed_onset - label_onset - 0.1,
        "closed": closed_onset - label_onset,
    }
    return window, values


def build_figure_s1(output_dir: Path) -> list[Path]:
    blocks, flow = visual_block_tables()
    timeline, timing = representative_timeline()
    blocks.to_csv(output_dir / "FigureS1_event_blocks.csv", index=False)
    flow.to_csv(output_dir / "FigureS1_flow.csv", index=False)
    timeline.to_csv(output_dir / "FigureS1_representative_timeline.csv", index=False)

    fig = plt.figure(figsize=(13.2, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.72, 1.0], width_ratios=[1.2, 1.0])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    right = grid[1, 1].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.38)
    ax_c = fig.add_subplot(right[0, 0])
    ax_d = fig.add_subplot(right[0, 1])

    pulse_times = timeline.loc[timeline["value"].eq("Photo/HV mark"), "relative_onset_s"].to_numpy()
    if len(pulse_times) > 1:
        large_gaps = np.flatnonzero(np.diff(pulse_times) > 0.5)
        if len(large_gaps):
            pulse_times = pulse_times[: large_gaps[0] + 1]
    ax_a.eventplot(pulse_times, lineoffsets=0.74, linelengths=0.28, colors="#8A8A8A", linewidths=0.75)
    ax_a.axvspan(timing["open"], timing["closed"], color=GOLD, alpha=0.14, label="Annotated eyes-open exposure")
    ax_a.axvspan(timing["analysis_start"], timing["analysis_stop"], color=BLUE, alpha=0.28, label="Strict analysis window")
    for key, color, y in [("label", GREY, 0.18), ("open", GREEN, 0.36), ("closed", RED, 0.36)]:
        ax_a.axvline(timing[key], color=color, lw=1.4, ls="--" if key != "label" else "-")
        label = {"label": "10-Hz label", "open": "Open eyes", "closed": "Closed eyes"}[key]
        ax_a.text(timing[key], y, f"{label}\n{timing[key]:.3f} s", color=color, ha="center", va="bottom", fontsize=7.4)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Time from frequency label (s)")
    ax_a.set_title("Representative event reconstruction: sub-001, 10-Hz block")
    ax_a.legend(loc="upper right", frameon=False, ncol=2)
    panel_label(ax_a, "A")

    usable = blocks.loc[blocks["reconstructable"]].copy()
    sns.boxplot(data=usable, x="frequency_hz", y="open_duration_s", order=FREQUENCIES,
                color="#DCEAF4", width=0.50, fliersize=0, linewidth=0.9, ax=ax_b)
    for index, frequency in enumerate(FREQUENCIES):
        values = usable.loc[usable["frequency_hz"].eq(frequency), "open_duration_s"].sort_values()
        ax_b.scatter(index + deterministic_jitter(len(values)), values, s=11, facecolor="white",
                     edgecolor=BLUE, linewidth=0.45, alpha=0.75, zorder=3)
        ax_b.text(index, ax_b.get_ylim()[1], f"n={len(values)}", ha="center", va="top", fontsize=7)
    ax_b.set(title="Reconstructed eyes-open duration", xlabel="Stimulation frequency (Hz)", ylabel="Duration (s)")
    panel_label(ax_b, "B")

    sns.boxplot(data=usable, x="frequency_hz", y="open_after_stimulus_label_s", order=FREQUENCIES,
                color="#F8E6C2", width=0.50, fliersize=0, linewidth=0.9, ax=ax_c)
    for index, frequency in enumerate(FREQUENCIES):
        values = usable.loc[usable["frequency_hz"].eq(frequency), "open_after_stimulus_label_s"].sort_values()
        ax_c.scatter(index + deterministic_jitter(len(values)), values, s=10, facecolor="white",
                     edgecolor=GOLD, linewidth=0.45, alpha=0.75, zorder=3)
    ax_c.set(title="Label-to-open delay", xlabel="Frequency (Hz)", ylabel="Delay (s)")
    ax_c.text(-0.30, 1.10, "C", transform=ax_c.transAxes, fontsize=12,
              fontweight="bold", va="top")
    ax_c.tick_params(axis="x", labelrotation=35)

    stages = ["Labelled", "Open-eye", "Analyzed"]
    bars = ax_d.bar(np.arange(len(flow)), flow["n_blocks"], color=["#BDBDBD", GOLD, BLUE], width=0.68)
    for bar, value in zip(bars, flow["n_blocks"]):
        ax_d.text(bar.get_x() + bar.get_width()/2, value + 5, str(value), ha="center", va="bottom", fontweight="bold")
    ax_d.set_xticks(np.arange(len(flow)), stages, rotation=20, ha="right")
    ax_d.set_ylim(0, max(flow["n_blocks"]) * 1.16)
    ax_d.set(title="Block-level analysis flow", ylabel="Blocks")
    ax_d.text(-0.38, 1.10, "D", transform=ax_d.transAxes, fontsize=12,
              fontweight="bold", va="top")
    ax_d.text(0.5, -0.30, "Counts refer to blocks, not participants", transform=ax_d.transAxes,
              ha="center", va="top", fontsize=7.0, color=GREY)
    return save_figure(fig, output_dir, "FigureS1_visual_event_reconstruction")


def build_figure_s2(output_dir: Path) -> list[Path]:
    visual = pd.read_csv(SOURCE_PATHS[2])
    auditory = pd.read_csv(SOURCE_PATHS[3])
    visual.to_csv(output_dir / "FigureS2_visual_channel_summary.csv", index=False)
    auditory.to_csv(output_dir / "FigureS2_auditory_channel_summary.csv", index=False)

    visual_matrix = visual.pivot(index="channel", columns="frequency_hz", values="median_snr_db")
    order = [channel for channel in CHANNEL_ORDER if channel in visual_matrix.index]
    visual_matrix = visual_matrix.reindex(index=order, columns=FREQUENCIES)
    auditory = auditory.set_index("channel").reindex([c for c in CHANNEL_ORDER if c in set(auditory["channel"])]).reset_index()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.8, 7.4), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1.0, 1.22]})
    cmap = sns.diverging_palette(250, 20, s=85, l=48, as_cmap=True)
    sns.heatmap(visual_matrix, cmap=cmap, center=0, annot=True, fmt=".1f", annot_kws={"fontsize": 6.2},
                cbar_kws={"label": "Median local SNR (dB)", "shrink": 0.72}, linewidths=0.35,
                linecolor="white", ax=ax_a)
    ax_a.set(title="Visual response across the scalp", xlabel="Stimulation frequency (Hz)", ylabel="Channel")
    for tick in ax_a.get_yticklabels():
        if tick.get_text() in {"O1", "O2"}:
            tick.set_color(RED)
            tick.set_fontweight("bold")
    panel_label(ax_a, "A")

    y = np.arange(len(auditory))
    ax_b.hlines(y, auditory["q25"], auditory["q75"], color="#86B8D5", lw=3.0, alpha=0.9)
    ax_b.scatter(auditory["median"], y, color=BLUE, s=28, zorder=3, edgecolor="white", linewidth=0.4)
    ax_b.axvline(0, color="#555555", lw=0.9, ls="--")
    ax_b.set_yticks(y, auditory["channel"])
    ax_b.invert_yaxis()
    ax_b.set(title="Auditory 40-Hz response across the scalp", xlabel="Stimulus − rest local SNR (dB)", ylabel="Channel")
    ax_b.text(0.98, 0.02, "Dot: median\nLine: IQR", transform=ax_b.transAxes, ha="right", va="bottom", fontsize=7.3)
    for tick in ax_b.get_yticklabels():
        if tick.get_text() in {"Fz", "Cz"}:
            tick.set_color(GREEN)
            tick.set_fontweight("bold")
    panel_label(ax_b, "B")
    return save_figure(fig, output_dir, "FigureS2_full_channel_response_distributions")


def build_figure_s3(output_dir: Path) -> list[Path]:
    direct = pd.read_csv(SOURCE_PATHS[4])
    visual_ref = pd.read_csv(SOURCE_PATHS[5])
    auditory_ref = pd.read_csv(SOURCE_PATHS[6])
    loo = pd.read_csv(SOURCE_PATHS[7])
    direct.to_csv(output_dir / "FigureS3_scale_sensitivity.csv", index=False)
    visual_ref.to_csv(output_dir / "FigureS3_visual_reference_sensitivity.csv", index=False)
    auditory_ref.to_csv(output_dir / "FigureS3_auditory_reference_sensitivity.csv", index=False)
    loo.to_csv(output_dir / "FigureS3_auditory_leave_one_out.csv", index=False)

    fig = plt.figure(figsize=(14.5, 10.4), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.07, top=0.96, wspace=0.30, hspace=0.34)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.15], width_ratios=[1.22, 1.0],
                            wspace=0.28, hspace=0.34)
    forest_grid = grid[0, :].subgridspec(1, 3, wspace=0.38)
    outcomes = [("raw_di", "Raw DI", "DI units"), ("log1p_di", "log(1 + DI)", "Log units"), ("winsor_di", "Winsorized DI", "DI units")]
    contrasts = ["AD-CN", "FTD-CN"]
    forest_axes = []
    for index, (outcome, title, xlabel) in enumerate(outcomes):
        ax = fig.add_subplot(forest_grid[0, index])
        frame = direct.loc[direct["outcome"].eq(outcome)].set_index("contrast").reindex(contrasts)
        y = np.arange(len(frame))
        ax.errorbar(frame["coefficient"], y,
                    xerr=np.vstack([frame["coefficient"] - frame["ci_low"], frame["ci_high"] - frame["coefficient"]]),
                    fmt="o", color=[BLUE, RED][index % 2], ecolor="#666666", capsize=3, ms=5)
        ax.axvline(0, color="#444444", lw=0.9, ls="--")
        ax.set_yticks(y, contrasts if index == 0 else ["", ""])
        ax.invert_yaxis()
        ax.set(title=title, xlabel=xlabel)
        forest_axes.append(ax)
    panel_label(forest_axes[0], "A")
    forest_axes[0].text(-0.12, 1.01, "Separate axes: scale-dependent estimates", transform=forest_axes[0].transAxes,
                        fontsize=7.2, color=GREY)

    ax_b = fig.add_subplot(grid[1, 0])
    visual_ref["row"] = visual_ref["metric"].map({
        "posterior_frontal_abs_imcoh": "|ImCoh|",
        "posterior_frontal_dwpli2": "dwPLI²",
    }) + " · " + visual_ref["comparison"].str.replace("_", " ")
    vm = visual_ref.pivot(index="row", columns="frequency_hz", values="spearman_rho").reindex(columns=FREQUENCIES)
    sns.heatmap(vm, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"fontsize": 6.6}, linewidths=0.4, cbar=False, ax=ax_b)
    ax_b.set(title="Visual reference sensitivity (Spearman ρ; −1 to 1)", xlabel="Stimulation frequency (Hz)", ylabel="Metric · reference comparison")
    panel_label(ax_b, "B")

    right = grid[1, 1].subgridspec(2, 1, height_ratios=[0.70, 1.30], hspace=0.42)
    ax_c = fig.add_subplot(right[0, 0])
    auditory_ref["row"] = auditory_ref["metric"].map({
        "central_frontal_abs_imcoh": "|ImCoh|",
        "central_frontal_dwpli2": "dwPLI²",
    })
    am = auditory_ref.pivot(index="row", columns="comparison", values="spearman_rho")
    am = am.reindex(columns=["average_vs_native", "average_vs_csd", "native_vs_csd"])
    am.columns = ["Avg–native", "Avg–CSD", "Native–CSD"]
    sns.heatmap(am, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7}, linewidths=0.4, cbar=False, ax=ax_c)
    ax_c.set(title="Auditory reference sensitivity (Spearman ρ)", xlabel="Reference comparison", ylabel="Metric")
    panel_label(ax_c, "C")

    loo_grid = right[1, 0].subgridspec(2, 1, hspace=0.95)
    loo_labels = {
        "central_minus_frontal_snr_db": "Central − frontal SNR",
        "csd_central_frontal_abs_imcoh": "CSD |ImCoh|",
    }
    loo_axes = []
    for index, row in loo.reset_index(drop=True).iterrows():
        ax = fig.add_subplot(loo_grid[index, 0])
        ax.hlines(0, row["leave_one_out_min"], row["leave_one_out_max"], color="#888888", lw=5)
        ax.scatter(row["full_coefficient"], 0, s=42, color=RED, edgecolor="white", linewidth=0.5, zorder=3)
        ax.axvline(0, color="#444444", lw=0.9, ls="--")
        ax.set_yticks([])
        ax.set_ylim(-0.7, 0.7)
        ax.set(title=loo_labels[row["outcome"]], xlabel="MMSE coefficient")
        loo_axes.append(ax)
    loo_axes[0].text(-0.14, 1.18, "D", transform=loo_axes[0].transAxes,
                     fontsize=12, fontweight="bold", va="top")
    loo_axes[0].text(0.98, 0.90, "Grey: leave-one-out range\nOrange: full estimate",
                     transform=loo_axes[0].transAxes, ha="right", va="top", fontsize=6.8, color=GREY)
    return save_figure(fig, output_dir, "FigureS3_scale_reference_influence_sensitivity")


def q_family_summary() -> pd.DataFrame:
    group = pd.read_csv(SOURCE_PATHS[8])
    severity = pd.read_csv(SOURCE_PATHS[9])
    rest = pd.read_csv(SOURCE_PATHS[10])
    auditory = pd.read_csv(SOURCE_PATHS[11])
    rows = []
    for family, frame in group.groupby("family"):
        rows.append({"analysis_family": f"Visual diagnosis · {family.replace('group_', '')}", "minimum_q_bh": frame["q_bh"].min(), "n_models": len(frame)})
    for family, frame in severity.groupby("family"):
        rows.append({"analysis_family": f"Visual severity · {family.replace('severity_', '')}", "minimum_q_bh": frame["q_bh"].min(), "n_models": len(frame)})
    rows.append({"analysis_family": "Rest → advanced visual", "minimum_q_bh": rest["q_bh"].min(), "n_models": len(rest)})
    for family, frame in auditory.groupby("family"):
        rows.append({"analysis_family": f"Auditory MMSE · {family}", "minimum_q_bh": frame["q_bh"].min(), "n_models": len(frame)})
    incremental = auditory.loc[(auditory["family"].eq("exploratory")) & (auditory["model"].eq("incremental_over_local_snr"))]
    rows.append({"analysis_family": "Auditory exploratory · incremental subset", "minimum_q_bh": incremental["q_bh"].min(), "n_models": len(incremental)})
    return pd.DataFrame(rows)


def _q_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame["family"].str.replace("group_", "", regex=False).str.replace("severity_", "", regex=False) + " · " + frame["outcome"].str.replace("_", " ")
    return frame.assign(row=labels).pivot(index="row", columns="frequency_hz", values="q_bh").reindex(columns=FREQUENCIES)


def build_figure_s4(output_dir: Path) -> list[Path]:
    group = pd.read_csv(SOURCE_PATHS[8])
    severity = pd.read_csv(SOURCE_PATHS[9])
    rest = pd.read_csv(SOURCE_PATHS[10])
    family = q_family_summary().sort_values("minimum_q_bh").reset_index(drop=True)
    family.to_csv(output_dir / "FigureS4_family_minimum_q.csv", index=False)
    group.to_csv(output_dir / "FigureS4_visual_diagnosis_models.csv", index=False)
    severity.to_csv(output_dir / "FigureS4_visual_severity_models.csv", index=False)
    rest.to_csv(output_dir / "FigureS4_rest_to_advanced_models.csv", index=False)

    fig = plt.figure(figsize=(15.6, 10.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[0.86, 1.14], height_ratios=[0.92, 1.08])
    ax_a = fig.add_subplot(grid[0, 0])
    y = np.arange(len(family))
    ax_a.scatter(family["minimum_q_bh"], y, s=48, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    ax_a.axvline(0.05, color=RED, lw=1.1, ls="--", label="FDR q = 0.05")
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.04, 1.08)
    ax_a.set_yticks(y, family["analysis_family"])
    ax_a.invert_yaxis()
    ax_a.set(title="Minimum adjusted q within each frozen family", xlabel="Minimum Benjamini–Hochberg q", ylabel="")
    ax_a.legend(frameon=False, loc="lower right")
    for yi, row in family.iterrows():
        ax_a.text(row["minimum_q_bh"] * 1.04, yi, f"{row['minimum_q_bh']:.3f} (m={int(row['n_models'])})", va="center", fontsize=6.9)
    panel_label(ax_a, "A")

    cmap = sns.light_palette(BLUE, as_cmap=True)
    ax_b = fig.add_subplot(grid[0, 1])
    sns.heatmap(_q_matrix(group), cmap=cmap, vmin=0, vmax=1, annot=True, fmt=".2f",
                annot_kws={"fontsize": 6.3}, linewidths=0.4, cbar_kws={"label": "BH q", "shrink": 0.72}, ax=ax_b)
    ax_b.set(title="Visual diagnosis-model families", xlabel="Stimulation frequency (Hz)", ylabel="Family · outcome")
    panel_label(ax_b, "B")

    ax_c = fig.add_subplot(grid[1, 0])
    sns.heatmap(_q_matrix(severity), cmap=cmap, vmin=0, vmax=1, annot=True, fmt=".2f",
                annot_kws={"fontsize": 6.1}, linewidths=0.4, cbar_kws={"label": "BH q", "shrink": 0.72}, ax=ax_c)
    ax_c.set(title="Visual patient-severity families", xlabel="Frequency (Hz)", ylabel="Family · outcome")
    panel_label(ax_c, "C")

    ax_d = fig.add_subplot(grid[1, 1])
    rest_labels = rest["outcome"].map({
        "posterior_minus_frontal_snr_db": "Post−front SNR",
        "csd_posterior_frontal_abs_imcoh": "CSD |ImCoh|",
    }) + " · " + rest["predictor"].map({
        "posterior_iaf_hz": "IAF",
        "slowing_ratio": "slowing",
        "spectral_exponent": "exponent",
        "relative_alpha": "rel. alpha",
    })
    rm = rest.assign(row=rest_labels).pivot(index="row", columns="frequency_hz", values="q_bh").reindex(columns=FREQUENCIES)
    sns.heatmap(rm, cmap=cmap, vmin=0, vmax=1, annot=True, fmt=".2f",
                annot_kws={"fontsize": 5.9}, linewidths=0.4, cbar_kws={"label": "BH q", "shrink": 0.72}, ax=ax_d)
    ax_d.set(title="Resting features → advanced visual measures", xlabel="Frequency (Hz)", ylabel="Outcome · resting predictor")
    panel_label(ax_d, "D")
    return save_figure(fig, output_dir, "FigureS4_advanced_model_family_gate")


def write_manifest(output_dir: Path) -> Path:
    rows = []
    for path in SOURCE_PATHS + [Path(__file__)]:
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    destination = output_dir / "Supplementary_Figures_source_manifest_sha256.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    generated = []
    generated.extend(build_figure_s1(output_dir))
    generated.extend(build_figure_s2(output_dir))
    generated.extend(build_figure_s3(output_dir))
    generated.extend(build_figure_s4(output_dir))
    generated.append(write_manifest(output_dir))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
