#!/usr/bin/env python3
"""Build source-backed manuscript Figures 4–6 and their audit tables.

The figures deliberately separate datasets, states, response definitions, and
measurement levels. They are descriptive/measurement-validation assets; they
do not support an intervention-efficacy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_figures"

COLORS = {
    "CN": "#0072B2",
    "MCI": "#E69F00",
    "AD": "#D55E00",
    "FTD": "#009E73",
    "overall": "#4D4D4D",
    "light": "#D9D9D9",
    "blue": "#0072B2",
    "gold": "#E69F00",
}

GROUP_MAP = {"A": "AD", "C": "CN", "F": "FTD"}
PREDICTORS = [
    "posterior_iaf_hz",
    "iaf_distance_to_10hz",
    "slowing_ratio",
    "spectral_exponent",
    "relative_delta",
    "relative_theta",
    "relative_alpha",
]
PREDICTOR_LABELS = {
    "posterior_iaf_hz": "Posterior IAF",
    "iaf_distance_to_10hz": "|IAF − 10 Hz|",
    "slowing_ratio": "Slowing ratio",
    "spectral_exponent": "Spectral exponent",
    "relative_delta": "Relative delta",
    "relative_theta": "Relative theta",
    "relative_alpha": "Relative alpha",
}


def set_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
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
        -0.13,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def deterministic_jitter(n: int, width: float = 0.14) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = 10_000, seed: int = 20260919
) -> tuple[float, float, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[keep]
    y = np.asarray(y)[keep]
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(x), len(x))
        value = spearmanr(x[indices], y[indices]).statistic
        if np.isfinite(value):
            estimates.append(float(value))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return observed, float(low), float(high)


def fit_rest_to_visual_models() -> pd.DataFrame:
    paired_path = ROOT / "outputs/qc/ds004504_full_foundation/paired_rest_visual_features.csv"
    participants_path = ROOT / "data/public/ds004504/v1.0.9_derivatives/participants.tsv"
    paired = pd.read_csv(paired_path)
    participants = (
        pd.read_csv(participants_path, sep="\t")
        .rename(columns={"participant_id": "subject", "Gender": "sex"})[["subject", "sex"]]
    )
    data = paired.merge(participants, on="subject", how="left", validate="many_to_one")
    rows: list[dict[str, float | int | str]] = []
    for frequency_hz, frequency_frame in data.groupby("frequency_hz", sort=True):
        for predictor in PREDICTORS:
            frame = frequency_frame[
                ["local_snr_db", predictor, "age", "sex", "group_rest"]
            ].dropna()
            frame = frame.copy()
            frame["predictor_z"] = (
                frame[predictor] - frame[predictor].mean()
            ) / frame[predictor].std(ddof=0)
            frame["age_z"] = (frame["age"] - frame["age"].mean()) / frame["age"].std(ddof=0)
            model = smf.ols(
                "local_snr_db ~ predictor_z + age_z + C(sex) + C(group_rest)",
                data=frame,
            ).fit(cov_type="HC3")
            estimate = float(model.params["predictor_z"])
            standard_error = float(model.bse["predictor_z"])
            rows.append(
                {
                    "frequency_hz": float(frequency_hz),
                    "predictor": predictor,
                    "n": len(frame),
                    "coefficient_db_per_predictor_sd": estimate,
                    "standard_error": standard_error,
                    "ci_low": estimate - 1.96 * standard_error,
                    "ci_high": estimate + 1.96 * standard_error,
                    "p_value": float(model.pvalues["predictor_z"]),
                    "r_squared": float(model.rsquared),
                }
            )
    result = pd.DataFrame(rows)
    result["q_bh"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def build_figure4(output_dir: Path) -> list[Path]:
    visual = pd.read_csv(
        ROOT / "outputs/qc/ds006036_full_foundation/subject_frequency_summary.csv"
    )
    exact = pd.read_csv(ROOT / "outputs/competition/ds006036_exact_preprint/features_ica.csv")
    direct = pd.read_csv(
        ROOT / "outputs/competition/ds006036_competitor_models/direct_10hz_models.csv"
    )
    selectivity = pd.read_csv(
        ROOT / "outputs/competition/ds006036_competitor_models/selectivity_models.csv"
    )

    exact = exact[["sub_id", "group", "mmse", "post_10Hz_di"]].dropna().copy()
    exact["group_label"] = exact["group"].map(GROUP_MAP)
    group_order = ["CN", "AD", "FTD"]
    frequency_order = [5.0, 10.0, 15.0, 20.0]

    fig = plt.figure(figsize=(12.2, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    d_grid = grid[1, 1].subgridspec(1, 3, wspace=0.42)
    ax_d = [fig.add_subplot(d_grid[0, i]) for i in range(3)]

    sns.boxplot(
        data=visual,
        x="frequency_hz",
        y="local_snr_db",
        order=frequency_order,
        color="#DCEAF4",
        width=0.5,
        fliersize=0,
        linewidth=0.9,
        ax=ax_a,
    )
    for index, frequency in enumerate(frequency_order):
        values = visual.loc[visual["frequency_hz"] == frequency, "local_snr_db"].sort_values()
        ax_a.scatter(
            index + deterministic_jitter(len(values)),
            values,
            s=13,
            facecolor="white",
            edgecolor=COLORS["blue"],
            linewidth=0.55,
            alpha=0.8,
            zorder=3,
        )
        ax_a.text(
            (index + 0.5) / 4,
            0.98,
            f"n={len(values)}\n>0: {(values > 0).sum()}",
            transform=ax_a.transAxes,
            ha="center",
            va="top",
            fontsize=7.1,
        )
    ax_a.axhline(0, color="#555555", lw=0.9, ls="--")
    ax_a.set(
        title="Eyes-open local response detectability",
        xlabel="Visual stimulation frequency (Hz)",
        ylabel="O1/O2 local SNR (dB)",
    )
    panel_label(ax_a, "A")

    for index, group in enumerate(group_order):
        values = exact.loc[exact["group_label"] == group, "post_10Hz_di"].sort_values()
        parts = ax_b.violinplot(values, positions=[index], widths=0.72, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS[group])
            body.set_edgecolor(COLORS[group])
            body.set_alpha(0.18)
        ax_b.scatter(
            index + deterministic_jitter(len(values)),
            values,
            s=16,
            color=COLORS[group],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
        )
        median = float(values.median())
        ax_b.plot([index - 0.20, index + 0.20], [median, median], color="#111111", lw=2.0)
        ax_b.text(
            (index + 0.5) / 3,
            0.98,
            f"n={len(values)}",
            transform=ax_b.transAxes,
            ha="center",
            va="top",
            fontsize=7.2,
        )
    ax_b.set_xticks(range(3), group_order)
    ax_b.set(
        title="Closed-eye 10-Hz posterior driving index",
        xlabel="Diagnosis group",
        ylabel="Raw driving index (ratio)",
    )
    panel_label(ax_b, "B")

    forest = selectivity[selectivity["outcome"] == "selectivity_raw"].copy()
    y_positions = np.arange(len(forest))[::-1]
    for y, row in zip(y_positions, forest.itertuples(index=False)):
        color = COLORS["AD"] if row.contrast == "AD-CN" else COLORS["FTD"]
        ax_c.errorbar(
            row.coefficient,
            y,
            xerr=[[row.coefficient - row.ci_low], [row.ci_high - row.coefficient]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
        )
    ax_c.axvline(0, color="#555555", lw=0.9, ls="--")
    ax_c.set_yticks(y_positions, forest["contrast"])
    ax_c.set_ylim(-0.7, len(forest) - 0.3)
    ax_c.set(
        title="Closed-eye 10-Hz selectivity contrast",
        xlabel="Adjusted difference in raw DI\n(10 Hz minus mean of 5/15/20 Hz)",
        ylabel="",
    )
    panel_label(ax_c, "C")

    outcomes = ["raw_di", "log1p_di", "winsor_di"]
    outcome_titles = ["Raw", "log1p", "Winsorized"]
    for axis, outcome, title in zip(ax_d, outcomes, outcome_titles):
        frame = direct[direct["outcome"] == outcome]
        ypos = np.arange(len(frame))[::-1]
        for y, row in zip(ypos, frame.itertuples(index=False)):
            color = COLORS["AD"] if row.contrast == "AD-CN" else COLORS["FTD"]
            axis.errorbar(
                row.coefficient,
                y,
                xerr=[[row.coefficient - row.ci_low], [row.ci_high - row.coefficient]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2.5,
                markersize=4.5,
            )
        axis.axvline(0, color="#555555", lw=0.8, ls="--")
        axis.set_yticks(ypos, frame["contrast"] if axis is ax_d[0] else [])
        axis.set_title(f"Direct effect: {title}", fontsize=8.0)
        axis.set_xlabel("Adjusted difference", fontsize=7.1)
        axis.tick_params(axis="x", labelsize=6.6)
    ax_d[0].text(
        -0.28,
        1.14,
        "D",
        transform=ax_d[0].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    fig.suptitle(
        "Figure 4. Visual response estimates depend on state, frequency, and metric",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "Figure4_visual_state_frequency_metric")

    summary = (
        visual.groupby("frequency_hz", as_index=False)
        .agg(
            n=("subject", "nunique"),
            n_positive_snr=("local_snr_db", lambda values: int((values > 0).sum())),
            median_snr_db=("local_snr_db", "median"),
        )
    )
    summary.to_csv(output_dir / "Figure4_panelA_summary.csv", index=False)
    return [output_dir / "Figure4_visual_state_frequency_metric.png"]


def build_figure5(output_dir: Path) -> list[Path]:
    exact = pd.read_csv(ROOT / "outputs/competition/ds006036_exact_preprint/features_ica.csv")
    exact = exact[["sub_id", "group", "mmse", "post_10Hz_di"]].dropna().copy()
    exact["group_label"] = exact["group"].map(GROUP_MAP)
    models = fit_rest_to_visual_models()
    models.to_csv(output_dir / "Figure5_rest_to_visual_28_models.csv", index=False)

    advanced_severity = pd.read_csv(
        ROOT / "outputs/network/ds006036_models/patient_severity_models.csv"
    )
    advanced_groups = pd.read_csv(
        ROOT / "outputs/network/ds006036_models/diagnosis_group_models.csv"
    )
    advanced_rest = pd.read_csv(
        ROOT / "outputs/network/ds006036_models/rest_to_advanced_visual_models.csv"
    )

    fig = plt.figure(figsize=(11.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.03, 0.97], width_ratios=[1.08, 0.92])
    scatter_grid = grid[0, 0].subgridspec(1, 2, wspace=0.25)
    scatter_axes = [fig.add_subplot(scatter_grid[0, i]) for i in range(2)]
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    bootstrap_rows = []
    for axis, group, seed in zip(scatter_axes, ["AD", "FTD"], [20260919, 20260920]):
        frame = exact[exact["group_label"] == group]
        rho, ci_low, ci_high = bootstrap_spearman(
            frame["mmse"].to_numpy(float),
            frame["post_10Hz_di"].to_numpy(float),
            seed=seed,
        )
        p_value = float(spearmanr(frame["mmse"], frame["post_10Hz_di"]).pvalue)
        bootstrap_rows.append(
            {
                "group": group,
                "n": len(frame),
                "spearman_rho": rho,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "p_value": p_value,
                "bootstrap_resamples": 10_000,
            }
        )
        sns.regplot(
            data=frame,
            x="mmse",
            y="post_10Hz_di",
            lowess=True,
            ci=None,
            scatter_kws={"s": 23, "alpha": 0.75, "color": COLORS[group], "edgecolor": "white"},
            line_kws={"color": COLORS[group], "lw": 1.5},
            ax=axis,
        )
        axis.set_title(
            f"{group}: ρ={rho:.3f}\nbootstrap 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
            fontsize=8.6,
        )
        axis.set_xlabel("MMSE")
        axis.set_ylabel("10-Hz driving index" if axis is scatter_axes[0] else "")
    panel_label(scatter_axes[0], "A")
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "Figure5_panelA_bootstrap_correlations.csv", index=False
    )

    ladder = [
        ("Group-level effect", "Supported in closed-eye\n10-Hz selectivity", "supported"),
        ("Individual reliability", "Split-half ICC = 0.06\n(same-data preprint)", "failed"),
        ("Continuous severity", "AD/FTD within-group\nassociations weak", "failed"),
        ("Incremental validity", "No classification gain over\nresting EEG (preprint)", "failed"),
    ]
    ax_b.set_xlim(-0.15, 1.05)
    ax_b.set_ylim(-0.25, 3.25)
    for i, (heading, detail, status) in enumerate(ladder):
        y = 3 - i
        if i < len(ladder) - 1:
            ax_b.plot([0.12, 0.12], [y - 0.72, y - 0.18], color="#888888", lw=1.2)
        face = COLORS["blue"] if status == "supported" else "white"
        edge = COLORS["blue"] if status == "supported" else "#666666"
        ax_b.scatter(0.12, y, s=190, facecolor=face, edgecolor=edge, lw=1.6, zorder=3)
        ax_b.text(0.12, y, "✓" if status == "supported" else "—", ha="center", va="center", color="white" if status == "supported" else "#555555", fontsize=10, fontweight="bold")
        ax_b.text(0.24, y + 0.10, heading, ha="left", va="center", fontsize=7.8, fontweight="bold")
        ax_b.text(0.24, y - 0.13, detail.replace("\n", " "), ha="left", va="center", fontsize=7.0, color="#444444")
    ax_b.axis("off")
    ax_b.set_title("Validation ladder", pad=8)
    panel_label(ax_b, "B")

    ten_hz = models[models["frequency_hz"] == 10].copy()
    ten_hz["label"] = ten_hz["predictor"].map(PREDICTOR_LABELS)
    ten_hz = ten_hz.iloc[::-1]
    ypos = np.arange(len(ten_hz))
    ax_c.errorbar(
        ten_hz["coefficient_db_per_predictor_sd"],
        ypos,
        xerr=np.vstack(
            [
                ten_hz["coefficient_db_per_predictor_sd"] - ten_hz["ci_low"],
                ten_hz["ci_high"] - ten_hz["coefficient_db_per_predictor_sd"],
            ]
        ),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        capsize=2.5,
        markersize=4.5,
    )
    ax_c.axvline(0, color="#555555", lw=0.9, ls="--")
    ax_c.set_yticks(ypos, ten_hz["label"])
    ax_c.set(
        title="Resting features do not predict eyes-open 10-Hz local SNR",
        xlabel="Adjusted SNR difference (dB) per predictor SD",
        ylabel="",
    )
    ax_c.text(
        0.98,
        0.97,
        f"28 models across four frequencies; minimum FDR q={models['q_bh'].min():.3f}",
        transform=ax_c.transAxes,
        fontsize=7.2,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
    )
    panel_label(ax_c, "C")

    family_rows = pd.DataFrame(
        [
            {
                "family": "Patient severity",
                "minimum_q": advanced_severity["q_bh"].min(),
                "reference_check": "reference-sensitive",
            },
            {
                "family": "Diagnosis groups",
                "minimum_q": advanced_groups["q_bh"].min(),
                "reference_check": "reference-sensitive",
            },
            {
                "family": "Rest → advanced visual",
                "minimum_q": advanced_rest["q_bh"].min(),
                "reference_check": "exploratory only",
            },
        ]
    )
    family_rows.to_csv(output_dir / "Figure5_panelD_advanced_gate.csv", index=False)
    y = np.arange(len(family_rows))[::-1]
    ax_d.barh(y, family_rows["minimum_q"], color="#BDBDBD", height=0.52)
    ax_d.axvline(0.05, color="#555555", lw=0.9, ls="--")
    for yi, value in zip(y, family_rows["minimum_q"]):
        ax_d.text(value + 0.015, yi, f"min q={value:.3f}", va="center", fontsize=7.3)
    ax_d.set_yticks(y, family_rows["family"])
    ax_d.set_xlim(0, max(1.0, family_rows["minimum_q"].max() + 0.14))
    ax_d.set(
        title="Advanced visual analyses do not pass FDR gates",
        xlabel="Minimum Benjamini–Hochberg q within family",
        ylabel="",
    )
    panel_label(ax_d, "D")

    fig.suptitle(
        "Figure 5. A visual group effect does not establish individual clinical validity",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "Figure5_visual_individual_validity_gate")
    return [output_dir / "Figure5_visual_individual_validity_gate.png"]


def auditory_split_half_points() -> pd.DataFrame:
    blocks = pd.read_csv(ROOT / "outputs/qc/ds005048_full_foundation/block_channel_metrics.csv")
    roi = blocks[
        (blocks["condition"] == "stimulus")
        & (blocks["channel"].isin(["Fz", "Cz", "C3", "C4"]))
        & (blocks["block_index"] <= 6)
    ]
    block_level = (
        roi.groupby(["participant_id", "block_index"], as_index=False)
        .agg(snr_narrow_db=("snr_narrow_db", "mean"))
    )
    first = (
        block_level[block_level["block_index"] <= 3]
        .groupby("participant_id")["snr_narrow_db"]
        .mean()
        .rename("blocks_1_3_snr_db")
    )
    second = (
        block_level[block_level["block_index"].between(4, 6)]
        .groupby("participant_id")["snr_narrow_db"]
        .mean()
        .rename("blocks_4_6_snr_db")
    )
    return pd.concat([first, second], axis=1).dropna().reset_index()


def build_figure6(output_dir: Path) -> list[Path]:
    subjects = pd.read_csv(ROOT / "outputs/qc/ds005048_full_foundation/subject_summary.csv")
    split = auditory_split_half_points()
    split.to_csv(output_dir / "Figure6_panelB_split_half_points.csv", index=False)
    network_subjects = pd.read_csv(ROOT / "outputs/network/ds005048_full/subject_features.csv")
    models = pd.read_csv(ROOT / "outputs/network/ds005048_models/mmse_models.csv")
    reliability = pd.read_csv(ROOT / "outputs/network/ds005048_full/split_half_reliability.csv")

    fig = plt.figure(figsize=(11.2, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.02, 0.98],
        height_ratios=[1.0, 1.0],
        hspace=0.22,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    d_grid = grid[1, 1].subgridspec(2, 1, hspace=0.72)
    ax_d1 = fig.add_subplot(d_grid[0, 0])
    ax_d2 = fig.add_subplot(d_grid[1, 0])

    ordered = subjects.sort_values("snr_narrow_db_stim_minus_rest").reset_index(drop=True)
    colors = np.where(ordered["snr_narrow_db_stim_minus_rest"] > 0, COLORS["blue"], "#8C8C8C")
    ax_a.barh(
        np.arange(len(ordered)),
        ordered["snr_narrow_db_stim_minus_rest"],
        color=colors,
        height=0.72,
    )
    ax_a.axvline(0, color="#333333", lw=0.9)
    ax_a.axvline(
        ordered["snr_narrow_db_stim_minus_rest"].median(),
        color=COLORS["gold"],
        lw=1.6,
        ls="--",
        label=f"Median = {ordered['snr_narrow_db_stim_minus_rest'].median():.3f} dB",
    )
    ax_a.set_yticks([])
    ax_a.set(
        title="Subject-level 40-Hz target engagement",
        xlabel="Stimulus − rest narrow-band SNR (dB)",
        ylabel="35 participants (ordered)",
    )
    ax_a.legend(frameon=False, loc="lower right")
    ax_a.text(
        0.02,
        0.96,
        f"{(ordered['snr_narrow_db_stim_minus_rest'] > 0).sum()}/35 positive",
        transform=ax_a.transAxes,
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    panel_label(ax_a, "A")

    sns.regplot(
        data=split,
        x="blocks_1_3_snr_db",
        y="blocks_4_6_snr_db",
        ci=None,
        scatter_kws={"s": 26, "alpha": 0.78, "color": COLORS["blue"], "edgecolor": "white"},
        line_kws={"color": COLORS["blue"], "lw": 1.4},
        ax=ax_b,
    )
    rho_split, p_split = spearmanr(split["blocks_1_3_snr_db"], split["blocks_4_6_snr_db"])
    ax_b.set(
        title=f"Within-recording split-half stability\nSpearman ρ={rho_split:.3f}, n={len(split)}",
        xlabel="Blocks 1–3 mean 40-Hz SNR (dB)",
        ylabel="Blocks 4–6 mean 40-Hz SNR (dB)",
    )
    panel_label(ax_b, "B")

    clinical = subjects.dropna(subset=["mmse", "snr_narrow_db_stim_minus_rest"])
    rho_mmse, p_mmse = spearmanr(clinical["mmse"], clinical["snr_narrow_db_stim_minus_rest"])
    sns.regplot(
        data=clinical,
        x="mmse",
        y="snr_narrow_db_stim_minus_rest",
        lowess=True,
        ci=None,
        scatter_kws={"s": 26, "alpha": 0.78, "color": COLORS["overall"], "edgecolor": "white"},
        line_kws={"color": COLORS["overall"], "lw": 1.4},
        ax=ax_c,
    )
    ax_c.axhline(0, color="#777777", lw=0.8, ls="--")
    ax_c.set(
        title=f"Weak monotonic association with MMSE\nSpearman ρ={rho_mmse:.3f}, p={p_mmse:.3f}, n={len(clinical)}",
        xlabel="MMSE",
        ylabel="Stimulus − rest narrow-band SNR (dB)",
    )
    panel_label(ax_c, "C")

    primary = models[
        (models["family"] == "primary")
        & (models["model"] == "incremental_over_local_snr")
    ].copy()
    primary_labels = {
        "central_minus_frontal_snr_db": "Central−frontal SNR",
        "csd_central_frontal_abs_imcoh": "CSD |ImCoh|",
    }
    outcome_sd = (
        network_subjects.dropna(subset=["mmse"])
        .set_index("participant_id")[["central_minus_frontal_snr_db", "csd_central_frontal_abs_imcoh"]]
        .std(ddof=0)
    )
    primary["outcome_sd"] = primary["outcome"].map(outcome_sd)
    primary["standardized_coefficient"] = primary["coefficient_per_mmse_sd"] / primary["outcome_sd"]
    primary["standardized_se"] = primary["standard_error"] / primary["outcome_sd"]
    primary["ci_low"] = primary["standardized_coefficient"] - 1.96 * primary["standardized_se"]
    primary["ci_high"] = primary["standardized_coefficient"] + 1.96 * primary["standardized_se"]
    primary.to_csv(output_dir / "Figure6_panelD_standardized_models.csv", index=False)
    y1 = np.arange(len(primary))[::-1]
    ax_d1.errorbar(
        primary["standardized_coefficient"],
        y1,
        xerr=np.vstack(
            [
                primary["standardized_coefficient"] - primary["ci_low"],
                primary["ci_high"] - primary["standardized_coefficient"],
            ]
        ),
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        capsize=2.5,
        markersize=4.5,
    )
    ax_d1.axvline(0, color="#555555", lw=0.8, ls="--")
    ax_d1.set_yticks(y1, [primary_labels[x] for x in primary["outcome"]])
    ax_d1.set(
        title="MMSE increment beyond local SNR",
        xlabel="Standardized coefficient\n(95% CI)",
        ylabel="",
    )

    dynamics = reliability[
        reliability["feature"].isin(["within_block_slope_db_per_quartile", "q4_minus_q1_snr_db"])
    ].copy()
    dynamics_labels = {
        "within_block_slope_db_per_quartile": "Within-block slope",
        "q4_minus_q1_snr_db": "Q4−Q1",
    }
    y2 = np.arange(len(dynamics))[::-1]
    ax_d2.scatter(dynamics["split_half_spearman_rho"], y2, color=COLORS["gold"], s=30)
    ax_d2.axvline(0, color="#555555", lw=0.8, ls="--")
    ax_d2.set_yticks(y2, [dynamics_labels[x] for x in dynamics["feature"]])
    ax_d2.set_xlim(-0.35, 0.15)
    ax_d2.set(
        title="Dynamic features lack within-record stability",
        xlabel="Split-half Spearman ρ",
        ylabel="",
    )
    ax_d1.text(
        -0.30,
        1.12,
        "D",
        transform=ax_d1.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    fig.suptitle(
        "Figure 6. Auditory 40-Hz target engagement is measurable but clinically limited",
        fontsize=12,
        fontweight="bold",
    )
    save_figure(fig, output_dir, "Figure6_auditory_40hz_clinical_boundary")

    derived = pd.DataFrame(
        [
            {"metric": "positive_snr_delta", "value": int((subjects["snr_narrow_db_stim_minus_rest"] > 0).sum()), "n": len(subjects), "unit": "participants"},
            {"metric": "median_snr_delta", "value": subjects["snr_narrow_db_stim_minus_rest"].median(), "n": len(subjects), "unit": "dB"},
            {"metric": "split_half_spearman", "value": rho_split, "n": len(split), "unit": "rho"},
            {"metric": "mmse_spearman", "value": rho_mmse, "n": len(clinical), "unit": "rho"},
            {"metric": "mmse_spearman_p", "value": p_mmse, "n": len(clinical), "unit": "p"},
        ]
    )
    derived.to_csv(output_dir / "Figure6_derived_values.csv", index=False)
    return [output_dir / "Figure6_auditory_40hz_clinical_boundary.png"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output_dir: Path) -> None:
    sources = [
        "outputs/qc/ds006036_full_foundation/subject_frequency_summary.csv",
        "outputs/competition/ds006036_exact_preprint/features_ica.csv",
        "outputs/competition/ds006036_competitor_models/direct_10hz_models.csv",
        "outputs/competition/ds006036_competitor_models/selectivity_models.csv",
        "outputs/network/ds006036_models/patient_severity_models.csv",
        "outputs/network/ds006036_models/diagnosis_group_models.csv",
        "outputs/network/ds006036_models/rest_to_advanced_visual_models.csv",
        "outputs/qc/ds004504_full_foundation/paired_rest_visual_features.csv",
        "outputs/qc/ds005048_full_foundation/subject_summary.csv",
        "outputs/qc/ds005048_full_foundation/block_channel_metrics.csv",
        "outputs/network/ds005048_full/subject_features.csv",
        "outputs/network/ds005048_full/split_half_reliability.csv",
        "outputs/network/ds005048_models/mmse_models.csv",
    ]
    rows = []
    for relative in sources:
        path = ROOT / relative
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "source_manifest_sha256.csv", index=False)


def validate_outputs(output_dir: Path) -> None:
    model_table = pd.read_csv(output_dir / "Figure5_rest_to_visual_28_models.csv")
    if len(model_table) != 28:
        raise AssertionError(f"Expected 28 rest-to-visual models, found {len(model_table)}")
    if not np.isclose(model_table["q_bh"].min(), 0.6413180695031035, atol=1e-10):
        raise AssertionError("Rest-to-visual FDR minimum does not match the audited result")
    auditory = pd.read_csv(output_dir / "Figure6_derived_values.csv").set_index("metric")
    checks = {
        "positive_snr_delta": 32.0,
        "median_snr_delta": 1.3445258384592433,
        "split_half_spearman": 0.8383753501400562,
        "mmse_spearman": 0.085,
    }
    for metric, expected in checks.items():
        actual = float(auditory.loc[metric, "value"])
        tolerance = 5e-4 if metric == "mmse_spearman" else 1e-10
        if not np.isclose(actual, expected, atol=tolerance):
            raise AssertionError(f"{metric}: expected {expected}, found {actual}")
    for stem in [
        "Figure4_visual_state_frequency_metric",
        "Figure5_visual_individual_validity_gate",
        "Figure6_auditory_40hz_clinical_boundary",
    ]:
        for suffix in [".png", ".pdf"]:
            path = output_dir / f"{stem}{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"Missing or empty figure: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    build_figure4(output_dir)
    build_figure5(output_dir)
    build_figure6(output_dir)
    write_manifest(output_dir)
    validate_outputs(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "figures": [4, 5, 6],
                "validated": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
