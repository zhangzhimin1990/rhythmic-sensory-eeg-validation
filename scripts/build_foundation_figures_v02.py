#!/usr/bin/env python3
"""Build source-backed manuscript Figures 1–4 for the public-data route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.proportion import proportion_confint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "manuscript_figures_v02"

BLUE = "#2C6E9B"
ORANGE = "#D97735"
GOLD = "#C8A23A"
CHARCOAL = "#30343B"
MID = "#7A828B"
LIGHT = "#E8EDF1"
PALE_BLUE = "#DCEAF4"
PALE_ORANGE = "#F6E4D7"
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
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")


def save_figure(fig: plt.Figure, out: Path, stem: str) -> list[Path]:
    png = out / f"{stem}.png"
    pdf = out / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata=PDF_METADATA)
    plt.close(fig)
    return [png, pdf]


def bootstrap_spearman_ci(x, y, n_boot: int = 20_000, seed: int = 20260921):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        value = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(value):
            values.append(value)
    low, high = np.quantile(values, [0.025, 0.975])
    return observed, float(low), float(high), len(x)


def build_figure1(out: Path) -> list[Path]:
    transitions = [
        "T1  Response →\nmechanism",
        "T2  Reliability →\ncognitive validity",
        "T3  Mean effect →\nwithin-person coupling",
        "T4  Association →\nheld-out prediction",
    ]
    datasets = [
        "ds004504\nresting", "ds006036\nvisual", "ds005048\nauditory",
        "ds006222\naudiovisual", "ds007648\n36/40 Hz", "ds006780\n27/40 Hz",
        "Dryad\ntheta",
    ]
    # Role codes describe design contribution, not effect direction or evidence strength.
    # 0 no direct role, 1 context/data boundary, 2 direct transition test.
    values = np.array(
        [
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [0, 2, 0, 0],
            [1, 0, 0, 0],
            [0, 2, 0, 2],
            [0, 2, 0, 2],
            [0, 0, 2, 2],
        ],
        dtype=int,
    )
    labels = {0: "—", 1: "Context", 2: "Direct"}
    matrix_rows = []
    for dataset, row in zip(datasets, values):
        for transition, value in zip(transitions, row):
            matrix_rows.append(
                {
                    "dataset": dataset.replace("\n", " "),
                    "transition": transition.replace("\n", " "),
                    "role_code": int(value),
                    "role": labels[int(value)],
                }
            )
    role_matrix = pd.DataFrame(matrix_rows)
    role_matrix.to_csv(out / "Figure1_transition_role_matrix.csv", index=False)
    # Keep the legacy filename as an explicit alias so older release recipes do
    # not retain a stale seven-level matrix after the four-transition redesign.
    role_matrix.to_csv(out / "Figure1_validation_matrix.csv", index=False)

    evidence_rows = [
        {
            "transition_id": "T1",
            "antecedent": "Detectable or group-sensitive response",
            "stronger_claim": "State-general endogenous mechanism",
            "result": "Not established",
            "evidence": "ds006036: state/metric conditional; L041: phase concentration not mechanism-specific",
        },
        {
            "transition_id": "T2",
            "antecedent": "High within-recording reliability",
            "stronger_claim": "Matched cognitive validity",
            "result": "Not established in tested settings",
            "evidence": "ds005048: rho 0.838 vs MMSE rho 0.085; ds006780: ITPC 0.972, d-prime 0.931, beta 0.058",
        },
        {
            "transition_id": "T3",
            "antecedent": "Target engagement and acute active-control benefit",
            "stronger_claim": "Within-person neural coupling",
            "result": "Not supported for tested coupling estimand",
            "evidence": "Dryad N=35: ITPC +0.088; RT -27.73 ms; all four coupling Holm p=1.000",
        },
        {
            "transition_id": "T4",
            "antecedent": "In-sample neural association",
            "stronger_claim": "Held-out incremental prediction",
            "result": "No consistent increment",
            "evidence": "ds007648 and ds006780: no gain; Dryad N=35: no gain over both prediction comparators",
        },
    ]
    pd.DataFrame(evidence_rows).to_csv(out / "Figure1_transition_evidence.csv", index=False)

    fig = plt.figure(figsize=(13.0, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.7, 2.8])
    ax = fig.add_subplot(gs[0])
    cmap = ListedColormap(["#F1F3F5", "#F2DCA7", "#AFCFDF"])
    ax.imshow(values, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(transitions)), transitions)
    ax.set_yticks(range(len(datasets)), datasets)
    ax.tick_params(length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, labels[int(values[i, j])], ha="center", va="center",
                    color=CHARCOAL, fontsize=8, fontweight="bold" if values[i, j] else "normal")
    ax.set_title("A. Each cohort contributes to defined claim-transition tests", loc="left")
    ax.set_xticks(np.arange(-0.5, len(transitions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(
        handles=[
            Patch(facecolor="#AFCFDF", edgecolor="none", label="Direct transition test"),
            Patch(facecolor="#F2DCA7", edgecolor="none", label="Context or data boundary"),
            Patch(facecolor="#F1F3F5", edgecolor="none", label="No direct role"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.text(0.0, 1.02, "B. Four stronger claims required evidence not supplied by the preceding result",
             transform=ax2.transAxes, fontsize=10, fontweight="bold", va="top")
    y_positions = [0.82, 0.61, 0.40, 0.19]
    for row, yy in zip(evidence_rows, y_positions):
        left = FancyBboxPatch(
            (0.02, yy - 0.055), 0.29, 0.12,
            boxstyle="round,pad=0.010,rounding_size=0.010",
            facecolor=PALE_BLUE, edgecolor=CHARCOAL, linewidth=0.8,
            transform=ax2.transAxes,
        )
        right = FancyBboxPatch(
            (0.48, yy - 0.055), 0.49, 0.12,
            boxstyle="round,pad=0.010,rounding_size=0.010",
            facecolor=PALE_ORANGE, edgecolor=CHARCOAL, linewidth=0.8,
            transform=ax2.transAxes,
        )
        ax2.add_patch(left)
        ax2.add_patch(right)
        ax2.text(0.165, yy + 0.005, f"{row['transition_id']}  {row['antecedent']}",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=7.6)
        ax2.text(0.725, yy + 0.025, row["stronger_claim"],
                 ha="center", va="center", transform=ax2.transAxes, fontsize=7.6)
        ax2.annotate(
            "", xy=(0.475, yy + 0.005), xytext=(0.315, yy + 0.005),
            xycoords=ax2.transAxes, textcoords=ax2.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.1, color=MID),
        )
        ax2.text(0.725, yy - 0.025, row["result"], ha="center", va="center",
                 transform=ax2.transAxes, fontsize=7.1, fontweight="bold", color=ORANGE)
        ax2.text(0.40, yy - 0.085, row["evidence"], ha="center", va="top",
                 transform=ax2.transAxes, fontsize=6.9, color=CHARCOAL)
    ax2.text(
        0.02, 0.005,
        "Scope: human EEG secondary analyses; transition outcomes do not imply universal null effects, chronic efficacy, or disease modification.",
        transform=ax2.transAxes, fontsize=8.0, color=CHARCOAL,
    )
    fig.suptitle("Figure 1. Seven public cohorts test four inferential transitions", fontsize=13,
                 fontweight="bold")
    return save_figure(fig, out, "Figure1_multicohort_validation_design")


def build_figure2(out: Path) -> list[Path]:
    rows = [
        ["ds004504", "88", "88 usable", "N=88", "MMSE N=88"],
        ["ds006036", "88", "88 records; 372 reconstructable blocks", "N=76–80 by primary visual endpoint", "MMSE/group N varies by estimand"],
        ["ds005048", "35", "35 usable", "40-Hz N=35", "MMSE N=33"],
        ["ds006222", "69 participants / 70 records", "6/6 sampled signal gate", "Full-cohort signal not modeled", "Behavior unavailable"],
        ["ds007648", "22", "20 neural usable", "8,579 accuracy trials", "7,456 correct RT trials"],
        ["ds006780", "123 ASSR", "111 technical", "d-prime N=111", "FSIQ N=109"],
        ["Dryad theta", "44", "44 complete released table", "132 rhythmic condition rows", "R2: 3/5 technical; R2b pending"],
    ]
    cols = ["Dataset", "Released", "Signal/structural gate", "Primary estimation", "Outcome-specific N / boundary"]
    pd.DataFrame(rows, columns=cols).to_csv(out / "Figure2_cohort_flow.csv", index=False)

    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    ax.axis("off")
    widths = [0.12, 0.17, 0.25, 0.22, 0.24]
    xpos = np.cumsum([0] + widths[:-1])
    row_h = 0.095
    top = 0.80
    for j, (col, xx, width) in enumerate(zip(cols, xpos, widths)):
        ax.add_patch(plt.Rectangle((xx, top), width, row_h, transform=ax.transAxes,
                                   facecolor=CHARCOAL, edgecolor="white", linewidth=1))
        ax.text(xx + 0.01, top + row_h / 2, col, transform=ax.transAxes,
                va="center", color="white", fontsize=8, fontweight="bold")
    for i, row in enumerate(rows):
        yy = top - (i + 1) * row_h
        fill = "#F7F9FA" if i % 2 == 0 else "#EAF0F3"
        for j, (value, xx, width) in enumerate(zip(row, xpos, widths)):
            ax.add_patch(plt.Rectangle((xx, yy), width, row_h, transform=ax.transAxes,
                                       facecolor=fill, edgecolor="white", linewidth=1))
            color = ORANGE if ("unavailable" in value.lower() or "pending" in value.lower()) else CHARCOAL
            weight = "bold" if j == 0 else "normal"
            ax.text(xx + 0.01, yy + row_h / 2, value, transform=ax.transAxes,
                    va="center", fontsize=7.5, color=color, fontweight=weight, wrap=True)
    ax.text(0, 0.97, "Figure 2. Outcome-specific sample flow and public-data boundaries",
            transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    ax.text(0, 0.035,
            "Counts are not additive across columns. Trials, blocks, runs, and participant-condition rows are repeated observations, not independent participants.",
            transform=ax.transAxes, fontsize=8.2, color=CHARCOAL)
    return save_figure(fig, out, "Figure2_outcome_specific_sample_flow")


def build_figure3(out: Path) -> list[Path]:
    detect = pd.DataFrame(
        [
            ("Visual 5 Hz", 64, 80), ("Visual 10 Hz", 72, 76),
            ("Visual 15 Hz", 76, 78), ("Visual 20 Hz", 70, 77),
            ("Auditory 40 Hz", 32, 35),
        ], columns=["response", "positive", "n"]
    )
    low, high = proportion_confint(detect["positive"], detect["n"], alpha=0.05, method="wilson")
    detect["proportion"] = detect["positive"] / detect["n"]
    detect["ci_low"], detect["ci_high"] = low, high
    detect.to_csv(out / "Figure3A_detectability.csv", index=False)

    reliability = pd.DataFrame(
        [
            ("ds005048 auditory 40 Hz", 35, 0.838, "block split-half"),
            ("ds007648 visual 36 Hz", 20, 0.8700085, "odd/even; SB corrected"),
            ("ds007648 auditory 40 Hz", 20, 0.9847328, "odd/even; SB corrected"),
            ("ds006780 auditory 27 Hz", 111, 0.9514894, "odd/even; SB corrected"),
            ("ds006780 auditory 40 Hz", 111, 0.9724817, "odd/even; SB corrected"),
        ], columns=["measure", "n", "reliability", "definition"]
    )
    reliability.to_csv(out / "Figure3B_reliability.csv", index=False)

    visual = pd.read_csv(ROOT / "outputs/competition/ds006036_competitor_models/selectivity_models.csv")
    visual = visual[visual["outcome"] == "selectivity_raw"].copy()
    visual.to_csv(out / "Figure3C_visual_sensitivity.csv", index=False)
    dev = pd.read_csv(ROOT / "outputs/models/ds006780_behavioral_validity/replication_models.csv")
    dev = dev[(dev["frequency_hz"] == 40) & (dev["term"] == "age_z") &
              (dev["feature_family"].isin(["local_log_snr", "itpc"]))].copy()
    subjects = pd.read_csv(ROOT / "outputs/models/ds006780_behavioral_validity/subject_level_features.csv")
    subjects = subjects[subjects["technical_usable"] == True]
    outcome_fields = {"local_log_snr": "local_log_snr_db_mean_40", "itpc": "itpc_40"}
    standard_deviations = {
        family: float(subjects[field].std(ddof=0)) for family, field in outcome_fields.items()
    }
    dev["outcome_sd"] = dev["feature_family"].map(standard_deviations)
    for field in ["estimate", "ci_low", "ci_high"]:
        dev[f"standardized_{field}"] = dev[field] / dev["outcome_sd"]
    dev.to_csv(out / "Figure3D_developmental_sensitivity.csv", index=False)

    fig = plt.figure(figsize=(13.0, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    y = np.arange(len(detect))[::-1]
    ax_a.errorbar(detect["proportion"], y,
                  xerr=[detect["proportion"] - detect["ci_low"], detect["ci_high"] - detect["proportion"]],
                  fmt="o", color=BLUE, ecolor=BLUE, capsize=3)
    ax_a.set_yticks(y, detect["response"])
    ax_a.set_xlim(0.55, 1.02)
    ax_a.set_xlabel("Participants with response > 0 (proportion; Wilson 95% CI)")
    ax_a.set_title("Detectability in cohorts with a binary response rule", loc="left")
    for yy, row in zip(y, detect.itertuples()):
        ax_a.text(row.ci_high + 0.01, yy, f"{row.positive}/{row.n}", va="center", fontsize=7.3)
    panel_label(ax_a, "A")

    y = np.arange(len(reliability))[::-1]
    ax_b.scatter(reliability["reliability"], y, color=ORANGE, s=38)
    ax_b.set_yticks(y, reliability["measure"])
    ax_b.set_xlim(0.75, 1.01)
    ax_b.set_xlabel("Within-recording split-half coefficient")
    ax_b.set_title("Participant ranking can be highly stable", loc="left")
    for yy, row in zip(y, reliability.itertuples()):
        ax_b.text(row.reliability + 0.006, yy, f"{row.reliability:.3f}", va="center", fontsize=7.3)
    ax_b.text(0.0, -0.19, "Point estimates; interval estimates were not available on a common basis.",
              transform=ax_b.transAxes, fontsize=7.2, color=MID)
    panel_label(ax_b, "B")

    y = np.arange(len(visual))[::-1]
    ax_c.errorbar(visual["coefficient"], y,
                  xerr=[visual["coefficient"] - visual["ci_low"], visual["ci_high"] - visual["coefficient"]],
                  fmt="o", color=BLUE, ecolor=BLUE, capsize=3)
    ax_c.axvline(0, color=MID, ls="--", lw=0.9)
    ax_c.set_yticks(y, visual["contrast"])
    ax_c.set_xlabel("Adjusted raw DI difference (95% CI)")
    ax_c.set_title("Visual 10-Hz selectivity is group-sensitive", loc="left")
    panel_label(ax_c, "C")

    labels = {"local_log_snr": "Local log-SNR", "itpc": "ITPC"}
    y = np.arange(len(dev))[::-1]
    for yy, row in zip(y, dev.itertuples()):
        ax_d.errorbar(row.standardized_estimate, yy,
                      xerr=[[row.standardized_estimate-row.standardized_ci_low],
                            [row.standardized_ci_high-row.standardized_estimate]],
                      fmt="o", color=ORANGE, ecolor=ORANGE, capsize=3)
    ax_d.axvline(0, color=MID, ls="--", lw=0.9)
    ax_d.set_yticks(y, [labels[v] for v in dev["feature_family"]])
    ax_d.set_xlabel("Response SD per 1-SD older age (95% CI)")
    ax_d.set_title("40-Hz ASSR is developmentally sensitive", loc="left")
    panel_label(ax_d, "D")
    fig.suptitle("Figure 3. Detectability, reliability, and sensitivity are distinct properties",
                 fontsize=13, fontweight="bold")
    return save_figure(fig, out, "Figure3_detectability_reliability_sensitivity")


def build_figure4(out: Path) -> list[Path]:
    resting = pd.read_csv(ROOT / "outputs/qc/ds004504_full_foundation/resting_features.csv")
    features = [
        ("Posterior IAF", "posterior_iaf_hz"), ("Slowing ratio", "slowing_ratio"),
        ("Spectral exponent", "spectral_exponent"), ("Relative delta", "relative_delta"),
        ("Relative theta", "relative_theta"), ("Relative alpha", "relative_alpha"),
    ]
    rest_rows = []
    for idx, (label, field) in enumerate(features):
        rho, low, high, n = bootstrap_spearman_ci(resting[field], resting["mmse"], seed=20260921 + idx)
        rest_rows.append((label, n, rho, low, high))
    rest = pd.DataFrame(rest_rows, columns=["feature", "n", "rho", "ci_low", "ci_high"])
    rest.to_csv(out / "Figure4A_resting_mmse.csv", index=False)

    selectivity = pd.read_csv(ROOT / "outputs/competition/ds006036_competitor_models/selectivity_models.csv")
    selectivity = selectivity[selectivity["outcome"] == "selectivity_raw"].copy()
    selectivity.to_csv(out / "Figure4B_visual_group_selectivity.csv", index=False)
    precision = pd.read_csv(ROOT / "outputs/models/validation_strength/correlation_precision.csv")
    within = precision.loc[
        precision["dataset"].eq("ds006036"),
        ["population", "n", "estimate", "ci95_low", "ci95_high"],
    ].rename(
        columns={
            "population": "group",
            "estimate": "spearman_rho",
            "ci95_low": "bootstrap_ci_low",
            "ci95_high": "bootstrap_ci_high",
        }
    )
    within.to_csv(out / "Figure4C_visual_within_group_mmse.csv", index=False)
    auditory = precision.copy()
    auditory = auditory[(auditory["dataset"] == "ds005048") & (auditory["outcome"] == "MMSE")].copy()
    auditory.to_csv(out / "Figure4D_auditory_mmse.csv", index=False)

    fig = plt.figure(figsize=(13.0, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    ax = axes[0]
    y = np.arange(len(rest))[::-1]
    ax.errorbar(rest["rho"], y, xerr=[rest["rho"]-rest["ci_low"], rest["ci_high"]-rest["rho"]],
                fmt="o", color=BLUE, ecolor=BLUE, capsize=3)
    ax.axvline(0, color=MID, ls="--", lw=0.9)
    ax.set_yticks(y, rest["feature"])
    ax.set_xlim(-0.8, 0.7)
    ax.set_xlabel("Spearman rho with MMSE (bootstrap 95% CI)")
    ax.set_title("Resting EEG captures a disease-related spectral axis", loc="left")
    panel_label(ax, "A")

    ax = axes[1]
    y = np.arange(len(selectivity))[::-1]
    ax.errorbar(selectivity["coefficient"], y,
                xerr=[selectivity["coefficient"]-selectivity["ci_low"], selectivity["ci_high"]-selectivity["coefficient"]],
                fmt="o", color=ORANGE, ecolor=ORANGE, capsize=3)
    ax.axvline(0, color=MID, ls="--", lw=0.9)
    ax.set_yticks(y, selectivity["contrast"])
    ax.set_xlabel("Adjusted 10-Hz selectivity difference (95% CI)")
    ax.set_title("A visual group contrast is present", loc="left")
    panel_label(ax, "B")

    ax = axes[2]
    y = np.arange(len(within))[::-1]
    ax.errorbar(within["spearman_rho"], y,
                xerr=[within["spearman_rho"]-within["bootstrap_ci_low"], within["bootstrap_ci_high"]-within["spearman_rho"]],
                fmt="o", color=BLUE, ecolor=BLUE, capsize=3)
    ax.axvline(0, color=MID, ls="--", lw=0.9)
    ax.set_yticks(y, [f"{g} (n={n})" for g, n in zip(within["group"], within["n"])])
    ax.set_xlim(-0.8, 0.7)
    ax.set_xlabel("Within-group DI–MMSE Spearman rho (95% CI)")
    ax.set_title("The group contrast does not show a stable severity map", loc="left")
    panel_label(ax, "C")

    ax = axes[3]
    row = auditory.iloc[0]
    ax.errorbar(row["estimate"], 0,
                xerr=[[row["estimate"]-row["ci95_low"]], [row["ci95_high"]-row["estimate"]]],
                fmt="o", color=ORANGE, ecolor=ORANGE, capsize=4, markersize=6)
    ax.axvline(0, color=MID, ls="--", lw=0.9)
    ax.set_yticks([0], [f"Auditory 40 Hz (n={int(row['n'])})"])
    ax.set_xlim(-0.8, 0.7)
    ax.set_xlabel("SNR–MMSE Spearman rho (bootstrap 95% CI)")
    ax.set_title("Auditory target engagement does not establish a severity marker", loc="left")
    panel_label(ax, "D")
    fig.suptitle("Figure 4. Group effects and target engagement do not establish individual clinical validity",
                 fontsize=13, fontweight="bold")
    return save_figure(fig, out, "Figure4_clinical_validity_boundary")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    set_style()
    files = []
    files += build_figure1(out)
    files += build_figure2(out)
    files += build_figure3(out)
    files += build_figure4(out)
    sources = [
        ROOT / "outputs/manuscript_tables_v02/Table1_public_cohorts.csv",
        ROOT / "outputs/manuscript_tables_v02/Table2_inferential_transitions.csv",
        ROOT / "outputs/manuscript_tables_v02/Table2_validation_claims.csv",
        ROOT / "outputs/innovation/claim_transition_falsification_matrix.csv",
        ROOT / "outputs/qc/ds004504_full_foundation/resting_features.csv",
        ROOT / "outputs/competition/ds006036_exact_preprint/features_ica.csv",
        ROOT / "outputs/competition/ds006036_competitor_models/selectivity_models.csv",
        ROOT / "outputs/models/validation_strength/correlation_precision.csv",
        ROOT / "outputs/models/ds007648_behavioral_validity/split_half_reliability.csv",
        ROOT / "outputs/models/ds006780_behavioral_validity/split_half_reliability.csv",
        ROOT / "outputs/models/ds006780_behavioral_validity/replication_models.csv",
    ]
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "figure_files": [{"path": str(p.relative_to(ROOT)), "sha256": file_sha256(p)} for p in files],
        "source_files": [{"path": str(p.relative_to(ROOT)), "sha256": file_sha256(p)} for p in sources],
        "notes": [
            "Figure 1 matrix cells encode cohort roles in four claim-transition tests, not effect direction, evidence strength, or meta-analytic effect sizes.",
            "Figure 2 counts are outcome specific and are not additive across columns.",
            "Figure 3 reliability values are point estimates because common-basis intervals were unavailable.",
            "Figure 4 uses separate axes for incomparable effect scales.",
        ],
    }
    (out / "foundation_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
