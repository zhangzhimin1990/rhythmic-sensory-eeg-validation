#!/usr/bin/env python3
"""Build HBM Figure 4: external reliability, validity, and prediction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DS7 = ROOT / "outputs/models/ds007648_behavioral_validity"
DS6 = ROOT / "outputs/models/ds006780_behavioral_validity"
AUDIT = ROOT / "outputs/models/hbm_prediction_audit_v1"
UNCERTAINTY = ROOT / "outputs/models/hbm_reviewer_uncertainty_audit_v1"
ALIGNMENT = ROOT / "outputs/models/estimand_alignment/matched_reliability_validity.csv"
OUTPUT = ROOT / "figures/hbm_research_article_v1"
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(*args, **kwargs):
        """Skip optional authoring-time layout audit when its helper is unavailable."""
        return None


COLORS = {
    "ds7": "#3A7CA5",
    "ds6": "#7C5E99",
    "boundary": "#D39A3A",
    "dark": "#263238",
    "grid": "#D9DEE3",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label, xy=(0, 1), xycoords="axes fraction", xytext=(-18, 7),
        textcoords="offset points", fontsize=9, fontweight="bold",
        ha="left", va="bottom",
    )


def reliability_table() -> pd.DataFrame:
    frame = pd.read_csv(UNCERTAINTY / "matched_itpc_reliability_bootstrap.csv")
    frame = frame.loc[
        frame.dataset.eq("ds007648")
        | (frame.dataset.eq("ds006780") & frame.frequency_hz.eq(40))
    ].copy()
    frame["label"] = frame.apply(
        lambda row: f"{row.dataset}\n{int(row.frequency_hz)}-Hz ITPC", axis=1
    )
    frame["value"] = frame.spearman_brown_full_length
    frame["low"] = frame.sb_bootstrap_ci95_low
    frame["high"] = frame.sb_bootstrap_ci95_high
    return frame[["label", "value", "low", "high", "dataset"]]


def validity_table() -> pd.DataFrame:
    matched = pd.read_csv(ALIGNMENT)
    matched = matched.loc[
        matched.evidence_pair.isin([
            "aggregate_itpc_to_aggregate_accuracy",
            "aggregate_itpc_to_aggregate_correct_rt",
            "itpc_to_dprime_40",
        ])
    ].copy()
    rows = []
    for row in matched.itertuples(index=False):
        if row.dataset == "ds007648":
            label = "ds007648 accuracy" if "accuracy" in row.evidence_pair else "ds007648 correct RT"
            unit = "Spearman rho"
            interpretation = "unresolved"
        else:
            label = "ds006780 d-prime"
            unit = "outcome SD"
            interpretation = "equivalence not established"
        rows.append({
            "outcome": label,
            "estimate_ci": f"{row.validity_estimate:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}] {unit}",
            "interpretation": interpretation,
        })
    return pd.DataFrame(rows)


def prediction_summary() -> pd.DataFrame:
    summary = pd.read_csv(AUDIT / "prediction_audit_summary.csv")
    summary = summary.loc[summary.dataset.ne("Dryad")].copy()
    intervals = pd.read_csv(UNCERTAINTY / "prediction_increment_bootstrap.csv")
    intervals = intervals.loc[intervals.dataset.ne("Dryad")].copy()
    summary = summary.merge(
        intervals[["dataset", "outcome", "bootstrap_ci95_low", "bootstrap_ci95_high"]],
        on=["dataset", "outcome"], validate="one_to_one",
    )
    summary["mean_pct"] = 100 * summary.observed_mean_delta / summary.mean_base
    summary["ci_low_pct"] = 100 * summary.bootstrap_ci95_low / summary.mean_base
    summary["ci_high_pct"] = 100 * summary.bootstrap_ci95_high / summary.mean_base
    summary["label"] = summary.apply(
        lambda row: {
            ("ds007648", "accuracy"): "ds007648 accuracy",
            ("ds007648", "rt"): "ds007648 RT",
            ("ds006780", "dprime_40"): "ds006780 d-prime",
            ("ds006780", "fsiq"): "ds006780 FSIQ",
        }[(row.dataset, row.outcome)],
        axis=1,
    )
    return summary


def build() -> dict[str, object]:
    reliability = reliability_table()
    validity = validity_table()
    prediction = prediction_summary()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reliability.to_csv(OUTPUT / "Figure4_source_reliability.csv", index=False)
    validity.to_csv(OUTPUT / "Figure4_source_validity.csv", index=False)
    prediction.to_csv(OUTPUT / "Figure4_source_prediction.csv", index=False)

    fig = plt.figure(figsize=(7.0866, 4.6063), constrained_layout=False)
    grid = fig.add_gridspec(2, 2)
    axa = fig.add_subplot(grid[0, 0])
    axb = fig.add_subplot(grid[0, 1])
    axc = fig.add_subplot(grid[1, :])
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.15, top=0.91, wspace=0.42, hspace=0.60)

    x = np.arange(len(reliability))
    bar_colors = [COLORS["ds7"], COLORS["ds7"], COLORS["ds6"]]
    bars = axa.bar(x, reliability.value, color=bar_colors, width=0.64)
    axa.errorbar(
        x, reliability.value,
        yerr=[reliability.value - reliability.low, reliability.high - reliability.value],
        fmt="none", ecolor=COLORS["dark"], capsize=2.5, lw=0.9,
    )
    axa.set_xticks(x, reliability.label)
    axa.set_ylim(0, 1.15)
    axa.set_ylabel("Spearman–Brown reliability")
    axa.set_title("Aggregate response reliability", loc="left", fontweight="bold")
    for bar, value in zip(bars, reliability.value):
        axa.text(bar.get_x() + bar.get_width() / 2, value + 0.085, f"{value:.3f}",
                 ha="center", va="bottom", fontsize=6.5)
    axa.axhline(0.8, color=COLORS["grid"], lw=0.8, ls="--")
    panel_label(axa, "a")

    axb.set_axis_off()
    axb.set_title("Matched aggregate validity remains unresolved", loc="left", fontweight="bold", pad=8)
    y_positions = np.linspace(0.78, 0.15, len(validity))
    for ypos, row in zip(y_positions, validity.itertuples(index=False)):
        axb.text(0.00, ypos + 0.035, row.outcome, transform=axb.transAxes,
                 ha="left", va="center", fontsize=6.4, fontweight="bold")
        axb.text(0.00, ypos - 0.045, row.estimate_ci, transform=axb.transAxes,
                 ha="left", va="center", fontsize=6.0)
        color = COLORS["boundary"] if row.interpretation != "supported" else COLORS["ds7"]
        axb.text(0.98, ypos + 0.035, row.interpretation, transform=axb.transAxes,
                 ha="right", va="center", fontsize=6.2, color=color)
    axb.text(0.00, -0.03, "Native units retained; rows are not pooled.", transform=axb.transAxes,
             ha="left", va="top", fontsize=6, color=COLORS["dark"])
    panel_label(axb, "b")

    order = ["ds007648 accuracy", "ds007648 RT", "ds006780 d-prime", "ds006780 FSIQ"]
    view = prediction.set_index("label").loc[order].reset_index()
    y = np.arange(len(view))[::-1]
    colors = [COLORS["ds7"], COLORS["ds7"], COLORS["ds6"], COLORS["ds6"]]
    for ypos, row, color in zip(y, view.itertuples(index=False), colors):
        axc.errorbar(
            row.mean_pct, ypos,
            xerr=[[row.mean_pct - row.ci_low_pct], [row.ci_high_pct - row.mean_pct]],
            fmt="D", color=color, ecolor=color, capsize=2.5, lw=1.1, ms=4,
        )
    axc.axvline(0, color=COLORS["dark"], lw=0.8)
    labels_with_fraction = [
        f"{row.label} ({100 * row.fraction_subjects_improved:.1f}% improved)"
        for row in view.itertuples(index=False)
    ]
    axc.set_yticks(y, labels_with_fraction)
    axc.set_xlim(-7.5, 7.5)
    axc.set_xlabel(
        "Mean loss increment (% of baseline loss); negative values favor neural features\n"
        "error bars: participant-bootstrap 95% CI"
    )
    axc.set_title("No stable incremental prediction", loc="left", fontweight="bold")
    panel_label(axc, "c")

    for ax in (axa, axc):
        ax.tick_params(length=3, width=0.7)
    fig.canvas.draw()
    stem = OUTPUT / "Figure4_External_validation"
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    result = {
        "figure": "Figure 4",
        "reliability_rows": len(reliability),
        "validity_rows": len(validity),
        "prediction_rows": len(prediction),
        "status": "rendered_pending_qa",
    }
    (OUTPUT / "Figure4_build_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
