#!/usr/bin/env python3
"""Build HBM Figure 1: claim ladder and three-cohort evidence design."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures/hbm_research_article_v1"
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(*args, **kwargs):
        """Skip optional authoring-time layout audit when its helper is unavailable."""
        return None


COLORS = {
    "target": "#2C6E9B",
    "reliability": "#5A8FB0",
    "validity": "#D39A3A",
    "prediction": "#7C5E99",
    "neutral": "#E8ECEF",
    "edge": "#53616B",
    "text": "#263238",
    "white": "#FFFFFF",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label, xy=(0, 1), xycoords="axes fraction", xytext=(-18, 7),
        textcoords="offset points", fontsize=9, fontweight="bold",
        ha="left", va="bottom",
    )


def box(ax: plt.Axes, x: float, y: float, width: float, height: float,
        text: str, face: str, edge: str = COLORS["edge"], text_color: str = COLORS["text"],
        fontsize: float = 6.5, weight: str = "normal") -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color=text_color,
            fontweight=weight, linespacing=1.12)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = COLORS["edge"], style: str = "-|>") -> None:
    patch = FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle=style,
        mutation_scale=8, linewidth=0.9, color=color,
    )
    ax.add_patch(patch)


def build() -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = pd.DataFrame([
        ("Dryad", "primary", 35, "target engagement; behavior; coupling; prediction; spatial robustness"),
        ("ds007648", "external", 20, "reliability; concurrent validity; prediction"),
        ("ds006780", "external", 111, "reliability; concurrent validity; prediction"),
    ], columns=["dataset", "role", "analysis_n", "direct_evidence"])
    source.to_csv(OUTPUT / "Figure1_source_design.csv", index=False)

    fig = plt.figure(figsize=(7.0866, 4.0551), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.22])
    axa = fig.add_subplot(grid[0, :])
    axb = fig.add_subplot(grid[1, 0])
    axc = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.07, top=0.91, wspace=0.22, hspace=0.34)
    for ax in (axa, axb, axc):
        ax.set_axis_off()

    axa.set_title("Each stronger claim requires a matched test", loc="left", fontweight="bold", pad=6)
    stages = [
        (0.01, "Target engagement", "Does EEG follow\nthe stimulus?", COLORS["target"]),
        (0.265, "Within-recording\nreliability", "Are individual rankings\nstable?", COLORS["reliability"]),
        (0.52, "Concurrent cognitive\nvalidity", "Does neural variation\ntrack behavior?", COLORS["validity"]),
        (0.775, "Incremental\nprediction", "Does EEG improve\nheld-out prediction?", COLORS["prediction"]),
    ]
    for x, label, question, color in stages:
        box(axa, x, 0.47, 0.205, 0.25, label, color, edge=color,
            text_color=COLORS["white"], fontsize=6.8, weight="bold")
        axa.text(x + 0.1025, 0.35, question, transform=axa.transAxes,
                 ha="center", va="center", fontsize=6.0, color=COLORS["text"])
    for left, right in zip(stages[:-1], stages[1:]):
        arrow(axa, (left[0] + 0.205, 0.595), (right[0] - 0.012, 0.595))
    axa.text(0.5, 0.10, "Arrows are empirical questions—not assumed implications.",
             transform=axa.transAxes, ha="center", va="center", fontsize=6.5,
             color=COLORS["text"])
    panel_label(axa, "a")

    axb.set_title("Primary within-participant test", loc="left", fontweight="bold", pad=6)
    box(axb, 0.02, 0.72, 0.22, 0.16, "Public cohort\nN=44", COLORS["neutral"], weight="bold")
    arrow(axb, (0.24, 0.80), (0.32, 0.80))
    box(axb, 0.33, 0.72, 0.25, 0.16, "Development\nn=5", COLORS["neutral"], weight="bold")
    box(axb, 0.33, 0.49, 0.25, 0.16, "Frozen-pipeline\napplication n=39", COLORS["neutral"], fontsize=5.8, weight="bold")
    arrow(axb, (0.24, 0.78), (0.32, 0.57))
    arrow(axb, (0.58, 0.57), (0.66, 0.57))
    box(axb, 0.67, 0.49, 0.29, 0.16, "Frozen cohort\nn=35", COLORS["target"],
        edge=COLORS["target"], text_color=COLORS["white"], weight="bold")
    box(axb, 0.05, 0.16, 0.34, 0.17, "Theta-plus\n1.33 × individual theta", "#D8E8F1",
        edge=COLORS["target"], weight="bold")
    box(axb, 0.49, 0.16, 0.34, 0.17, "Non-rhythmic\nactive comparator", COLORS["neutral"], weight="bold")
    arrow(axb, (0.39, 0.245), (0.49, 0.245), style="<->")
    axb.text(0.44, 0.02, "Target engagement • acute behavior • coupling • participant-held-out prediction",
             transform=axb.transAxes, ha="center", va="bottom", fontsize=6.1,
             color=COLORS["text"])
    panel_label(axb, "b")

    axc.set_title("Matched and complementary external tests", loc="left",
                  fontweight="bold", pad=6)
    rows = [
        (0.60, "ds007648\nn=20", "36/40-Hz cross-modal task", COLORS["target"]),
        (0.20, "ds006780\nn=111/109", "27/40-Hz auditory oddball", COLORS["prediction"]),
    ]
    for y, label, descriptor, color in rows:
        box(axc, 0.01, y, 0.25, 0.24, label, color, edge=color,
            text_color=COLORS["white"], fontsize=6.4, weight="bold")
        box(axc, 0.34, y + 0.13, 0.17, 0.10, "Aggregate\nreliability", "#D8E8F1",
            edge=COLORS["reliability"], fontsize=6.1, weight="bold")
        box(axc, 0.57, y + 0.13, 0.17, 0.10, "Matched\nvalidity", "#F4E4C6",
            edge=COLORS["validity"], fontsize=6.1, weight="bold")
        box(axc, 0.80, y + 0.13, 0.17, 0.10, "Prediction", "#E4DDEC",
            edge=COLORS["prediction"], fontsize=6.1, weight="bold")
        arrow(axc, (0.26, y + 0.18), (0.33, y + 0.18))
        arrow(axc, (0.51, y + 0.18), (0.56, y + 0.18))
        arrow(axc, (0.74, y + 0.18), (0.79, y + 0.18))
        axc.text(0.62, y + 0.03, f"{descriptor}; dataset-specific estimands",
                 transform=axc.transAxes, ha="center", va="center", fontsize=5.9,
                 color=COLORS["text"])
    panel_label(axc, "c")

    fig.canvas.draw()
    stem = OUTPUT / "Figure1_Claim_matched_design"
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
    result = {"figure": "Figure 1", "cohorts": 3, "status": "rendered_pending_qa"}
    (OUTPUT / "Figure1_build_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
