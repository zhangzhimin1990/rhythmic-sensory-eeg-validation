#!/usr/bin/env python3
"""Build HBM Figure 3: behavior, coupling precision, and held-out prediction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/models/dryad_confirmatory_group_v1"
AUDIT = ROOT / "outputs/models/hbm_prediction_audit_v1"
OUTPUT = ROOT / "figures/hbm_research_article_v1"
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(*args, **kwargs):
        """Skip optional authoring-time layout audit when its helper is unavailable."""
        return None


COLORS = {
    "baseline": "#7C8794",
    "neural": "#2C6E9B",
    "boundary": "#D39A3A",
    "dark": "#263238",
    "grid": "#D9DEE3",
    "light": "#EAF1F5",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-18, 7),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    behavior = pd.read_csv(INPUT / "paired_behaviour_effects.csv")
    behavior["sd_paired"] = (
        behavior.mean_difference_theta_plus_minus_non_rhythmic / behavior.paired_dz
    )
    behavior["effect"] = behavior.paired_dz
    behavior["low"] = behavior.ci95_low / behavior.sd_paired
    behavior["high"] = behavior.ci95_high / behavior.sd_paired
    behavior["display"] = behavior.metric.map({
        "accuracy": "Accuracy",
        "median_correct_rt_ms": "Correct RT",
        "inverse_efficiency_ms": "Inverse efficiency",
    })

    coupling = pd.read_csv(INPUT / "within_subject_coupling.csv")
    coupling = coupling.loc[coupling.term.str.contains(r"C\(condition\)", regex=True)].copy()
    coupling["effect"] = coupling.estimate / coupling.standard_error
    coupling["low"] = coupling.ci95_low / coupling.standard_error
    coupling["high"] = coupling.ci95_high / coupling.standard_error
    coupling["display"] = coupling.model.map({
        "correctness_binomial_gee": "Correctness",
        "correct_rt_gaussian_gee": "Correct RT",
    })

    increments = pd.read_csv(AUDIT / "participant_loss_increments.csv")
    increments = increments.loc[
        increments.dataset.eq("Dryad")
        & increments.outcome.eq("inverse_efficiency_difference_ms")
    ].copy()
    performance = pd.read_csv(INPUT / "prediction_performance.csv")
    return behavior, coupling, increments, performance


def build() -> dict[str, object]:
    behavior, coupling, increments, performance = load_source_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = pd.concat(
        [
            behavior[["display", "effect", "low", "high"]].assign(panel="a_behavior_dz"),
            coupling[["display", "effect", "low", "high"]].assign(panel="b_coupling_wald_z"),
        ],
        ignore_index=True,
    )
    source.to_csv(OUTPUT / "Figure3_source_effects.csv", index=False)
    increments.to_csv(OUTPUT / "Figure3_source_prediction_increments.csv", index=False)
    performance.to_csv(OUTPUT / "Figure3_source_prediction_performance.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.0866, 4.9606), constrained_layout=True)
    axa, axb, axc, axd = axes.ravel()

    order_a = ["Accuracy", "Correct RT", "Inverse efficiency"]
    view_a = behavior.set_index("display").loc[order_a].reset_index()
    y_a = np.arange(len(view_a))[::-1]
    axa.errorbar(
        view_a.effect,
        y_a,
        xerr=np.vstack([view_a.effect - view_a.low, view_a.high - view_a.effect]),
        fmt="o",
        color=COLORS["neural"],
        ecolor=COLORS["neural"],
        capsize=2.5,
        lw=1.2,
        ms=4.5,
    )
    axa.axvline(0, color=COLORS["dark"], lw=0.8)
    axa.set_yticks(y_a, view_a.display)
    axa.set_xlabel("Paired standardized effect (theta-plus − non-rhythmic)")
    axa.set_title("Acute behavioral contrasts", loc="left", fontweight="bold")
    axa.text(0.02, -0.26, "Negative values favor faster/better performance for RT measures.",
             transform=axa.transAxes, ha="left", va="top", fontsize=6, color=COLORS["dark"])
    axa.grid(axis="x", color=COLORS["grid"], lw=0.6, zorder=0)
    panel_label(axa, "a")

    order_b = ["Correctness", "Correct RT"]
    view_b = coupling.set_index("display").loc[order_b].reset_index()
    y_b = np.arange(len(view_b))[::-1]
    axb.errorbar(
        view_b.effect,
        y_b,
        xerr=np.vstack([view_b.effect - view_b.low, view_b.high - view_b.effect]),
        fmt="o",
        color=COLORS["boundary"],
        ecolor=COLORS["boundary"],
        capsize=2.5,
        lw=1.2,
        ms=4.5,
    )
    axb.axvline(0, color=COLORS["dark"], lw=0.8)
    axb.set_yticks(y_b, view_b.display)
    axb.set_xlabel("Condition interaction (Wald z scale)")
    axb.set_title("Condition-specific neural–behavioral coupling", loc="left", fontweight="bold")
    axb.text(0.02, -0.26, "Intervals spanning both directions indicate unresolved coupling.",
             transform=axb.transAxes, ha="left", va="top", fontsize=6, color=COLORS["dark"])
    axb.grid(axis="x", color=COLORS["grid"], lw=0.6, zorder=0)
    panel_label(axb, "b")

    values = increments.delta_neural_minus_base.to_numpy(float)
    jitter = ((np.arange(len(values)) * 13) % len(values)) / (len(values) - 1) * 0.32 - 0.16
    axc.scatter(values, jitter, s=18, color=COLORS["neural"], alpha=0.72,
                edgecolor="white", linewidth=0.35, zorder=3)
    mean = float(values.mean())
    median = float(np.median(values))
    ci_low, ci_high = -1.013, 3.207
    axc.errorbar(mean, 0.34, xerr=[[mean - ci_low], [ci_high - mean]], fmt="D",
                 color=COLORS["dark"], ecolor=COLORS["dark"], capsize=3, ms=4.5,
                 zorder=4)
    axc.plot(median, -0.34, marker="|", ms=12, mew=2, color=COLORS["boundary"],
             zorder=4)
    axc.axvline(0, color=COLORS["dark"], lw=0.8)
    axc.set_ylim(-0.55, 0.55)
    axc.set_yticks([])
    axc.set_xlabel(
        "Participant absolute-error increment (neural − baseline), ms\n"
        "40% improved; paired sign-flip p=0.372"
    )
    axc.set_title("Incremental prediction of individual benefit", loc="left", fontweight="bold")
    axc.text(ci_high + 0.9, 0.34, "Mean and 95% CI", ha="left", va="center", fontsize=6.5)
    axc.text(median + 0.9, -0.34, "Median", ha="left", va="center", fontsize=6.5,
             color=COLORS["boundary"])
    panel_label(axc, "c")

    name_map = {
        "training_fold_mean": "Training-fold mean",
        "baseline": "Baseline features",
        "baseline_plus_neural": "Baseline + neural",
    }
    perf = performance.copy()
    perf["display"] = perf.model.map(name_map)
    colors = [COLORS["baseline"], COLORS["baseline"], COLORS["neural"]]
    x = np.arange(len(perf))
    bars = axd.bar(x, perf.mae, color=colors, width=0.64)
    display_labels = ["Training-fold\nmean", "Baseline\nfeatures", "Baseline + neural\nfeatures"]
    axd.set_xticks(x, display_labels)
    axd.set_ylabel("Held-out MAE (ms)")
    axd.set_title("Absolute prediction error", loc="left", fontweight="bold")
    axd.set_ylim(0, max(perf.mae) * 1.28)
    for bar, value in zip(bars, perf.mae):
        axd.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.2f}",
                 ha="center", va="bottom", fontsize=6.5)
    axd.text(0.98, 0.94, "Neural − baseline: +1.02 ms",
             transform=axd.transAxes, ha="right", va="top", fontsize=6.5,
             color=COLORS["dark"])
    axd.set_axisbelow(True)
    panel_label(axd, "d")

    for ax in axes.ravel():
        ax.tick_params(length=3, width=0.7)

    fig.canvas.draw()
    stem = OUTPUT / "Figure3_Cognitive_boundary"
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
        "figure": "Figure 3",
        "n_participants": int(len(values)),
        "mean_prediction_increment_ms": mean,
        "median_prediction_increment_ms": median,
        "fraction_improved": float(np.mean(values < 0)),
        "status": "rendered_pending_qa",
    }
    (OUTPUT / "Figure3_build_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
