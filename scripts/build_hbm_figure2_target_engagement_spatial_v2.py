#!/usr/bin/env python3
"""Build HBM Figure 2 with a layout-only topomap alignment exemption.

Statistical inputs and visual encodings are identical to the pre-result builder.
The equal-aspect topomap axis and its colorbar are exempted from rectangular
axes-edge alignment while all comparable Cartesian panels remain audited.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mne
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_SUBJECTS = ROOT / "outputs/models/dryad_confirmatory_subjects_v1"
CONFIRMATORY_GROUP = ROOT / "outputs/models/dryad_confirmatory_group_v1"
SPATIAL_SUBJECTS = ROOT / "outputs/models/dryad_hbm_spatial_subjects_v1"
SPATIAL_GROUP = ROOT / "outputs/models/dryad_hbm_spatial_group_v1"
PROTOCOL = ROOT / "configs/hbm_additional_analysis_protocol_v1.json"
OUTPUT = ROOT / "figures/hbm_research_article_v1"
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(*args, **kwargs):
        """Skip optional authoring-time layout audit when its helper is unavailable."""
        return None


COLORS = {
    "control": "#8B959F",
    "theta": "#2C6E9B",
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


def load_complete_inputs() -> tuple[
    dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    subjects = list(map(int, protocol["dryad_spatial_robustness"]["subjects"]))
    condition_tables, channel_tables = [], []
    missing = []
    for subject in subjects:
        condition_path = CONFIRMATORY_SUBJECTS / f"S{subject}/condition_neural_metrics.csv"
        channel_path = SPATIAL_SUBJECTS / f"S{subject}/channel_condition_metrics.csv"
        subject_missing = []
        for path in (condition_path, channel_path):
            if not path.is_file():
                subject_missing.append(str(path.relative_to(ROOT)))
        missing.extend(subject_missing)
        if not subject_missing:
            condition_tables.append(pd.read_csv(condition_path))
            channel_tables.append(pd.read_csv(channel_path))
    required_group = [
        SPATIAL_GROUP / "channel_effects.csv",
        SPATIAL_GROUP / "spatial_clusters.csv",
        SPATIAL_GROUP / "roi_specificity.csv",
    ]
    missing.extend(str(path.relative_to(ROOT)) for path in required_group if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "Figure 2 is locked until all 35 spatial analyses and the frozen group analysis complete: "
            + "; ".join(sorted(set(missing))[:8])
        )
    conditions = pd.concat(condition_tables, ignore_index=True)
    channels = pd.concat(channel_tables, ignore_index=True)
    if conditions.subject.nunique() != 35 or channels.subject.nunique() != 35:
        raise RuntimeError("Figure 2 requires the complete frozen 35-participant cohort")
    neural = pd.read_csv(CONFIRMATORY_GROUP / "paired_neural_effects.csv")
    channel_effects = pd.read_csv(SPATIAL_GROUP / "channel_effects.csv")
    clusters = pd.read_csv(SPATIAL_GROUP / "spatial_clusters.csv")
    roi = pd.read_csv(SPATIAL_GROUP / "roi_specificity.csv")
    return protocol, conditions, channels, neural, channel_effects, clusters, roi


def participant_roi_contrasts(channels: pd.DataFrame, metric: str,
                              primary: list[str], control: list[str]) -> pd.Series:
    selected = channels.loc[channels.channel.isin(primary + control)].copy()
    selected["roi"] = np.where(selected.channel.isin(primary), "frontocentral", "posterior")
    means = selected.groupby(["subject", "condition", "roi"])[metric].mean().unstack(["condition", "roi"])
    front = means[("f_theta_plus", "frontocentral")] - means[("non_rhythmic", "frontocentral")]
    posterior = means[("f_theta_plus", "posterior")] - means[("non_rhythmic", "posterior")]
    contrasts = front - posterior
    complete = contrasts.loc[contrasts.notna()]
    if len(complete) != len(contrasts):
        raise RuntimeError(
            f"{metric}: missing ROI contrast for {len(contrasts) - len(complete)} participant(s)"
        )
    return complete


def build() -> dict[str, object]:
    protocol, conditions, channels, neural, channel_effects, clusters, roi = load_complete_inputs()
    spatial = protocol["dryad_spatial_robustness"]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.0866, 5.1181), constrained_layout=True)
    axa, axb, axc, axd = axes.ravel()

    metric = "mean_itpc_in_stimulation_window"
    paired_all = conditions.pivot(index="subject", columns="condition", values=metric)
    paired = paired_all.loc[paired_all.notna().all(axis=1)]
    if len(paired) != len(paired_all) or len(paired) != 35:
        raise RuntimeError("primary ITPC panel requires 35 complete paired observations")
    primary_effect = neural.set_index("metric").loc[metric]
    for row in paired.itertuples():
        axa.plot([0, 1], [row.non_rhythmic, row.f_theta_plus], color="#B9C0C6", lw=0.6, alpha=0.7)
    axa.scatter(np.zeros(len(paired)), paired.non_rhythmic, s=14, color=COLORS["control"], zorder=3)
    axa.scatter(np.ones(len(paired)), paired.f_theta_plus, s=14, color=COLORS["theta"], zorder=3)
    axa.set_xticks([0, 1], ["Non-rhythmic", "Theta-plus"])
    axa.set_ylabel("Frontocentral ITPC")
    axa.set_title("Frozen primary target engagement", loc="left", fontweight="bold", pad=15)
    axa.text(
        1.0,
        1.015,
        (
            f"Δ={primary_effect.mean_difference_theta_plus_minus_non_rhythmic:.4f} "
            f"[{primary_effect.ci95_low:.4f}, {primary_effect.ci95_high:.4f}]"
        ),
        transform=axa.transAxes,
             ha="right", va="bottom", fontsize=6.5)
    panel_label(axa, "a")

    order = [
        "mean_itpc_in_stimulation_window", "evoked_local_log_snr_db",
        "total_power_local_log_snr_db", "induced_local_log_snr_db",
    ]
    labels = ["ITPC", "Evoked SNR", "Total-power SNR", "Induced SNR"]
    view = neural.set_index("metric").loc[order].reset_index()
    y = np.arange(len(view))[::-1]
    scale = view.mean_difference_theta_plus_minus_non_rhythmic / view.paired_dz
    low = view.ci95_low / scale
    high = view.ci95_high / scale
    axb.errorbar(view.paired_dz, y,
                 xerr=np.vstack([view.paired_dz - low, high - view.paired_dz]),
                 fmt="o", color=COLORS["theta"], ecolor=COLORS["theta"],
                 capsize=2.5, lw=1.1, ms=4.2)
    axb.axvline(0, color=COLORS["dark"], lw=0.8)
    axb.set_yticks(y, labels)
    axb.set_xlabel("Paired standardized effect")
    axb.set_title("Phase-consistent response decomposition", loc="left", fontweight="bold")
    panel_label(axb, "b")

    itpc_channels = channel_effects.loc[channel_effects.metric.eq(metric)].copy()
    montage = mne.channels.make_standard_montage("biosemi64")
    montage_channels = set(montage.ch_names)
    itpc_channels = itpc_channels.loc[itpc_channels.channel.isin(montage_channels)].copy()
    info = mne.create_info(
        ch_names=itpc_channels.channel.astype(str).tolist(), sfreq=1.0, ch_types="eeg"
    )
    info.set_montage(montage, on_missing="raise")
    values = itpc_channels.mean_theta_plus_minus_non_rhythmic.to_numpy(float)
    limit = float(np.max(np.abs(values)))
    cmap = LinearSegmentedColormap.from_list("hbm_diverging", ["#7C5E99", "#F7F7F7", "#2C6E9B"])
    significant_itpc = clusters.loc[
        clusters.metric.eq(metric) & clusters.permutation_p_value.lt(0.05)
    ].copy()
    significant_channels: set[str] = set()
    for channel_list in significant_itpc.channels.dropna().astype(str):
        significant_channels.update(item for item in channel_list.split(";") if item)
    mask = itpc_channels.channel.astype(str).isin(significant_channels).to_numpy()
    image, _ = mne.viz.plot_topomap(
        values, info, axes=axc, show=False, contours=0, sensors=True,
        cmap=cmap, vlim=(-limit, limit), outlines="head", mask=mask,
        mask_params={
            "marker": "o", "markerfacecolor": "none", "markeredgecolor": COLORS["dark"],
            "linestyle": "None", "linewidth": 0.9, "markersize": 5.5,
        },
    )
    colorbar = fig.colorbar(image, ax=axc, shrink=0.72, pad=0.03)
    colorbar.set_label("ITPC contrast")
    axc.set_title("Channel-level theta-plus − non-rhythmic effect", loc="left", fontweight="bold")
    cluster_note = (
        "Outlined sensors: cluster FWE p<0.05"
        if significant_channels else "No channel cluster survived FWE p<0.05"
    )
    axc.text(0.5, -0.08, cluster_note, transform=axc.transAxes,
             ha="center", va="top", fontsize=6, color=COLORS["dark"])
    panel_label(axc, "c")

    primary, control = list(spatial["primary_roi"]), list(spatial["anatomical_control_roi"])
    roi_metrics = [metric, "evoked_local_log_snr_db"]
    roi_labels = ["ITPC", "Evoked SNR"]
    for ypos, roi_metric, label, color in zip([1, 0], roi_metrics, roi_labels,
                                               [COLORS["theta"], COLORS["boundary"]]):
        values_raw = participant_roi_contrasts(channels, roi_metric, primary, control)
        if len(values_raw) != 35:
            raise RuntimeError(f"{roi_metric}: expected 35 complete ROI contrasts")
        values_z = values_raw / values_raw.std(ddof=1)
        rng_index = np.arange(len(values_z))
        jitter = ((rng_index * 13) % len(rng_index)) / (len(rng_index) - 1) * 0.28 - 0.14
        axd.scatter(values_z, ypos + jitter, s=14, color=color, alpha=0.72,
                    edgecolor="white", linewidth=0.3)
        mean = float(values_z.mean())
        sem = float(values_z.std(ddof=1) / np.sqrt(len(values_z)))
        critical = float(stats.t.ppf(0.975, len(values_z) - 1))
        axd.errorbar(mean, ypos + 0.28, xerr=critical * sem, fmt="D", color=COLORS["dark"],
                     ecolor=COLORS["dark"], capsize=2.5, ms=4)
    axd.axvline(0, color=COLORS["dark"], lw=0.8)
    axd.set_yticks([1, 0], roi_labels)
    axd.set_xlabel("Frontocentral-minus-posterior condition contrast (SD)")
    axd.set_title("Frozen anatomical-control comparison", loc="left", fontweight="bold")
    panel_label(axd, "d")

    conditions.to_csv(OUTPUT / "Figure2_source_condition_metrics.csv", index=False)
    channels.to_csv(OUTPUT / "Figure2_source_channel_condition_metrics.csv", index=False)
    channel_effects.to_csv(OUTPUT / "Figure2_source_channel_effects.csv", index=False)
    clusters.to_csv(OUTPUT / "Figure2_source_spatial_clusters.csv", index=False)
    roi.to_csv(OUTPUT / "Figure2_source_roi_specificity.csv", index=False)
    fig.canvas.draw()
    stem = OUTPUT / "Figure2_Target_engagement_spatial"
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        exclude_axes=[axc, colorbar.ax],
    )
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    result = {"figure": "Figure 2", "n_participants": 35, "status": "rendered_pending_qa"}
    (OUTPUT / "Figure2_build_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
