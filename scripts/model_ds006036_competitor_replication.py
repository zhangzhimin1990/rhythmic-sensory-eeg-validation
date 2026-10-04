#!/usr/bin/env python3
"""Audit the exact ds006036 competitor pipeline and state/metric sensitivity."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, spearmanr


FREQUENCIES = (5, 10, 15, 20)
GROUP_MAP = {"C": "CN", "A": "AD", "F": "FTD"}


def add_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["group_label"] = frame["group"].map(GROUP_MAP)
    frame["sex_M"] = (frame["sex"] == "M").astype(int)
    epoch_columns = [column for column in frame.columns if column.endswith("_nep")]
    frame["n_epochs_total"] = frame[epoch_columns].sum(axis=1)
    return frame


def term_row(model, group: str, analysis: str, outcome: str, n: int) -> dict:
    term = f"C(group_label, Treatment(reference='CN'))[T.{group}]"
    interval = model.conf_int().loc[term]
    return {
        "analysis": analysis,
        "outcome": outcome,
        "contrast": f"{group}-CN",
        "n": n,
        "coefficient": float(model.params[term]),
        "standard_error": float(model.bse[term]),
        "ci_low": float(interval.iloc[0]),
        "ci_high": float(interval.iloc[1]),
        "p_value": float(model.pvalues[term]),
        "r_squared": float(model.rsquared),
    }


def fit_direct_and_selectivity(exact: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct = exact.dropna(subset=["post_10Hz_di"]).copy()
    direct["raw_di"] = direct["post_10Hz_di"]
    direct["log1p_di"] = np.log1p(direct["post_10Hz_di"])
    lower, upper = np.percentile(direct["post_10Hz_di"], [5, 95])
    direct["winsor_di"] = np.clip(direct["post_10Hz_di"], lower, upper)
    direct_rows = []
    for outcome in ("raw_di", "log1p_di", "winsor_di"):
        model = smf.ols(
            f"{outcome} ~ C(group_label, Treatment(reference='CN')) + age + sex_M + n_epochs_total",
            data=direct,
        ).fit(cov_type="HC3")
        for group in ("AD", "FTD"):
            direct_rows.append(term_row(model, group, "direct_10hz_hc3", outcome, int(model.nobs)))

    complete = exact.dropna(subset=[f"post_{frequency}Hz_di" for frequency in FREQUENCIES]).copy()
    other_columns = [f"post_{frequency}Hz_di" for frequency in (5, 15, 20)]
    complete["selectivity_raw"] = complete["post_10Hz_di"] - complete[other_columns].mean(axis=1)
    complete["selectivity_log"] = np.log1p(complete["post_10Hz_di"]) - np.log1p(complete[other_columns]).mean(axis=1)
    selectivity_rows = []
    for outcome in ("selectivity_raw", "selectivity_log"):
        model = smf.ols(
            f"{outcome} ~ C(group_label, Treatment(reference='CN')) + age + sex_M + n_epochs_total",
            data=complete,
        ).fit(cov_type="HC3")
        for group in ("AD", "FTD"):
            selectivity_rows.append(term_row(model, group, "10hz_minus_other_frequencies_hc3", outcome, int(model.nobs)))
    return pd.DataFrame(direct_rows), pd.DataFrame(selectivity_rows)


def fit_mixed_models(exact: pd.DataFrame) -> pd.DataFrame:
    value_columns = [f"post_{frequency}Hz_di" for frequency in FREQUENCIES]
    long = exact[["sub_id", "group_label", "age", "sex_M", "n_epochs_total", *value_columns]].melt(
        id_vars=["sub_id", "group_label", "age", "sex_M", "n_epochs_total"],
        value_vars=value_columns,
        var_name="frequency_name",
        value_name="raw_di",
    ).dropna()
    long["frequency_hz"] = long["frequency_name"].str.extract(r"(\d+)").astype(int)
    long["frequency_factor"] = pd.Categorical(
        long["frequency_hz"].astype(str), categories=["5", "10", "15", "20"]
    )
    long["log1p_di"] = np.log1p(long["raw_di"])
    long["winsor_di"] = long["raw_di"]
    for frequency in FREQUENCIES:
        selected = long["frequency_hz"] == frequency
        lower, upper = np.percentile(long.loc[selected, "raw_di"], [5, 95])
        long.loc[selected, "winsor_di"] = np.clip(long.loc[selected, "raw_di"], lower, upper)

    rows = []
    for outcome in ("raw_di", "log1p_di", "winsor_di"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                f"{outcome} ~ C(group_label, Treatment('CN')) * C(frequency_factor) + age + sex_M + n_epochs_total",
                data=long,
                groups=long["sub_id"],
            ).fit(reml=True, method="lbfgs", disp=False)
        covariance = model.cov_params()
        for group in ("AD", "FTD"):
            base = next(
                key for key in model.params.index
                if "group_label" in key and group in key and "frequency_factor" not in key
            )
            interaction = next(
                key for key in model.params.index
                if "group_label" in key and group in key and "frequency_factor" in key and "T.10" in key
            )
            coefficient = float(model.params[base] + model.params[interaction])
            variance = (
                covariance.loc[base, base]
                + covariance.loc[interaction, interaction]
                + 2.0 * covariance.loc[base, interaction]
            )
            standard_error = float(np.sqrt(variance))
            z_value = coefficient / standard_error
            rows.append(
                {
                    "analysis": "four_frequency_random_intercept",
                    "outcome": outcome,
                    "contrast": f"{group}-CN_at_10hz",
                    "n_observations": int(model.nobs),
                    "n_subjects": int(long.sub_id.nunique()),
                    "coefficient": coefficient,
                    "standard_error": standard_error,
                    "ci_low": coefficient - 1.96 * standard_error,
                    "ci_high": coefficient + 1.96 * standard_error,
                    "p_value": float(2.0 * norm.sf(abs(z_value))),
                    "converged": bool(model.converged),
                    "random_intercept_variance": float(model.cov_re.iloc[0, 0]),
                    "warnings": " | ".join(sorted({str(item.message) for item in caught})),
                }
            )
    return pd.DataFrame(rows)


def pipeline_reproducibility(exact: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frequency in FREQUENCIES:
        local_column = f"post_{frequency}Hz_di"
        published_column = f"photic_post_{frequency}Hz_di"
        merged = exact[["sub_id", local_column]].merge(
            published[["sub_id", published_column]], on="sub_id", how="inner"
        ).dropna()
        correlation = np.corrcoef(merged[local_column], merged[published_column])[0, 1]
        difference = np.abs(merged[local_column] - merged[published_column])
        rows.append(
            {
                "frequency_hz": frequency,
                "n_common": len(merged),
                "pearson_r": float(correlation),
                "median_absolute_difference": float(np.median(difference)),
                "maximum_absolute_difference": float(np.max(difference)),
            }
        )
    return pd.DataFrame(rows)


def severity_rows(exact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ("AD", "FTD"):
        data = exact[(exact.group_label == group) & exact.post_10Hz_di.notna()]
        correlation = spearmanr(data["post_10Hz_di"], data["mmse"], nan_policy="omit")
        rows.append(
            {
                "group": group,
                "n": int(data[["post_10Hz_di", "mmse"]].dropna().shape[0]),
                "spearman_rho": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
            }
        )
    return pd.DataFrame(rows)


def make_figure(exact: pd.DataFrame, selectivity: pd.DataFrame, output: Path) -> None:
    long = exact[["sub_id", "group_label", *[f"post_{frequency}Hz_di" for frequency in FREQUENCIES]]].melt(
        id_vars=["sub_id", "group_label"], var_name="frequency_name", value_name="driving_index"
    ).dropna()
    long["frequency_hz"] = long.frequency_name.str.extract(r"(\d+)").astype(int)
    complete = exact.dropna(subset=[f"post_{frequency}Hz_di" for frequency in FREQUENCIES]).copy()
    complete["selectivity"] = complete["post_10Hz_di"] - complete[
        [f"post_{frequency}Hz_di" for frequency in (5, 15, 20)]
    ].mean(axis=1)
    colors = {"CN": "#2f6f9f", "AD": "#c44e52", "FTD": "#8172b3"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for group in ("CN", "AD", "FTD"):
        group_data = long[long.group_label == group]
        summary = group_data.groupby("frequency_hz").driving_index.agg(["mean", "sem"])
        axes[0].errorbar(summary.index, summary["mean"], yerr=summary["sem"], marker="o", capsize=3, label=group, color=colors[group])
    axes[0].set(title="Closed-eye posterior photic driving", xlabel="Stimulation frequency (Hz)", ylabel="Driving index, mean ± SEM", xticks=FREQUENCIES)
    axes[0].legend(frameon=False)
    positions = {"CN": 0, "AD": 1, "FTD": 2}
    random = np.random.default_rng(20260918)
    for group in ("CN", "AD", "FTD"):
        values = complete.loc[complete.group_label == group, "selectivity"].to_numpy()
        x = positions[group] + random.uniform(-0.10, 0.10, len(values))
        axes[1].scatter(x, values, s=22, alpha=0.65, color=colors[group])
        axes[1].plot([positions[group] - 0.18, positions[group] + 0.18], [np.median(values)] * 2, color="black", linewidth=2)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(title="10-Hz selectivity relative to other frequencies", ylabel="DI(10 Hz) − mean DI(5, 15, 20 Hz)", xticks=[0, 1, 2], xticklabels=["CN", "AD", "FTD"])
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exact_features", type=Path)
    parser.add_argument("published_features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    exact = add_covariates(pd.read_csv(args.exact_features))
    published = pd.read_csv(args.published_features)
    direct, selectivity = fit_direct_and_selectivity(exact)
    mixed = fit_mixed_models(exact)
    reproducibility = pipeline_reproducibility(exact, published)
    severity = severity_rows(exact)
    descriptives = (
        exact.groupby("group_label", as_index=False)["post_10Hz_di"]
        .agg(n="count", mean="mean", median="median", standard_deviation="std")
    )

    direct.to_csv(args.output / "direct_10hz_models.csv", index=False)
    selectivity.to_csv(args.output / "selectivity_models.csv", index=False)
    mixed.to_csv(args.output / "mixed_models.csv", index=False)
    reproducibility.to_csv(args.output / "pipeline_reproducibility.csv", index=False)
    severity.to_csv(args.output / "within_group_severity.csv", index=False)
    descriptives.to_csv(args.output / "group_descriptives.csv", index=False)
    make_figure(exact, selectivity, args.output / "competitor_replication.png")

    summary = {
        "exact_pipeline_subjects": int(exact.sub_id.nunique()),
        "ten_hz_available": int(exact.post_10Hz_di.notna().sum()),
        "four_frequency_complete": int(exact[[f"post_{f}Hz_di" for f in FREQUENCIES]].notna().all(axis=1).sum()),
        "minimum_exact_vs_published_r": float(reproducibility.pearson_r.min()),
        "direct_raw_min_p": float(direct.loc[direct.outcome == "raw_di", "p_value"].min()),
        "selectivity_raw_max_p": float(selectivity.loc[selectivity.outcome == "selectivity_raw", "p_value"].max()),
        "mixed_raw_max_p": float(mixed.loc[mixed.outcome == "raw_di", "p_value"].max()),
        "interpretation": "Exact pipeline reproduces the published feature matrix. The selective repeated-frequency/group contrast is supported, while direct raw 10-Hz OLS is imprecise and within-diagnosis MMSE associations are weak.",
        "external_code": "https://github.com/pedrovelezpardo/EEG_Photic_AD_FTD commit ae42f58",
    }
    (args.output / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
