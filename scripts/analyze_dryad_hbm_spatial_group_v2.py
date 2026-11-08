#!/usr/bin/env python3
"""Run the results-blind v2 wrapper of the frozen Dryad spatial group analysis.

The statistical implementation is identical to the parent-frozen script. The
only analysis-output change is a fixed CSV schema when no spatial cluster is
observed; see amendment hbm_spatial_output_schema_amendment_v1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/models/dryad_hbm_spatial_subjects_v1"
DEFAULT_PROTOCOL = ROOT / "configs/hbm_additional_analysis_protocol_v1.json"
DEFAULT_ADJACENCY = ROOT / "configs/dryad_biosemi64_adjacency_v1.csv"
DEFAULT_OUTPUT = ROOT / "outputs/models/dryad_hbm_spatial_group_v1"
CONDITIONS = ("f_theta_plus", "non_rhythmic")


def load_subject_tables(root: Path, expected_subjects: list[int]) -> pd.DataFrame:
    tables = []
    for subject in expected_subjects:
        path = root / f"S{subject}/channel_condition_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing spatial subject output: {path}")
        table = pd.read_csv(path)
        if set(table["subject"].astype(int)) != {subject}:
            raise ValueError(f"subject identity mismatch in {path}")
        if set(table["condition"].astype(str)) != set(CONDITIONS):
            raise ValueError(f"condition mismatch in {path}")
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    if set(combined["subject"].astype(int)) != set(expected_subjects):
        raise ValueError("combined subject set differs from protocol")
    return combined


def participant_channel_differences(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = table.pivot_table(
        index="subject", columns=["condition", "channel"], values=metric, aggfunc="first"
    )
    theta = pivot.get("f_theta_plus")
    control = pivot.get("non_rhythmic")
    if theta is None or control is None:
        raise ValueError("both active conditions are required")
    common = theta.columns.intersection(control.columns)
    return theta[common] - control[common]


def channel_statistics(differences: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for channel in differences.columns:
        values = differences[channel].dropna().to_numpy(float)
        if len(values) < 3:
            continue
        mean = float(np.mean(values))
        sem = float(stats.sem(values))
        critical = float(stats.t.ppf(0.975, len(values) - 1))
        test = stats.ttest_1samp(values, 0.0)
        rows.append({
            "metric": metric,
            "channel": channel,
            "n_subjects": len(values),
            "mean_theta_plus_minus_non_rhythmic": mean,
            "ci95_low": mean - critical * sem,
            "ci95_high": mean + critical * sem,
            "t_statistic": float(test.statistic),
            "p_value_uncorrected": float(test.pvalue),
        })
    return pd.DataFrame(rows)


def connected_components(nodes: set[str], adjacency: dict[str, set[str]]) -> list[set[str]]:
    remaining = set(nodes)
    components = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = adjacency.get(current, set()) & remaining
            remaining -= neighbours
            component |= neighbours
            frontier.extend(neighbours)
        components.append(component)
    return components


def find_clusters(
    t_values: pd.Series,
    n_values: pd.Series,
    adjacency: dict[str, set[str]],
    alpha: float,
) -> list[dict[str, object]]:
    positive, negative = set(), set()
    for channel, value in t_values.items():
        n = int(n_values[channel])
        if n < 3 or not np.isfinite(value):
            continue
        threshold = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
        if value >= threshold:
            positive.add(str(channel))
        elif value <= -threshold:
            negative.add(str(channel))
    rows = []
    for sign, nodes in ((1, positive), (-1, negative)):
        for component in connected_components(nodes, adjacency):
            mass = float(np.abs(t_values.loc[list(component)]).sum())
            rows.append({
                "sign": sign,
                "channels": sorted(component),
                "n_channels": len(component),
                "cluster_mass": mass,
            })
    return rows


def cluster_permutation(
    differences: pd.DataFrame,
    adjacency: dict[str, set[str]],
    permutations: int,
    seed: int,
    alpha: float = 0.05,
) -> list[dict[str, object]]:
    n_values = differences.notna().sum(axis=0)
    observed_t = differences.apply(
        lambda column: stats.ttest_1samp(column.dropna(), 0.0).statistic
        if column.notna().sum() >= 3 else np.nan,
        axis=0,
    )
    observed = find_clusters(observed_t, n_values, adjacency, alpha)
    if not observed:
        return []
    rng = np.random.default_rng(seed)
    null_max = np.zeros(permutations, dtype=float)
    values = differences.to_numpy(float)
    for index in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=(len(differences), 1))
        permuted = pd.DataFrame(values * signs, columns=differences.columns)
        permuted_t = permuted.apply(
            lambda column: stats.ttest_1samp(column.dropna(), 0.0).statistic
            if column.notna().sum() >= 3 else np.nan,
            axis=0,
        )
        clusters = find_clusters(permuted_t, n_values, adjacency, alpha)
        null_max[index] = max((row["cluster_mass"] for row in clusters), default=0.0)
    for number, row in enumerate(sorted(observed, key=lambda item: -item["cluster_mass"]), 1):
        row["cluster_id"] = number
        row["permutation_p_value"] = float(
            (1 + np.sum(null_max >= float(row["cluster_mass"]))) / (permutations + 1)
        )
    return observed


def load_adjacency(path: Path) -> dict[str, set[str]]:
    table = pd.read_csv(path)
    adjacency: dict[str, set[str]] = {}
    for row in table.itertuples(index=False):
        a, b = str(row.channel_a), str(row.channel_b)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def roi_specificity(table: pd.DataFrame, metric: str, primary: list[str], control: list[str]) -> dict[str, object]:
    selected = table.loc[table["channel"].isin(primary + control)].copy()
    selected["roi"] = np.where(selected["channel"].isin(primary), "frontocentral", "posterior_control")
    counts = selected.groupby(["subject", "condition", "roi"])["channel"].nunique()
    if (counts < 3).any():
        raise ValueError("fewer than three usable channels in a frozen ROI")
    means = selected.groupby(["subject", "condition", "roi"])[metric].mean().unstack(["condition", "roi"])
    front = means[("f_theta_plus", "frontocentral")] - means[("non_rhythmic", "frontocentral")]
    posterior = means[("f_theta_plus", "posterior_control")] - means[("non_rhythmic", "posterior_control")]
    contrast = (front - posterior).dropna().to_numpy(float)
    mean = float(np.mean(contrast))
    sem = float(stats.sem(contrast))
    critical = float(stats.t.ppf(0.975, len(contrast) - 1))
    test = stats.ttest_1samp(contrast, 0.0)
    return {
        "metric": metric,
        "estimand": "frontocentral_condition_contrast_minus_posterior_condition_contrast",
        "n_subjects": len(contrast),
        "mean": mean,
        "ci95_low": mean - critical * sem,
        "ci95_high": mean + critical * sem,
        "paired_dz": mean / float(np.std(contrast, ddof=1)),
        "p_value_two_sided": float(test.pvalue),
    }


def analyze(input_root: Path, protocol_path: Path, adjacency_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    spatial = protocol["dryad_spatial_robustness"]
    subjects = list(map(int, spatial["subjects"]))
    table = load_subject_tables(input_root, subjects)
    adjacency = load_adjacency(adjacency_path)
    channel_rows, cluster_rows, roi_rows = [], [], []
    for metric in spatial["metrics"]:
        differences = participant_channel_differences(table, metric)
        channel_rows.append(channel_statistics(differences, metric))
        clusters = cluster_permutation(
            differences,
            adjacency,
            int(spatial["channel_inference"]["permutations"]),
            int(spatial["channel_inference"]["random_seed"]),
            float(spatial["channel_inference"]["cluster_forming_alpha_two_sided"]),
        )
        for row in clusters:
            cluster_rows.append({**row, "metric": metric, "channels": ";".join(row["channels"])})
        roi_rows.append(roi_specificity(
            table, metric, list(spatial["primary_roi"]), list(spatial["anatomical_control_roi"])
        ))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(channel_rows, ignore_index=True).to_csv(output_dir / "channel_effects.csv", index=False)
    cluster_columns = [
        "metric", "sign", "channels", "n_channels", "cluster_mass",
        "cluster_id", "permutation_p_value",
    ]
    pd.DataFrame(cluster_rows, columns=cluster_columns).to_csv(
        output_dir / "spatial_clusters.csv", index=False
    )
    pd.DataFrame(roi_rows).to_csv(output_dir / "roi_specificity.csv", index=False)
    summary = {
        "protocol_id": protocol["protocol_id"],
        "classification": spatial["classification"],
        "n_subjects": len(subjects),
        "metrics": list(spatial["metrics"]),
        "status": "hbm_spatial_group_analysis_complete",
    }
    (output_dir / "spatial_group_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input_root, args.protocol, args.adjacency, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
