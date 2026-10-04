#!/usr/bin/env python3
"""Develop the condition-blind spatial channel-QC component for Dryad R2b.

This module is restricted to the five-participant development stage. It builds
a deterministic BioSemi64 neighbour graph and evaluates channel consistency
against spatial neighbours rather than the global scalp median. It deliberately
does not read trial conditions, compute target-frequency responses, or produce
behavioural associations.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal

try:
    from scripts.qc_dryad_signal_pilot import (
        EXTERNAL_CHANNELS,
        channel_metrics,
        robust_z,
    )
except ModuleNotFoundError:  # Direct execution via ``python scripts/...``.
    from qc_dryad_signal_pilot import EXTERNAL_CHANNELS, channel_metrics, robust_z


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADJACENCY = ROOT / "configs/dryad_biosemi64_adjacency_v1.csv"
DEFAULT_METADATA = ROOT / "configs/dryad_biosemi64_adjacency_v1.json"
DEFAULT_NEAREST_NEIGHBOURS = 4
DEVELOPMENT_NEIGHBOUR_CORRELATION_FLOOR = 0.20
DEVELOPMENT_BRIDGE_CORRELATION_CEILING = 0.9995
DEVELOPMENT_ROBUST_Z_LIMIT = 5.0
PROHIBITED_TECHNICAL_OUTPUT_TOKENS = (
    "condition",
    "response",
    "rt_ms",
    "theta",
    "target_frequency",
    "behaviour",
    "behavior",
    "accuracy",
)


def biosemi64_positions() -> dict[str, np.ndarray]:
    """Return normalized 3-D positions for the standard BioSemi64 montage."""
    montage = mne.channels.make_standard_montage("biosemi64")
    positions = montage.get_positions()["ch_pos"]
    if len(positions) != 64:
        raise RuntimeError(f"expected 64 BioSemi channels, found {len(positions)}")
    normalized: dict[str, np.ndarray] = {}
    for channel, position in positions.items():
        vector = np.asarray(position, dtype=float)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"invalid BioSemi64 position for {channel}")
        normalized[channel] = vector / norm
    return normalized


def build_adjacency(nearest_neighbours: int = DEFAULT_NEAREST_NEIGHBOURS) -> pd.DataFrame:
    """Build a symmetric graph from fixed angular k-nearest neighbours.

    Each channel first selects exactly ``nearest_neighbours`` channels using
    angular distance on normalized standard-montage coordinates. The final
    undirected graph is the union of those directed selections, so every channel
    has at least k neighbours without data-dependent topology.
    """
    if nearest_neighbours < 1:
        raise ValueError("nearest_neighbours must be positive")
    positions = biosemi64_positions()
    channels = sorted(positions)
    if nearest_neighbours >= len(channels):
        raise ValueError("nearest_neighbours must be smaller than channel count")
    distances: dict[tuple[str, str], float] = {}
    for channel in channels:
        candidates = []
        for other in channels:
            if other == channel:
                continue
            cosine = float(np.clip(np.dot(positions[channel], positions[other]), -1, 1))
            candidates.append((float(np.arccos(cosine)), other))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for distance, other in candidates[:nearest_neighbours]:
            edge = tuple(sorted((channel, other)))
            distances[edge] = min(distance, distances.get(edge, np.inf))
    rows = [
        {
            "channel_a": edge[0],
            "channel_b": edge[1],
            "angular_distance_radians": distance,
        }
        for edge, distance in sorted(distances.items())
    ]
    result = pd.DataFrame(rows)
    validate_adjacency(result, expected_minimum_degree=nearest_neighbours)
    return result


def adjacency_map(edges: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Convert an undirected edge table to a sorted neighbour map."""
    required = {"channel_a", "channel_b"}
    if not required.issubset(edges.columns):
        raise ValueError(f"adjacency table missing columns: {sorted(required - set(edges))}")
    neighbours: dict[str, set[str]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        a = str(row.channel_a)
        b = str(row.channel_b)
        if a == b:
            raise ValueError(f"self-edge is not allowed: {a}")
        neighbours[a].add(b)
        neighbours[b].add(a)
    return {channel: tuple(sorted(values)) for channel, values in sorted(neighbours.items())}


def validate_adjacency(edges: pd.DataFrame, expected_minimum_degree: int = 4) -> None:
    """Require an exact, symmetric-by-construction BioSemi64 graph."""
    neighbours = adjacency_map(edges)
    expected = set(biosemi64_positions())
    observed = set(neighbours)
    if observed != expected:
        raise ValueError(
            f"adjacency channels differ from BioSemi64; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    duplicate_edges = edges.duplicated(["channel_a", "channel_b"]).any()
    if duplicate_edges:
        raise ValueError("adjacency contains duplicate edges")
    if (edges["channel_a"] >= edges["channel_b"]).any():
        raise ValueError("each edge must use canonical channel_a < channel_b order")
    minimum_degree = min(len(values) for values in neighbours.values())
    if minimum_degree < expected_minimum_degree:
        raise ValueError(
            f"minimum adjacency degree {minimum_degree} is below {expected_minimum_degree}"
        )


def pairwise_finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation using finite paired samples, or NaN if undefined."""
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return np.nan
    left_finite = left[finite]
    right_finite = right[finite]
    if np.std(left_finite) == 0 or np.std(right_finite) == 0:
        return np.nan
    return float(np.corrcoef(left_finite, right_finite)[0, 1])


def spatial_correlation_metrics(
    data: np.ndarray,
    channel_names: list[str],
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate condition-blind local and global channel correlations.

    ``data`` has shape channels x samples and is expected to be technically
    preprocessed in a condition-blind manner before this function is called.
    The global correlation is retained only as a diagnostic; R2b candidate
    flagging uses the local-neighbour statistic.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(channel_names):
        raise ValueError("data must have shape len(channel_names) x samples")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("channel_names must be unique")
    neighbours = adjacency_map(edges)
    if set(channel_names) != set(neighbours):
        raise ValueError("data channels must match the frozen adjacency exactly")
    channel_index = {channel: index for index, channel in enumerate(channel_names)}
    rows = []
    for channel in channel_names:
        index = channel_index[channel]
        local_pairs = [
            (
                other,
                pairwise_finite_correlation(values[index], values[channel_index[other]]),
            )
            for other in neighbours[channel]
        ]
        local_values = [correlation for _, correlation in local_pairs]
        finite_local_pairs = [
            (other, correlation)
            for other, correlation in local_pairs
            if np.isfinite(correlation)
        ]
        closest_positive = max(
            finite_local_pairs,
            key=lambda pair: (pair[1], pair[0]),
            default=("", np.nan),
        )
        global_values = [
            pairwise_finite_correlation(values[index], values[other_index])
            for other, other_index in channel_index.items()
            if other != channel
        ]
        rows.append(
            {
                "channel": channel,
                "n_spatial_neighbours": len(neighbours[channel]),
                "median_neighbor_correlation": float(np.nanmedian(local_values)),
                "minimum_neighbor_correlation": float(np.nanmin(local_values)),
                "maximum_neighbor_correlation": float(closest_positive[1]),
                "maximum_correlation_neighbor": closest_positive[0],
                "median_global_correlation_diagnostic": float(np.nanmedian(global_values)),
            }
        )
    return pd.DataFrame(rows)


def condition_blind_spatial_data(
    raw: mne.io.BaseRaw,
    scalp_names: list[str],
) -> np.ndarray:
    """Sample, rereference, resample, and band-pass continuous signal for QC.

    Sampling windows are fixed across the complete recording and do not use
    events or conditions. The resulting matrix is channels x concatenated
    samples and is used only for technical spatial-consistency metrics.
    """
    sfreq = float(raw.info["sfreq"])
    window_samples = int(round(10.0 * sfreq))
    latest = max(0, raw.n_times - window_samples)
    starts = np.unique(np.linspace(0, latest, 18, dtype=int))
    chunks = []
    for start in starts:
        chunk = raw.get_data(
            picks=scalp_names,
            start=int(start),
            stop=int(start + window_samples),
        )
        chunk = chunk * 1e6
        chunk = chunk - np.nanmean(chunk, axis=0, keepdims=True)
        chunk = chunk - np.nanmedian(chunk, axis=1, keepdims=True)
        chunk = signal.resample_poly(chunk, 1, 4, axis=1)
        chunks.append(chunk)
    data = np.concatenate(chunks, axis=1)
    sfreq_qc = sfreq / 4.0
    sos = signal.butter(4, [1.0, 45.0], btype="bandpass", fs=sfreq_qc, output="sos")
    return signal.sosfiltfilt(sos, data, axis=1)


def flag_channels_r2b_development(metrics: pd.DataFrame) -> pd.DataFrame:
    """Apply the explicitly provisional R2b channel rules.

    Fixed artifact rules are inherited from R2. Only the correlation reference
    changes from the global scalp to the fixed spatial-neighbour statistic. The
    thresholds remain development candidates until the five-person calibration
    and synthetic-fault suite are frozen.
    """
    result = metrics.copy()
    required = {
        "finite_fraction",
        "rms_uv",
        "line_noise_db",
        "high_frequency_db",
        "median_neighbor_correlation",
        "maximum_neighbor_correlation",
    }
    if not required.issubset(result.columns):
        raise ValueError(f"metrics missing columns: {sorted(required - set(result))}")
    result["z_log_rms"] = robust_z(np.log(np.maximum(result["rms_uv"], 1e-12)))
    result["z_line_noise_db"] = robust_z(result["line_noise_db"])
    result["z_high_frequency_db"] = robust_z(result["high_frequency_db"])
    result["z_low_neighbor_correlation"] = robust_z(
        -result["median_neighbor_correlation"]
    )
    reasons = []
    for row in result.itertuples(index=False):
        row_reasons = []
        if row.finite_fraction < 0.999:
            row_reasons.append("nonfinite")
        if row.rms_uv < 0.1:
            row_reasons.append("flat")
        if abs(row.z_log_rms) > DEVELOPMENT_ROBUST_Z_LIMIT:
            row_reasons.append("rms_outlier")
        if row.z_line_noise_db > DEVELOPMENT_ROBUST_Z_LIMIT and row.line_noise_db > 3.0:
            row_reasons.append("line_noise_outlier")
        if (
            row.z_high_frequency_db > DEVELOPMENT_ROBUST_Z_LIMIT
            and row.high_frequency_db > -3.0
        ):
            row_reasons.append("high_frequency_outlier")
        if (
            row.z_low_neighbor_correlation > DEVELOPMENT_ROBUST_Z_LIMIT
            and row.median_neighbor_correlation
            < DEVELOPMENT_NEIGHBOUR_CORRELATION_FLOOR
        ):
            row_reasons.append("low_neighbor_correlation_outlier")
        if row.maximum_neighbor_correlation > DEVELOPMENT_BRIDGE_CORRELATION_CEILING:
            row_reasons.append("possible_spatial_bridge")
        reasons.append(";".join(row_reasons))
    result["r2b_bad_channel_reason_development"] = reasons
    result["r2b_bad_channel_development"] = result[
        "r2b_bad_channel_reason_development"
    ].ne("")
    return result


def r2b_channel_metrics(
    raw: mne.io.BaseRaw,
    scalp_names: list[str],
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Create condition-blind R2b development metrics for one recording."""
    base = channel_metrics(raw, scalp_names).drop(
        columns=[
            "z_log_rms",
            "z_line_noise_db",
            "z_high_frequency_db",
            "z_low_correlation",
            "bad_channel_reason",
            "bad_channel",
        ]
    )
    spatial_data = condition_blind_spatial_data(raw, scalp_names)
    spatial = spatial_correlation_metrics(spatial_data, scalp_names, edges)
    combined = base.merge(spatial, on="channel", how="inner", validate="one_to_one")
    if len(combined) != 64:
        raise RuntimeError(f"expected 64 merged channel rows, found {len(combined)}")
    return flag_channels_r2b_development(combined)


def require_technical_only_schema(
    table: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    """Reject any result-bearing field before writing development outputs."""
    labels = [str(column).lower() for column in table.columns]
    labels.extend(str(key).lower() for key in summary)
    violations = sorted(
        {
            token
            for label in labels
            for token in PROHIBITED_TECHNICAL_OUTPUT_TOKENS
            if token in label
        }
    )
    if violations:
        raise RuntimeError(
            f"technical-only output schema contains prohibited result fields: {violations}"
        )


def run_bdf_development_qc(
    bdf_path: Path,
    output_dir: Path,
    adjacency_path: Path = DEFAULT_ADJACENCY,
) -> dict[str, object]:
    """Run channel-only development QC without reading events or behaviour."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    edges = pd.read_csv(adjacency_path)
    validate_adjacency(edges, expected_minimum_degree=DEFAULT_NEAREST_NEIGHBOURS)
    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose="error")
    scalp_names = [
        name for name in raw.ch_names if name not in EXTERNAL_CHANNELS | {"Status"}
    ]
    if len(scalp_names) != 64:
        raise RuntimeError(f"expected 64 scalp channels, found {len(scalp_names)}")
    metrics = r2b_channel_metrics(raw, scalp_names, edges)
    output_dir.mkdir(parents=True, exist_ok=True)
    bad = metrics.loc[metrics["r2b_bad_channel_development"], "channel"].tolist()
    summary = {
        "source_bdf_name": bdf_path.name,
        "n_scalp_channels": len(scalp_names),
        "n_bad_channels_r2b_development": len(bad),
        "bad_channels_r2b_development": bad,
        "adjacency_file": adjacency_path.name,
        "correlation_reference": "median fixed BioSemi64 spatial-neighbour correlation",
        "input_scope": "continuous_signal_and_fixed_montage_only",
        "result_bearing_fields_emitted": False,
        "status": "development_only_not_confirmatory_freeze",
    }
    require_technical_only_schema(metrics, summary)
    metrics.to_csv(output_dir / "channel_metrics_r2b_development.csv", index=False)
    (output_dir / "technical_summary_r2b_development.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def write_frozen_adjacency(
    csv_path: Path = DEFAULT_ADJACENCY,
    metadata_path: Path = DEFAULT_METADATA,
    nearest_neighbours: int = DEFAULT_NEAREST_NEIGHBOURS,
) -> dict[str, object]:
    """Write the deterministic graph and a non-result-bearing metadata record."""
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite an existing adjacency freeze")
    edges = build_adjacency(nearest_neighbours)
    neighbours = adjacency_map(edges)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(csv_path, index=False, float_format="%.12f")
    metadata = {
        "identity": "Dryad R2b BioSemi64 spatial adjacency development v1",
        "montage": "MNE standard montage biosemi64",
        "coordinate_transform": "normalize each 3-D channel vector to unit length",
        "distance": "angular distance arccos(clipped dot product)",
        "directed_selection_k": int(nearest_neighbours),
        "symmetrization": "undirected union of channel-wise k-nearest selections",
        "n_channels": len(neighbours),
        "n_undirected_edges": int(len(edges)),
        "minimum_degree": min(len(values) for values in neighbours.values()),
        "maximum_degree": max(len(values) for values in neighbours.values()),
        "information_boundary": (
            "fixed standard coordinates only; no participant signal, trial condition, "
            "target-frequency response, behaviour, or outcome used"
        ),
        "status": "development_component_not_yet_full_r2b_freeze",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-adjacency", action="store_true")
    parser.add_argument("--bdf", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_ADJACENCY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--nearest-neighbours", type=int, default=DEFAULT_NEAREST_NEIGHBOURS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_adjacency:
        metadata = write_frozen_adjacency(
            args.output_csv, args.output_json, args.nearest_neighbours
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if args.bdf is not None and args.output_dir is not None:
        summary = run_bdf_development_qc(
            args.bdf, args.output_dir, args.output_csv
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    raise SystemExit(
        "no action selected; use --write-adjacency or provide --bdf and --output-dir"
    )


if __name__ == "__main__":
    main()
