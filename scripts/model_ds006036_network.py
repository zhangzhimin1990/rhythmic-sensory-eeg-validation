#!/usr/bin/env python3
"""Frozen foundation models for ds006036 temporal/network candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


PRIMARY_OUTCOMES = (
    "posterior_minus_frontal_snr_db",
    "csd_posterior_frontal_abs_imcoh",
)
EXPLORATORY_OUTCOMES = (
    "late_minus_early_snr_db",
    "csd_posterior_frontal_dwpli2",
)
REST_PREDICTORS = (
    "posterior_iaf_hz",
    "slowing_ratio",
    "spectral_exponent",
    "relative_alpha",
)


def zscore(frame: pd.DataFrame, column: str, target: str) -> None:
    standard_deviation = frame[column].std()
    frame[target] = (
        (frame[column] - frame[column].mean()) / standard_deviation
        if standard_deviation > 0
        else 0.0
    )


def fdr_by_family(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["q_bh"] = np.nan
    for _, indices in frame.groupby("family").groups.items():
        frame.loc[indices, "q_bh"] = multipletests(
            frame.loc[indices, "p_value"], method="fdr_bh"
        )[1]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("network_features", type=Path)
    parser.add_argument("resting_features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    network = pd.read_csv(args.network_features)
    resting = pd.read_csv(args.resting_features)

    reference_rows = []
    for frequency, data in network.groupby("frequency_hz"):
        for metric in ("posterior_frontal_abs_imcoh", "posterior_frontal_dwpli2"):
            variants = {
                "average": metric,
                "native": f"native_{metric}",
                "csd": f"csd_{metric}",
            }
            for first, second in (("average", "native"), ("average", "csd"), ("native", "csd")):
                correlation = spearmanr(
                    data[variants[first]], data[variants[second]], nan_policy="omit"
                )
                reference_rows.append(
                    {
                        "frequency_hz": frequency,
                        "metric": metric,
                        "comparison": f"{first}_vs_{second}",
                        "n": int(data[[variants[first], variants[second]]].dropna().shape[0]),
                        "spearman_rho": float(correlation.statistic),
                        "p_value": float(correlation.pvalue),
                    }
                )
    reference = pd.DataFrame(reference_rows)
    reference.to_csv(args.output / "reference_sensitivity.csv", index=False)

    severity_rows = []
    patients = network[network.group != "C"].copy()
    for frequency, frequency_data in patients.groupby("frequency_hz"):
        for family, outcomes in (
            ("severity_primary", PRIMARY_OUTCOMES),
            ("severity_exploratory", EXPLORATORY_OUTCOMES),
        ):
            for outcome in outcomes:
                data = frequency_data.copy()
                zscore(data, "mmse", "z_mmse")
                zscore(data, "age", "z_age")
                zscore(data, "posterior_snr_db", "z_local_snr")
                model = smf.ols(
                    f"{outcome} ~ z_mmse + z_age + z_local_snr + C(gender) + C(group)",
                    data=data,
                ).fit(cov_type="HC3")
                severity_rows.append(
                    {
                        "family": family,
                        "frequency_hz": frequency,
                        "outcome": outcome,
                        "n": int(model.nobs),
                        "coefficient_per_mmse_sd": float(model.params["z_mmse"]),
                        "standard_error": float(model.bse["z_mmse"]),
                        "p_value": float(model.pvalues["z_mmse"]),
                        "r_squared": float(model.rsquared),
                    }
                )
    severity = fdr_by_family(pd.DataFrame(severity_rows))
    severity.to_csv(args.output / "patient_severity_models.csv", index=False)

    group_rows = []
    for frequency, frequency_data in network.groupby("frequency_hz"):
        for family, outcomes in (
            ("group_primary", PRIMARY_OUTCOMES),
            ("group_exploratory", EXPLORATORY_OUTCOMES),
        ):
            for outcome in outcomes:
                data = frequency_data.copy()
                zscore(data, "age", "z_age")
                zscore(data, "posterior_snr_db", "z_local_snr")
                model = smf.ols(
                    f"{outcome} ~ z_age + z_local_snr + C(gender) + C(group)",
                    data=data,
                ).fit(cov_type="HC3")
                names = list(model.params.index)
                group_terms = [name for name in names if name.startswith("C(group)")]
                restriction = np.zeros((len(group_terms), len(names)))
                for row_index, term in enumerate(group_terms):
                    restriction[row_index, names.index(term)] = 1.0
                test = model.wald_test(restriction, scalar=True)
                group_rows.append(
                    {
                        "family": family,
                        "frequency_hz": frequency,
                        "outcome": outcome,
                        "n": int(model.nobs),
                        "wald_statistic": float(test.statistic),
                        "degrees_of_freedom": len(group_terms),
                        "p_value": float(test.pvalue),
                        "r_squared": float(model.rsquared),
                    }
                )
    groups = fdr_by_family(pd.DataFrame(group_rows))
    groups.to_csv(args.output / "diagnosis_group_models.csv", index=False)

    merged = network.merge(
        resting.drop(columns=["group", "age", "mmse"], errors="ignore"),
        on="subject",
        how="left",
        validate="many_to_one",
    )
    predictor_rows = []
    for frequency, frequency_data in merged.groupby("frequency_hz"):
        for outcome in PRIMARY_OUTCOMES:
            for predictor in REST_PREDICTORS:
                data = frequency_data.copy()
                zscore(data, predictor, "z_predictor")
                zscore(data, "age", "z_age")
                zscore(data, "posterior_snr_db", "z_local_snr")
                model = smf.ols(
                    f"{outcome} ~ z_predictor + z_age + z_local_snr + C(gender) + C(group)",
                    data=data,
                ).fit(cov_type="HC3")
                predictor_rows.append(
                    {
                        "family": "rest_to_advanced_visual",
                        "frequency_hz": frequency,
                        "outcome": outcome,
                        "predictor": predictor,
                        "n": int(model.nobs),
                        "coefficient_per_predictor_sd": float(
                            model.params["z_predictor"]
                        ),
                        "standard_error": float(model.bse["z_predictor"]),
                        "p_value": float(model.pvalues["z_predictor"]),
                        "r_squared": float(model.rsquared),
                    }
                )
    predictors = fdr_by_family(pd.DataFrame(predictor_rows))
    predictors.to_csv(args.output / "rest_to_advanced_visual_models.csv", index=False)

    summary = {
        "n_subject_frequency_cells": int(network.shape[0]),
        "n_subjects": int(network.subject.nunique()),
        "minimum_primary_severity_q": float(
            severity.loc[severity.family == "severity_primary", "q_bh"].min()
        ),
        "minimum_primary_group_q": float(
            groups.loc[groups.family == "group_primary", "q_bh"].min()
        ),
        "minimum_rest_predictor_q": float(predictors.q_bh.min()),
        "maximum_average_native_imcoh_rho": float(
            reference.loc[
                (reference.metric == "posterior_frontal_abs_imcoh")
                & (reference.comparison == "average_vs_native"),
                "spearman_rho",
            ].max()
        ),
        "analysis_boundary": "Foundation models; short windows and reference sensitivity preclude cortical propagation claims.",
    }
    (args.output / "model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
