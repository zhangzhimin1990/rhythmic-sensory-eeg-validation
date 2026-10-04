#!/usr/bin/env python3
"""Clinical and sensitivity models for ds005048 auditory advanced features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


PRIMARY = (
    "central_minus_frontal_snr_db",
    "csd_central_frontal_abs_imcoh",
)
EXPLORATORY = (
    "central_frontal_dwpli2",
    "csd_central_frontal_dwpli2",
    "block_adaptation_slope_db_per_block",
    "within_block_slope_db_per_quartile",
    "q4_minus_q1_snr_db",
)


def zscore(frame: pd.DataFrame, column: str, target: str) -> None:
    frame[target] = (frame[column] - frame[column].mean()) / frame[column].std()


def fit_mmse(data: pd.DataFrame, outcome: str, control_local_snr: bool) -> dict:
    frame = data.dropna(subset=["mmse", outcome]).copy()
    zscore(frame, "mmse", "z_mmse")
    zscore(frame, "age", "z_age")
    zscore(frame, "central_snr_db", "z_local_snr")
    formula = f"{outcome} ~ z_mmse + z_age + C(sex)"
    if control_local_snr:
        formula += " + z_local_snr"
    model = smf.ols(formula, frame).fit(cov_type="HC3")
    return {
        "n": int(model.nobs),
        "coefficient_per_mmse_sd": float(model.params["z_mmse"]),
        "standard_error": float(model.bse["z_mmse"]),
        "p_value": float(model.pvalues["z_mmse"]),
        "r_squared": float(model.rsquared),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_features", type=Path)
    parser.add_argument("block_features", type=Path)
    parser.add_argument("reliability", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_csv(args.subject_features)
    blocks = pd.read_csv(args.block_features)
    reliability = pd.read_csv(args.reliability)

    reference_rows = []
    for metric in ("central_frontal_abs_imcoh", "central_frontal_dwpli2"):
        variants = {
            "average": metric,
            "native": f"native_{metric}",
            "csd": f"csd_{metric}",
        }
        for first, second in (
            ("average", "native"),
            ("average", "csd"),
            ("native", "csd"),
        ):
            result = spearmanr(
                subjects[variants[first]], subjects[variants[second]], nan_policy="omit"
            )
            reference_rows.append(
                {
                    "metric": metric,
                    "comparison": f"{first}_vs_{second}",
                    "n": int(
                        subjects[[variants[first], variants[second]]].dropna().shape[0]
                    ),
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )
    pd.DataFrame(reference_rows).to_csv(
        args.output / "reference_sensitivity.csv", index=False
    )

    model_rows = []
    for family, outcomes in (("primary", PRIMARY), ("exploratory", EXPLORATORY)):
        for outcome in outcomes:
            for control_local_snr in (True, False):
                row = fit_mmse(subjects, outcome, control_local_snr)
                row.update(
                    {
                        "family": family,
                        "outcome": outcome,
                        "model": (
                            "incremental_over_local_snr"
                            if control_local_snr
                            else "age_sex_adjusted"
                        ),
                    }
                )
                model_rows.append(row)
    models = pd.DataFrame(model_rows)
    models["q_bh"] = np.nan
    for _, indices in models.groupby(["family", "model"]).groups.items():
        models.loc[indices, "q_bh"] = multipletests(
            models.loc[indices, "p_value"], method="fdr_bh"
        )[1]
    models.to_csv(args.output / "mmse_models.csv", index=False)

    # Window-amplitude sensitivity: recompute subject features after excluding
    # the 15/210 blocks whose central-ROI peak-to-peak amplitude exceeds 100 µV.
    clean_blocks = blocks[blocks.central_peak_to_peak_uv <= 100.0]
    clean_subjects = (
        clean_blocks.groupby("participant_id", as_index=False)
        .agg(
            sex=("sex", "first"),
            age=("age", "first"),
            mmse=("mmse", "first"),
            central_snr_db=("central_snr_db", "mean"),
            central_minus_frontal_snr_db=(
                "central_minus_frontal_snr_db",
                "mean",
            ),
            csd_central_frontal_abs_imcoh=(
                "csd_central_frontal_abs_imcoh",
                "mean",
            ),
            n_clean_blocks=("block_index", "nunique"),
        )
    )
    amplitude_rows = []
    clean_subjects_primary = clean_subjects[clean_subjects.n_clean_blocks >= 5].copy()
    for outcome in PRIMARY:
        row = fit_mmse(clean_subjects_primary, outcome, True)
        row.update(
            {
                "outcome": outcome,
                "subjects_with_at_least_five_clean_blocks": int(
                    (clean_subjects.n_clean_blocks >= 5).sum()
                ),
                "minimum_clean_blocks_in_model": int(
                    clean_subjects_primary.n_clean_blocks.min()
                ),
            }
        )
        amplitude_rows.append(row)
    pd.DataFrame(amplitude_rows).to_csv(
        args.output / "amplitude_sensitivity.csv", index=False
    )

    # Leave-one-out coefficient stability for the two primary incremental models.
    loo_rows = []
    analysis_subjects = subjects.dropna(subset=["mmse"]).participant_id.tolist()
    for outcome in PRIMARY:
        full = fit_mmse(subjects, outcome, True)
        coefficients = []
        for omitted in analysis_subjects:
            result = fit_mmse(
                subjects[subjects.participant_id != omitted], outcome, True
            )
            coefficients.append(result["coefficient_per_mmse_sd"])
        loo_rows.append(
            {
                "outcome": outcome,
                "full_coefficient": full["coefficient_per_mmse_sd"],
                "leave_one_out_min": float(np.min(coefficients)),
                "leave_one_out_max": float(np.max(coefficients)),
                "sign_stable": bool(
                    np.all(np.sign(coefficients) == np.sign(full["coefficient_per_mmse_sd"]))
                ),
            }
        )
    pd.DataFrame(loo_rows).to_csv(
        args.output / "leave_one_out_stability.csv", index=False
    )

    incremental_primary = models[
        (models.family == "primary")
        & (models.model == "incremental_over_local_snr")
    ]
    summary = {
        "n_subjects": int(subjects.participant_id.nunique()),
        "n_with_mmse": int(subjects.mmse.notna().sum()),
        "n_blocks": int(blocks.shape[0]),
        "n_blocks_at_or_below_100uv": int(
            (blocks.central_peak_to_peak_uv <= 100.0).sum()
        ),
        "minimum_primary_incremental_q": float(incremental_primary.q_bh.min()),
        "minimum_exploratory_incremental_q": float(
            models.loc[
                (models.family == "exploratory")
                & (models.model == "incremental_over_local_snr"),
                "q_bh",
            ].min()
        ),
        "minimum_split_half_rho": float(reliability.split_half_spearman_rho.min()),
        "analysis_boundary": "Advanced auditory features are measurable, but clinical increment requires FDR and sensitivity support.",
    }
    (args.output / "model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(models.to_string(index=False))


if __name__ == "__main__":
    main()
