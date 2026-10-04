#!/usr/bin/env python3
"""Audit ds006780 d-prime and neural association across response windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from analyze_ds006780_behavioral_validity import corrected_dprime
from derive_ds006780_assr_features import behavior_by_frequency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/public/ds006780/v1.0.0")
    )
    parser.add_argument(
        "--subjects",
        type=Path,
        default=Path("outputs/models/ds006780_behavioral_validity/subject_level_features.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/models/ds006780_behavioral_validity/response_window_audit"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for path in sorted(args.metadata.glob("sub-*/eeg/*task-ASSR_run-*_events.tsv")):
        subject = path.name.split("_", 1)[0]
        run = path.name.split("run-")[-1].split("_")[0]
        events = pd.read_csv(path, sep="\t")
        for window in (0.75, 1.0, 1.25):
            for row in behavior_by_frequency(events, response_window=window):
                run_rows.append({"subject": subject, "run": run, "window": window, **row})
    runs = pd.DataFrame(run_rows)
    runs.to_csv(args.output / "run_behavior_by_window.csv", index=False)

    subject_rows = []
    for (subject, window, frequency), group in runs.groupby(
        ["subject", "window", "frequency_hz"]
    ):
        counts = {
            column: int(group[column].sum())
            for column in ("hits", "misses", "false_alarms", "correct_rejections")
        }
        subject_rows.append(
            {
                "subject": subject,
                "window": window,
                "frequency_hz": frequency,
                **counts,
                "dprime": corrected_dprime(**counts),
                "unassigned_responses_total": int(group.unassigned_responses_total.sum()),
            }
        )
    behavior = pd.DataFrame(subject_rows)
    behavior.to_csv(args.output / "subject_behavior_by_window.csv", index=False)

    subjects = pd.read_csv(args.subjects)
    analysis = subjects.loc[subjects.technical_usable].copy()
    model_rows = []
    merged_by_window = {}
    for window in (0.75, 1.0, 1.25):
        selected = behavior.loc[
            behavior.window.eq(window) & behavior.frequency_hz.eq(40),
            ["subject", "dprime"],
        ].rename(columns={"dprime": "dprime_window"})
        merged = analysis.merge(selected, on="subject", how="inner")
        for column in ("local_log_snr_db_mean_40", "local_log_snr_db_mean_27", "age"):
            merged[f"z_{column}"] = (
                merged[column] - merged[column].mean()
            ) / merged[column].std(ddof=0)
        fitted = smf.ols(
            "dprime_window ~ z_local_log_snr_db_mean_40"
            " + z_local_log_snr_db_mean_27 + z_age + C(sex) + C(group)",
            data=merged,
        ).fit(cov_type="HC3")
        term = "z_local_log_snr_db_mean_40"
        confidence = fitted.conf_int().loc[term]
        model_rows.append(
            {
                "response_window_seconds": window,
                "estimate_40_per_sd": float(fitted.params[term]),
                "ci_low": float(confidence.iloc[0]),
                "ci_high": float(confidence.iloc[1]),
                "p_value": float(fitted.pvalues[term]),
                "n_subjects": int(fitted.nobs),
            }
        )
        merged_by_window[window] = merged.set_index("subject").dprime_window
    pd.DataFrame(model_rows).to_csv(args.output / "window_models.csv", index=False)

    correlations = {}
    for window in (0.75, 1.25):
        paired = pd.concat([merged_by_window[1.0], merged_by_window[window]], axis=1).dropna()
        rho, pvalue = spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
        correlations[str(window)] = {
            "spearman_with_1.0s": float(rho), "p_value": float(pvalue), "n": len(paired)
        }
    summary = {"runs_with_events": int(runs[["subject", "run"]].drop_duplicates().shape[0]),
               "subjects": int(runs.subject.nunique()), "dprime_correlations": correlations}
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
