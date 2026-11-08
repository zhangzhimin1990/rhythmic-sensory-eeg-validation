#!/usr/bin/env python3
"""Run the frozen HBM participant-level prediction robustness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.analyze_ds007648_behavioral_validity import add_features, prepare_trials


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/hbm_prediction_audit_amendment_v1.json"
DEFAULT_DRYAD = ROOT / "outputs/models/dryad_confirmatory_group_v1/nested_prediction_rows.csv"
DEFAULT_DS7 = ROOT / "outputs/derived/ds007648_trial_features/all_trial_features.csv"
DEFAULT_DS6 = ROOT / "outputs/models/ds006780_behavioral_validity/subject_level_features.csv"
DEFAULT_OUTPUT = ROOT / "outputs/models/hbm_prediction_audit_v1"


def ds007648_predictions(path: Path) -> pd.DataFrame:
    frame = add_features(prepare_trials(path), "phase_projected").copy()
    frame["hz40_auditory_interaction"] = frame.hz40_within_z * frame.target_auditory
    frame["hz36_auditory_interaction"] = frame.hz36_within_z * frame.target_auditory
    categorical = ["trial_type"]
    base_numeric = ["trial_order_z"]
    neural = [
        "hz40_within_z",
        "hz36_within_z",
        "hz40_auditory_interaction",
        "hz36_auditory_interaction",
    ]
    rows: list[pd.DataFrame] = []
    for held_out in sorted(frame.subject.unique()):
        train_all = frame.loc[frame.subject.ne(held_out)].copy()
        test_all = frame.loc[frame.subject.eq(held_out)].copy()
        for outcome in ("accuracy", "rt"):
            train, test = train_all, test_all
            observed_column = "Correct"
            if outcome == "rt":
                train = train.loc[train.Correct.eq(1) & train.response_time.gt(0)].copy()
                test = test.loc[test.Correct.eq(1) & test.response_time.gt(0)].copy()
                observed_column = "log_rt"
            for model_name, numeric in (("base", base_numeric), ("neural", base_numeric + neural)):
                transform = ColumnTransformer(
                    [
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("numeric", make_pipeline(SimpleImputer(), StandardScaler()), numeric),
                    ]
                )
                if outcome == "accuracy":
                    estimator = make_pipeline(transform, LogisticRegression(max_iter=2000, C=1.0))
                    estimator.fit(train[categorical + numeric], train.Correct)
                    predicted = estimator.predict_proba(test[categorical + numeric])[:, 1]
                else:
                    estimator = make_pipeline(transform, Ridge(alpha=1.0))
                    estimator.fit(train[categorical + numeric], train.log_rt)
                    predicted = estimator.predict(test[categorical + numeric])
                rows.append(pd.DataFrame({
                    "dataset": "ds007648",
                    "outcome": outcome,
                    "subject": held_out,
                    "model": model_name,
                    "observed": test[observed_column].to_numpy(float),
                    "predicted": np.asarray(predicted, dtype=float),
                }))
    return pd.concat(rows, ignore_index=True)


def ds006780_predictions(path: Path, seed: int = 20260921) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame.technical_usable.astype(bool)].copy()
    rows: list[pd.DataFrame] = []
    for outcome in ("dprime_40", "fsiq"):
        used = frame.dropna(
            subset=[outcome, "age", "sex", "group", "local_log_snr_db_mean_27", "local_log_snr_db_mean_40"]
        ).reset_index(drop=True)
        categorical = ["sex", "group"]
        base_numeric = ["age"]
        neural_numeric = base_numeric + ["local_log_snr_db_mean_27", "local_log_snr_db_mean_40"]
        folds = RepeatedKFold(n_splits=10, n_repeats=10, random_state=seed)
        for fold_index, (train_index, test_index) in enumerate(folds.split(used)):
            train, test = used.iloc[train_index], used.iloc[test_index]
            for model_name, numeric in (("base", base_numeric), ("neural", neural_numeric)):
                transform = ColumnTransformer(
                    [
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                        ("numeric", make_pipeline(SimpleImputer(), StandardScaler()), numeric),
                    ]
                )
                estimator = make_pipeline(transform, Ridge(alpha=1.0))
                estimator.fit(train[categorical + numeric], train[outcome])
                prediction = estimator.predict(test[categorical + numeric])
                rows.append(pd.DataFrame({
                    "dataset": "ds006780",
                    "outcome": outcome,
                    "subject": test.subject.astype(str).to_numpy(),
                    "model": model_name,
                    "observed": test[outcome].to_numpy(float),
                    "predicted": prediction,
                    "fold": fold_index,
                    "repeat": fold_index // 10,
                }))
    trial = pd.concat(rows, ignore_index=True)
    counts = trial.groupby(["outcome", "subject", "model"]).size()
    if not counts.eq(10).all():
        raise RuntimeError("each ds006780 participant must have ten held-out predictions per model")
    return trial.groupby(["dataset", "outcome", "subject", "model"], as_index=False).agg(
        observed=("observed", "first"), predicted=("predicted", "mean")
    )


def dryad_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["dataset"] = "Dryad"
    frame["outcome"] = "inverse_efficiency_difference_ms"
    frame["subject"] = frame.subject.astype(str)
    frame["model"] = frame.model.replace(
        {"baseline": "base", "baseline_plus_neural": "neural"}
    )
    return frame[["dataset", "outcome", "subject", "model", "observed", "predicted"]]


def participant_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, outcome, subject, model), group in predictions.groupby(
        ["dataset", "outcome", "subject", "model"], sort=True
    ):
        observed = group.observed.to_numpy(float)
        predicted = group.predicted.to_numpy(float)
        if outcome == "accuracy":
            loss = float(log_loss(observed, predicted, labels=[0, 1]))
            metric = "log_loss"
        else:
            loss = float(np.mean(np.abs(observed - predicted)))
            metric = "absolute_error"
        rows.append({
            "dataset": dataset,
            "outcome": outcome,
            "subject": subject,
            "model": model,
            "metric": metric,
            "loss": loss,
        })
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sign_flip_test(values: np.ndarray, permutations: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("sign-flip values must be a finite one-dimensional vector")
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    exceed = 0
    null_parts = []
    remaining = permutations
    while remaining:
        size = min(10000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        null = (signs * values).mean(axis=1)
        exceed += int(np.sum(np.abs(null) >= abs(observed)))
        null_parts.append(null)
        remaining -= size
    distribution = np.concatenate(null_parts)
    return {
        "observed_mean_delta": observed,
        "null_q025": float(np.quantile(distribution, 0.025)),
        "null_q500": float(np.quantile(distribution, 0.5)),
        "null_q975": float(np.quantile(distribution, 0.975)),
        "p_value_two_sided": float((1 + exceed) / (permutations + 1)),
    }


def loss_audit(losses: pd.DataFrame, permutations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = losses.pivot(
        index=["dataset", "outcome", "subject", "metric"], columns="model", values="loss"
    ).reset_index()
    wide["delta_neural_minus_base"] = wide.neural - wide.base
    summaries = []
    for index, ((dataset, outcome, metric), group) in enumerate(
        wide.groupby(["dataset", "outcome", "metric"], sort=True)
    ):
        values = group.delta_neural_minus_base.to_numpy(float)
        leave_one_out = np.asarray([
            np.delete(values, item).mean() for item in range(len(values))
        ])
        test = sign_flip_test(values, permutations, seed + index)
        summaries.append({
            "dataset": dataset,
            "outcome": outcome,
            "metric": metric,
            "n_subjects": len(values),
            "mean_base": float(group.base.mean()),
            "mean_neural": float(group.neural.mean()),
            **test,
            "median_delta": float(np.median(values)),
            "iqr_delta": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "fraction_subjects_improved": float(np.mean(values < 0)),
            "loo_mean_delta_min": float(leave_one_out.min()),
            "loo_mean_delta_max": float(leave_one_out.max()),
        })
    return wide, pd.DataFrame(summaries)


def continuous_calibration(group: pd.DataFrame) -> dict[str, float]:
    data = group.groupby("subject", as_index=False).agg(
        observed=("observed", "mean"), predicted=("predicted", "mean")
    )
    fitted = sm.OLS(data.observed, sm.add_constant(data.predicted)).fit(cov_type="HC3")
    confidence = fitted.conf_int()
    return {
        "n_subjects": len(data),
        "intercept": float(fitted.params.iloc[0]),
        "intercept_ci95_low": float(confidence.iloc[0, 0]),
        "intercept_ci95_high": float(confidence.iloc[0, 1]),
        "slope": float(fitted.params.iloc[1]),
        "slope_ci95_low": float(confidence.iloc[1, 0]),
        "slope_ci95_high": float(confidence.iloc[1, 1]),
        "r_squared": float(fitted.rsquared),
    }


def binary_gee_calibration(group: pd.DataFrame) -> dict[str, float]:
    data = group.copy()
    probability = np.clip(data.predicted.to_numpy(float), 1e-6, 1 - 1e-6)
    data["logit_prediction"] = np.log(probability / (1 - probability))
    fitted = sm.GEE.from_formula(
        "observed ~ logit_prediction",
        groups="subject",
        data=data,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    confidence = fitted.conf_int()
    return {
        "n_subjects": int(data.subject.nunique()),
        "n_observations": len(data),
        "intercept": float(fitted.params["Intercept"]),
        "intercept_ci95_low": float(confidence.loc["Intercept", 0]),
        "intercept_ci95_high": float(confidence.loc["Intercept", 1]),
        "slope": float(fitted.params["logit_prediction"]),
        "slope_ci95_low": float(confidence.loc["logit_prediction", 0]),
        "slope_ci95_high": float(confidence.loc["logit_prediction", 1]),
        "r_squared": np.nan,
    }


def calibration_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, outcome, model), group in predictions.groupby(["dataset", "outcome", "model"]):
        if dataset == "ds007648" and outcome == "accuracy":
            result = binary_gee_calibration(group)
            method = "clustered_logistic_gee"
        else:
            result = continuous_calibration(group)
            method = "participant_level_ols_hc3"
        rows.append({"dataset": dataset, "outcome": outcome, "model": model, "method": method, **result})
    return pd.DataFrame(rows)


def analyze(protocol_path: Path, dryad_path: Path, ds7_path: Path, ds6_path: Path, output_dir: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_audit_execution":
        raise RuntimeError("prediction-audit amendment is not frozen")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    dryad = dryad_predictions(dryad_path)
    ds7 = ds007648_predictions(ds7_path)
    ds6 = ds006780_predictions(ds6_path)
    predictions = pd.concat([dryad, ds7, ds6], ignore_index=True)
    losses = participant_losses(predictions)
    subject_increments, summary = loss_audit(
        losses,
        int(protocol["null_distribution"]["permutations"]),
        int(protocol["null_distribution"]["random_seed"]),
    )
    calibration = calibration_audit(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "held_out_prediction_rows.csv", index=False)
    losses.to_csv(output_dir / "participant_losses.csv", index=False)
    subject_increments.to_csv(output_dir / "participant_loss_increments.csv", index=False)
    summary.to_csv(output_dir / "prediction_audit_summary.csv", index=False)
    calibration.to_csv(output_dir / "calibration_summary.csv", index=False)
    result = {
        "protocol_id": protocol["protocol_id"],
        "classification": protocol["classification"],
        "datasets": sorted(predictions.dataset.unique().tolist()),
        "n_prediction_rows": len(predictions),
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "dryad_predictions": {"path": str(dryad_path), "sha256": sha256_file(dryad_path)},
            "ds007648_trial_features": {"path": str(ds7_path), "sha256": sha256_file(ds7_path)},
            "ds006780_subject_features": {"path": str(ds6_path), "sha256": sha256_file(ds6_path)},
        },
        "status": "hbm_prediction_audit_complete",
    }
    (output_dir / "prediction_audit_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dryad", type=Path, default=DEFAULT_DRYAD)
    parser.add_argument("--ds7", type=Path, default=DEFAULT_DS7)
    parser.add_argument("--ds6", type=Path, default=DEFAULT_DS6)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.protocol, args.dryad, args.ds7, args.ds6, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
