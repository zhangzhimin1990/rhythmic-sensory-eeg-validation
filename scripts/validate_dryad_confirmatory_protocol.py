#!/usr/bin/env python3
"""Validate the locked Dryad confirmatory protocol against the technical gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/dryad_confirmatory_protocol_v1.json"
DEFAULT_GATE = (
    ROOT
    / "outputs/qc/dryad_raw_stream/R2b_validation_gate_v1/r2b_validation_technical_gate.csv"
)
DEVELOPMENT = {2, 13, 23, 27, 37}


def validate(protocol_path: Path, gate_path: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    gate = pd.read_csv(gate_path)
    if len(gate) != 39 or gate["subject"].nunique() != 39:
        raise ValueError("technical gate must contain the exact 39-person validation set")
    observed_pass = set(
        gate.loc[
            gate["technical_pass_development_candidate"].astype(bool), "subject"
        ].astype(int)
    )
    observed_fail = set(gate["subject"].astype(int)) - observed_pass
    declared = set(map(int, protocol["confirmatory_subjects"]))
    declared_fail = set(map(int, protocol["technical_failures_excluded"]))
    declared_development = set(map(int, protocol["development_subjects_excluded"]))
    if declared != observed_pass:
        raise ValueError("confirmatory subjects do not equal frozen technical-pass subjects")
    if declared_fail != observed_fail:
        raise ValueError("technical-failure exclusion list does not match gate")
    if declared_development != DEVELOPMENT:
        raise ValueError("development exclusion list changed")
    if declared & declared_fail or declared & declared_development:
        raise ValueError("confirmatory and excluded populations overlap")
    if protocol["n_confirmatory_subjects"] != len(declared):
        raise ValueError("declared confirmatory sample size is inconsistent")
    contrast = protocol["active_contrast"]
    if contrast["experimental_condition"] != "f_theta_plus":
        raise ValueError("experimental condition drifted")
    if contrast["control_condition"] != "non_rhythmic":
        raise ValueError("active control drifted")
    primary = protocol["primary_neural_estimand"]
    if primary["metric"] != "mean_itpc_in_stimulation_window":
        raise ValueError("primary neural metric drifted")
    if primary["population"] != "35 frozen technical-pass validation participants":
        raise ValueError("primary population is not frozen")
    if primary["standardized_smallest_effect_of_interest_dz"] != 0.3:
        raise ValueError("primary standardized equivalence bound drifted")
    if "full 7-second epoch" not in protocol["preprocessing"]["narrowband_phase_filter"]:
        raise ValueError("narrowband filter order drifted")
    if protocol["behavioural_estimands"]["causal_claim"]:
        raise ValueError("causal claim must remain disabled")
    if protocol["behavioural_estimands"]["mediation_claim"]:
        raise ValueError("mediation claim must remain disabled")
    return {
        "protocol_id": protocol["protocol_id"],
        "n_confirmatory_subjects": len(declared),
        "technical_failures_excluded": sorted(declared_fail),
        "development_subjects_excluded": sorted(declared_development),
        "primary_metric": primary["metric"],
        "active_contrast": "f_theta_plus_minus_non_rhythmic",
        "protocol_valid": True,
        "outcomes_accessed_by_validator": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--technical-gate", type=Path, default=DEFAULT_GATE)
    args = parser.parse_args()
    print(json.dumps(validate(args.protocol, args.technical_gate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
