#!/usr/bin/env python3
"""Assemble frozen endpoint decisions into one conservative manuscript claim ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.classify_dryad_confirmatory_behaviour_branch import run as run_behaviour
from scripts.classify_dryad_confirmatory_coupling_branch import run as run_coupling
from scripts.classify_dryad_confirmatory_prediction_branch import run as run_prediction
from scripts.classify_dryad_confirmatory_primary_branch import run as run_primary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUP_ROOT = ROOT / "outputs/models/dryad_confirmatory_group_v1"
DEFAULT_AUDIT_ROOT = ROOT / "outputs/models/dryad_confirmatory_inference_audit_v1"
DEFAULT_OUTPUT = ROOT / "outputs/status/dryad_confirmatory_claim_release_v1"


def decide(primary: dict, behaviour: dict, coupling: dict, prediction: dict) -> dict[str, object]:
    target = bool(primary["theta_plus_superiority_claim_unlocked"])
    task = bool(behaviour["acute_task_performance_advantage_claim_unlocked"])
    specific = bool(coupling["theta_plus_specific_coupling_supported"])
    within = bool(coupling["theta_plus_within_condition_coupling_supported"])
    predictive = bool(prediction["prediction_increment_claim_unlocked"])

    if target and task and specific:
        ceiling = "target_engagement_behaviour_and_condition_specific_concurrent_coupling"
        sentence = (
            "The active-control contrast supported stronger stimulus-locked target engagement, "
            "an acute task-performance advantage, and condition-specific concurrent neural-behaviour coupling."
        )
    elif target and task:
        ceiling = "target_engagement_and_behaviour_without_specific_coupling"
        sentence = (
            "The active-control contrast supported target engagement and an acute task-performance "
            "advantage, but did not establish condition-specific neural-behaviour coupling."
        )
    elif target:
        ceiling = "target_engagement_without_behaviour_chain"
        sentence = (
            "The active-control contrast supported stronger stimulus-locked target engagement, "
            "but the prespecified behavioural-mechanistic chain was not established."
        )
    elif task:
        ceiling = "behaviour_without_primary_target_engagement"
        sentence = (
            "An acute task-performance advantage was observed without support for the prespecified "
            "primary target-engagement contrast."
        )
    else:
        ceiling = "measurement_and_translation_boundary"
        sentence = (
            "The confirmatory analysis did not support a continuous chain from the prespecified "
            "primary target-engagement contrast to acute behavioural advantage."
        )

    modifiers = []
    if within and not specific:
        modifiers.append("within-theta-plus concurrent coupling was present without condition-specific increment")
    if predictive:
        modifiers.append("exploratory internal out-of-sample prediction increment passed both comparator gates")
    else:
        modifiers.append("no consistent internal out-of-sample prediction increment was established")
    return {
        "status": "confirmatory_claim_release_decided",
        "claim_ceiling": ceiling,
        "abstract_core_sentence": sentence,
        "supporting_modifiers": modifiers,
        "endpoint_branches": {
            "primary": primary["branch"],
            "behaviour": behaviour["branch"],
            "coupling": coupling["branch"],
            "prediction": prediction["branch"],
        },
        "causal_mechanism_claim_unlocked": False,
        "chronic_cognitive_efficacy_claim_unlocked": False,
        "clinical_treatment_selection_claim_unlocked": False,
        "patient_generalisation_claim_unlocked": False,
        "prediction_cannot_raise_causal_claim_ceiling": True,
    }


def run(group_root: Path, audit_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    staging = output_dir.parent / f".{output_dir.name}_staging"
    if staging.exists() and any(staging.iterdir()):
        raise FileExistsError(f"refusing non-empty staging directory: {staging}")
    primary = run_primary(group_root, audit_root, staging / "primary")
    behaviour = run_behaviour(group_root, staging / "behaviour")
    coupling = run_coupling(group_root, audit_root, staging / "coupling")
    prediction = run_prediction(audit_root, staging / "prediction")
    result = decide(primary, behaviour, coupling, prediction)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "claim_release_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-root", type=Path, default=DEFAULT_GROUP_ROOT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.group_root, args.audit_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
