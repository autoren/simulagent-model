#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v202_model_free_controller_decision_sufficiency import (
    audit_evaluation,
    evaluate_controllers,
)
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v202-model-free-controller-decision-sufficiency-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V202 design lock")
    dependency_keys = [
        key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock
    ]
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V202 locked dependency changed: {key}")
    output_root = PROJECT_ROOT / "outputs/v202-model-free-controller-decision-sufficiency/evaluation"
    if output_root.exists():
        raise RuntimeError("V202 output already exists")
    input_keys = (
        "canonical_model_census",
        "transformed_model_census",
        "canonical_CHAR_LAST_predictions",
        "transformed_CHAR_LAST_scored_records",
        "hidden_targets",
        "canonical_hidden_option_map",
        "hidden_variant_maps",
        "primary_prior",
        "fixed_hierarchy_target_costs",
    )
    evaluation = evaluate_controllers(
        *[json.loads((PROJECT_ROOT / lock[key]).read_text()) for key in input_keys],
        lock["config_payload"],
    )
    audit = audit_evaluation(evaluation, lock["config_payload"])
    config = lock["config_payload"]
    selected = evaluation["summary"]["selected_policy_id"] is not None
    decision = config["decisionRule"][
        "ifAtLeastOnePolicyQualifies" if selected else "otherwise"
    ]
    result = {
        "schema_version": "202-model-free-controller-decision-sufficiency-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "scientific_selection_made": selected,
        "decision": decision,
        "claim_boundary": config["claimBoundary"],
        "checks": audit["checks"],
        "summary": audit["summary"],
        "authorization": {
            "preregister_new_fresh_confirmation_design_only": bool(audit["passed"] and selected),
            "immediate_confirmation_or_protected_reuse": False,
            "API_model_generation_training_registration_authority_action_or_execution": False,
        },
    }
    write_json(output_root / "scored-records.json", evaluation["scored_records"])
    write_json(output_root / "summary.json", evaluation["summary"])
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
