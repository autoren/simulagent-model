#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v176_four_constraint_confirmation_population import (
    audit_population,
    build_population,
)


DEPENDENCY_KEYS = (
    "config",
    "parent_V175_outcome",
    "source_V172_outcome",
    "source_V167_planner_lock",
    "V172_constraint_states",
    "V172_target_cases",
    "roadmap",
    "plan",
    "protocol",
    "tests",
    "runner",
    "verifier",
    "auditor",
    "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    V172_states = json.loads(
        (PROJECT_ROOT / lock["V172_constraint_states"]).read_text()
    )
    V172_targets = json.loads(
        (PROJECT_ROOT / lock["V172_target_cases"]).read_text()
    )
    population = build_population(V172_states, V172_targets)
    audit = audit_population(
        population, V172_states, V172_targets, lock["config_payload"]
    )
    return population, audit


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v176-four-constraint-confirmation-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v176-four-constraint-confirmation-population/population"
    if output_root.exists():
        raise RuntimeError("V176 formal population may be built only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V176 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V176 dependency drifted: {key}")

    population, audit = reconstruct(lock)
    config = lock["config_payload"]
    passed = audit["passed"]
    decision = (
        config["decisionRule"]["ifEveryPopulationGatePasses"]
        if passed
        else config["decisionRule"]["otherwise"]
    )
    access = {
        "formal_population_build_count": 1,
        "planner_policy_score_count": 0,
        "sandbox_transaction_count": 0,
        "evaluation_record_count": 0,
        "manual_judgment_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_real_state_mutation_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    states_path = output_root / "constraint-states.json"
    eligible_path = output_root / "confirmation-eligible-state-ids.json"
    targets_path = output_root / "target-cases.json"
    summary_path = output_root / "population-summary.json"
    write_json(
        states_path,
        {"states": population["states"], "all_source_states_retained": True},
    )
    write_json(
        eligible_path,
        {
            "state_ids": population["confirmation_eligible_state_ids"],
            "selection_uses_only_frozen_class_metadata": True,
        },
    )
    write_json(
        targets_path,
        {"target_cases": population["target_cases"], "target_subsampling": False},
    )
    write_json(summary_path, population["summary"])
    output_integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in {
            "constraint_states": states_path,
            "eligible_state_ids": eligible_path,
            "target_cases": targets_path,
            "population_summary": summary_path,
        }.items()
    }
    result = {
        "schema_version": "176-four-constraint-confirmation-population-result",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": decision,
        "summary": population["summary"],
        "population_gates": audit["checks"],
        "access": access,
        "output_integrity": output_integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
