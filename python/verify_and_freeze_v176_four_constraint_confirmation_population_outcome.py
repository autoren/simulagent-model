#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v176_four_constraint_confirmation_population import (
    DEPENDENCY_KEYS,
    reconstruct,
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v176-four-constraint-confirmation-population-lock.json"
    result_path = PROJECT_ROOT / "outputs/v176-four-constraint-confirmation-population/population/result.json"
    doc_path = PROJECT_ROOT / "docs/v176-four-constraint-confirmation-population-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v176_four_constraint_confirmation_population_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v176-four-constraint-confirmation-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v176-four-constraint-confirmation-population-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V176 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V176 population results before freezing")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    population, audit = reconstruct(lock)
    expected_outputs = {
        "constraint_states": {
            "states": population["states"],
            "all_source_states_retained": True,
        },
        "eligible_state_ids": {
            "state_ids": population["confirmation_eligible_state_ids"],
            "selection_uses_only_frozen_class_metadata": True,
        },
        "target_cases": {
            "target_cases": population["target_cases"],
            "target_subsampling": False,
        },
        "population_summary": population["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text())
        == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, value in expected_outputs.items()
    )
    config = lock["config_payload"]
    expected_decision = (
        config["decisionRule"]["ifEveryPopulationGatePasses"]
        if audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    checks = {
        "lock_and_every_dependency_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
                for key in DEPENDENCY_KEYS
            )
        ),
        "population_outputs_reconstruct_exactly": outputs_exact,
        "gates_summary_and_decision_reconstruct_exactly": bool(
            result["summary"] == population["summary"]
            and result["population_gates"] == audit["checks"]
            and result["passed"] == audit["passed"]
            and result["decision"] == expected_decision
        ),
        "complete_context_disjoint_population_is_frozen_before_scoring": bool(
            population["summary"]["source_state_count"] == 1120
            and population["summary"]["confirmation_eligible_state_count"] == 135
            and population["summary"]["target_case_count"] == 2160
            and population["summary"]["exact_target_context_signature_overlap_with_V172"] == 0
            and result["access"]["planner_policy_score_count"] == 0
            and result["access"]["sandbox_transaction_count"] == 0
        ),
        "candidate_reuse_is_explicit_and_model_authority_effect_boundary_holds": bool(
            population["summary"]["candidate_identity_overlap_with_V172_count"]
            == population["summary"]["unique_target_candidate_count"]
            and all(
                result["access"][key] == 0
                for key in (
                    "evaluation_record_count",
                    "manual_judgment_count",
                    "model_load_count",
                    "model_generation_count",
                    "API_call_count",
                    "training_run_count",
                    "ontology_registration_count",
                    "trusted_real_state_mutation_count",
                    "real_service_call_count",
                    "external_side_effect_count",
                    "actual_execution_count",
                )
            )
        ),
    }
    passed = all(checks.values())
    outcome_audit = {
        "schema_version": "176-four-constraint-confirmation-population-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_population_gates_passed": audit["passed"],
        "decision": "freeze_verified_V176_population" if passed else "reject_V176_outcome",
        "checks": checks,
        "independent_summary": population["summary"],
        "additional_access": {
            "planner_policy_score_count": 0,
            "sandbox_transaction_count": 0,
            "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, outcome_audit)
    if not passed:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "population_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "176-four-constraint-confirmation-population-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_population_gates_passed": audit["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_or_rebuild_V176": False,
            "preregister_one_unchanged_V175_confirmation": bool(audit["passed"]),
            "score_or_run_confirmation_without_separate_lock": False,
            "select_subsample_or_tune_using_V176_outcomes": False,
            "run_model_register_mutate_real_state_call_service_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
