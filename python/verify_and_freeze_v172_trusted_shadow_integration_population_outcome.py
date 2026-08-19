#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v172_trusted_shadow_integration_population import DEPENDENCY_KEYS, reconstruct


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v172-trusted-shadow-integration-population-lock.json"
    result_path = PROJECT_ROOT / "outputs/v172-trusted-shadow-integration-population/population/result.json"
    doc_path = PROJECT_ROOT / "docs/v172-trusted-shadow-integration-population-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v172_trusted_shadow_integration_population_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v172-trusted-shadow-integration-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v172-trusted-shadow-integration-population-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V172 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V172 results before freezing")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    artifacts = reconstruct(lock)
    population = artifacts["population"]
    expected = {
        "constraint_states": {"states": population["states"], "contains_policy_or_sandbox_scores": False},
        "eligible_state_ids": {"state_ids": population["integration_eligible_state_ids"], "selection_uses_scores": False},
        "target_cases": {"target_cases": population["target_cases"], "target_subsampling": False},
        "population_summary": population["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, value in expected.items()
    )
    config = lock["config_payload"]
    expected_decision = (
        config["decisionRule"]["ifEveryPopulationGatePasses"]
        if artifacts["audit"]["passed"]
        else config["decisionRule"]["otherwise"]
    )
    checks = {
        "population_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "population_outputs_reconstruct_exactly": outputs_exact,
        "gates_summary_and_decision_reconstruct_exactly": bool(
            result["summary"] == population["summary"]
            and result["gates"] == artifacts["audit"]["checks"]
            and result["passed"] == artifacts["audit"]["passed"]
            and result["decision"] == expected_decision
        ),
        "complete_targets_and_eligibility_hold_without_scores_or_transactions": bool(
            population["summary"]["planner_policy_score_count"] == 0
            and population["summary"]["sandbox_transaction_count"] == 0
            and population["summary"]["integration_eligible_state_count"] == 132
            and population["summary"]["target_case_count"] == 4224
            and all(artifacts["audit"]["checks"].values())
        ),
        "model_authority_effect_and_execution_boundary_holds": all(
            result["access"][key] == 0
            for key in (
                "planner_policy_score_count",
                "sandbox_transaction_count",
                "evaluation_record_count",
                "manual_judgment_count",
                "model_load_count",
                "model_generation_count",
                "API_call_count",
                "training_run_count",
                "ontology_registration_count",
                "trusted_state_mutation_count",
                "real_service_call_count",
                "external_side_effect_count",
                "actual_execution_count",
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "172-trusted-shadow-integration-population-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_population_gates_passed": result["passed"],
        "decision": "freeze_positive_V172_population_outcome" if passed and result["passed"] else "reject_V172_outcome",
        "checks": checks,
        "independent_summary": population["summary"],
        "additional_access": {"planner_policy_score_count": 0, "sandbox_transaction_count": 0, "model_load_count": 0, "actual_execution_count": 0},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
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
        "schema_version": "172-trusted-shadow-integration-population-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_population_gates_passed": result["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_or_rebuild_V172": False,
            "preregister_trusted_only_shadow_integration_on_all_eligible_states_and_targets": bool(result["passed"]),
            "score_or_run_integration_without_separate_lock": False,
            "select_subsample_or_tune_using_V172_outcomes": False,
            "run_model_register_mutate_real_state_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
