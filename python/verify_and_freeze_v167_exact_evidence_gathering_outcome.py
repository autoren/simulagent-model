#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v167_exact_evidence_gathering_planner import DEPENDENCY_KEYS, reconstruct
from v167_exact_evidence_gathering_planner import evaluate_gates


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    result_path = PROJECT_ROOT / "outputs/v167-exact-evidence-gathering/planner/result.json"
    doc_path = PROJECT_ROOT / "docs/v167-exact-evidence-gathering-planner-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v167_exact_evidence_gathering_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v167-exact-evidence-gathering/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V167 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V167 results before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    evaluation = reconstruct(lock)
    config = lock["config_payload"]
    gates = evaluate_gates(evaluation, result["access"], config)
    expected_outputs = {
        "case_policy_trees": {"cases": evaluation["cases"], "contains_language": False, "shadow_only": True},
        "planner_evaluation": {"summary": evaluation["summary"], "contains_language": False},
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == value
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"]) == result["output_integrity"][key]["sha256"]
        for key, value in expected_outputs.items()
    )
    expected_decision = config["decisionRule"]["ifEveryPlannerGatePasses"] if all(gates.values()) else config["decisionRule"]["otherwise"]
    summary = evaluation["summary"]
    checks = {
        "planner_lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "policy_trees_and_metrics_reconstruct_exactly": bool(outputs_exact and summary == result["summary"]),
        "gates_pass_and_decision_reconstructs": bool(gates == result["gates"] and result["passed"] == all(gates.values()) and result["decision"] == expected_decision),
        "information_value_and_history_dependence_are_realized": bool(
            summary["positive_value_of_information_case_count"] == 48
            and summary["strict_improvement_over_optimal_open_loop_case_count"] > 0
            and summary["history_dependent_second_action_case_count"] > 0
            and len(summary["unique_exact_bayes_root_queries"]) > 1
        ),
        "development_only_disclosure_is_preserved": bool(result["development_informed_not_confirmatory"] and config["developmentDesignDisclosure"]["confirmatoryClaimAllowed"] is False),
        "model_registration_state_service_and_execution_boundary_holds": all(result["access"][key] == 0 for key in (
            "evaluation_record_count", "manual_judgment_count", "model_load_count", "model_generation_count",
            "API_call_count", "training_run_count", "ontology_registration_count", "trusted_state_mutation_count",
            "real_service_call_count", "external_side_effect_count", "actual_execution_count",
        )),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "167-exact-evidence-gathering-planner-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "scientific_planner_gates_passed": result["passed"],
        "decision": "freeze_positive_development_V167_planner_outcome" if passed and result["passed"] else "reject_V167_outcome",
        "checks": checks,
        "independent_summary": summary,
        "additional_access": {
            "frozen_prediction_read_count": 1,
            "hidden_development_truth_read_count": 1,
            "model_load_count": 0,
            "API_call_count": 0,
            "trusted_state_mutation_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "planner_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V166_outcome": PROJECT_ROOT / lock["parent_V166_outcome"],
        "baseline_predictions": PROJECT_ROOT / lock["baseline_predictions"],
        "hidden_records": PROJECT_ROOT / lock["hidden_records"],
    }
    for key, integrity in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / integrity["path"]
    outcome: dict[str, Any] = {
        "schema_version": "167-exact-evidence-gathering-planner-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_planner_gates_passed": result["passed"],
            "development_informed_not_confirmatory": True,
            "decision": result["decision"],
            "summary": summary,
        },
        "authorization": {
            "modify_or_rerun_V167": False,
            "retain_V167_as_project_authored_development_mechanism_evidence": True,
            "claim_fresh_or_external_confirmation": False,
            "preregister_fixed_ontology_reversible_sandbox": bool(result["passed"]),
            "run_sandbox_without_separate_lock": False,
            "run_local_or_API_model": False,
            "register_provisional_primitive": False,
            "grant_candidate_or_planner_trusted_state_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
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
