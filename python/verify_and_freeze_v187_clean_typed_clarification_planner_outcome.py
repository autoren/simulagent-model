#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v187_clean_typed_clarification_planner import DEPENDENCY_KEYS, reconstruct


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v187-clean-typed-clarification-planner-lock.json"
    output_root = PROJECT_ROOT / "outputs/v187-clean-typed-clarification-planner/evaluation"
    result_path = output_root / "result.json"
    doc_path = PROJECT_ROOT / "docs/v187-clean-typed-clarification-planner-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v187_clean_typed_clarification_planner_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v187-clean-typed-clarification-planner/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v187-clean-typed-clarification-planner-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V187 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V187 results before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    problem, evaluation, independent = reconstruct(lock)
    config = lock["config_payload"]
    expected_problem = {
        "contract_ids": list(problem["contract_ids"]),
        "prior_counts": problem["prior_counts"],
        "prior": {key: float(value) for key, value in problem["prior"].items()},
        "raw_question_count": len(json.loads((PROJECT_ROOT / lock["question_codebook"]).read_text())["questions"]),
        "partition_distinct_questions": [
            {"question_id": q.question_id, "family": q.family, "value": q.value, "column": list(q.column)}
            for q in problem["questions"]
        ],
        "horizon": problem["horizon"],
        "typed_question_cost": float(problem["typed_cost"]),
        "generic_trusted_clarification_cost": float(problem["generic_cost"]),
        "safe_deferral_cost": float(problem["deferral_cost"]),
    }
    expected = {
        "problem_summary": expected_problem,
        "policy_summary": evaluation["summary"],
        "target_paths": evaluation["by_target"],
        "development_record_results": {"records": evaluation["record_rows"]},
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == payload
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"]) == result["output_integrity"][key]["sha256"]
        for key, payload in expected.items()
    )
    expected_decision = (
        config["decisionRule"]["ifEverySafetyExactnessCostAdaptivityAndDominanceGatePasses"]
        if independent["passed"] else config["decisionRule"]["otherwise"]
    )
    zero_access = (
        "protected_utterance_language_read_count", "utterance_or_dialogue_language_read_count",
        "model_load_count", "model_generation_count", "API_call_count", "training_run_count",
        "ontology_registration_count", "trusted_state_mutation_count", "service_call_count",
        "external_side_effect_count", "actual_execution_count",
    )
    checks = {
        "lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "all_policy_outputs_reconstruct_exactly": outputs_exact,
        "summary_gates_and_decision_reconstruct_exactly": bool(
            result["summary"] == evaluation["summary"]
            and result["development_gates"] == independent["checks"]
            and result["passed"] == independent["passed"]
            and result["decision"] == expected_decision
        ),
        "exact_policy_preserves_terminal_authority": bool(
            evaluation["summary"]["policy_summary"]["exact_adaptive"]["final_exactness_rate"] == 1.0
            and evaluation["summary"]["policy_summary"]["exact_adaptive"]["target_retention_rate"] == 1.0
        ),
        "protected_language_model_authority_and_effect_access_is_zero": all(result["access"][key] == 0 for key in zero_access),
    }
    verified = all(checks.values())
    outcome_audit = {
        "schema_version": "187-clean-typed-clarification-planner-outcome-audit",
        "experiment": config["experiment"],
        "passed": verified,
        "scientific_development_gates_passed": independent["passed"],
        "decision": "freeze_verified_V187_clean_development" if verified else "reject_V187_outcome",
        "checks": checks,
        "independent_summary": evaluation["summary"],
        "additional_access": {
            "protected_utterance_language_read_count": 0,
            "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, outcome_audit)
    if not verified:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "planner_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V186_outcome": PROJECT_ROOT / lock["parent_V186_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "187-clean-typed-clarification-planner-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_development_gates_passed": independent["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rerun_or_redefine_V187": False,
            "preregister_correlated_error_protocol": bool(independent["passed"]),
            "run_error_stress_without_separate_lock": False,
            "read_protected_or_utterance_language_run_model_API_or_training": False,
            "register_mutate_call_service_act_or_execute": False,
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
