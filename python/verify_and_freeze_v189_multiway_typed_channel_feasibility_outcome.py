#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v189_multiway_typed_channel_feasibility import DEPENDENCY_KEYS, reconstruct


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v189-multiway-typed-channel-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v189-multiway-typed-channel-feasibility/feasibility"
    result_path = output_root / "result.json"
    doc_path = PROJECT_ROOT / "docs/v189-multiway-typed-channel-feasibility-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v189_multiway_typed_channel_feasibility_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v189-multiway-typed-channel-feasibility/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v189-multiway-typed-channel-feasibility-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V189 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V189 results before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    problem, evaluation, independent = reconstruct(lock)
    expected = {
        "question_definitions": {
            "questions": [
                {"question_id": q.question_id, "field": q.field, "outcomes_by_contract": dict(zip(problem["contract_ids"], q.outcomes))}
                for q in problem["questions"]
            ],
            "coarse_question_ids": list(problem["coarse_question_ids"]),
        },
        "pricing_scenarios": {"scenarios": evaluation["scenarios"]},
        "target_paths": evaluation["paths"],
        "feasibility_summary": evaluation["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == payload
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"]) == result["output_integrity"][key]["sha256"]
        for key, payload in expected.items()
    )
    config = lock["config_payload"]
    if independent["passed"] and evaluation["summary"]["robust_multiway_value"]:
        expected_decision = config["decisionRule"]["ifRobustMultiwayValuePasses"]
    elif independent["passed"] and evaluation["summary"]["conditional_multiway_value"]:
        expected_decision = config["decisionRule"]["ifOnlyConditionalMultiwayValuePasses"]
    else:
        expected_decision = config["decisionRule"]["otherwise"]
    zero_access = (
        "utterance_or_dialogue_language_read_count", "protected_utterance_language_read_count",
        "model_load_count", "model_generation_count", "API_call_count", "training_run_count",
        "ontology_registration_count", "trusted_state_mutation_count", "service_call_count",
        "external_side_effect_count", "actual_execution_count",
    )
    checks = {
        "lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "all_multiway_outputs_reconstruct_exactly": outputs_exact,
        "summary_gates_and_decision_reconstruct_exactly": bool(
            result["summary"] == evaluation["summary"]
            and result["feasibility_gates"] == independent["checks"]
            and result["passed"] == independent["passed"]
            and result["decision"] == expected_decision
        ),
        "every_scenario_is_terminally_exact_and_safe": bool(
            evaluation["summary"]["all_scenario_minimum_exactness_rate"] == 1.0
            and evaluation["summary"]["all_scenario_minimum_retention_rate"] == 1.0
        ),
        "language_model_authority_and_effect_access_is_zero": all(result["access"][key] == 0 for key in zero_access),
    }
    verified = all(checks.values())
    outcome_audit = {
        "schema_version": "189-multiway-typed-channel-feasibility-outcome-audit",
        "experiment": lock["experiment"],
        "passed": verified,
        "scientific_feasibility_gates_passed": independent["passed"],
        "robust_multiway_value": evaluation["summary"]["robust_multiway_value"],
        "conditional_multiway_value": evaluation["summary"]["conditional_multiway_value"],
        "decision": "freeze_verified_V189_feasibility" if verified else "reject_V189_outcome",
        "checks": checks,
        "independent_summary": evaluation["summary"],
        "additional_access": {
            "pricing_scenario_score_count": 0,
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
        "feasibility_lock": lock_path, "result": result_path, "verifier": verifier_path,
        "audit": audit_path, "results_document": doc_path,
        "parent_V188_outcome": PROJECT_ROOT / lock["parent_V188_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "189-multiway-typed-channel-feasibility-outcome-lock",
        "experiment": lock["experiment"],
        "outcome": {
            "passed": True,
            "scientific_feasibility_gates_passed": independent["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rerun_or_redefine_V189": False,
            "preregister_clean_multiway_protocol": bool(independent["passed"] and evaluation["summary"]["robust_multiway_value"]),
            "require_external_cost_or_proposal_evidence_before_successor": bool(evaluation["summary"]["conditional_multiway_value"] and not evaluation["summary"]["robust_multiway_value"]),
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
