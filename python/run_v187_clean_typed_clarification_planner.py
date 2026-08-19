#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v187_clean_typed_clarification_planner import audit_evaluation, build_problem, evaluate


DEPENDENCY_KEYS = (
    "config", "parent_V186_outcome", "question_codebook", "contract_answer_vectors",
    "development_bindings", "protected_bindings", "plan", "protocol", "tests",
    "runner", "verifier", "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    question_payload = json.loads((PROJECT_ROOT / lock["question_codebook"]).read_text())
    vector_payload = json.loads((PROJECT_ROOT / lock["contract_answer_vectors"]).read_text())
    development_payload = json.loads((PROJECT_ROOT / lock["development_bindings"]).read_text())
    config = json.loads(json.dumps(lock["config_payload"]))
    config["_raw_questions"] = question_payload["questions"]
    problem = build_problem(question_payload, vector_payload, development_payload, config)
    evaluation = evaluate(problem, config)
    return problem, evaluation, audit_evaluation(evaluation, config)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v187-clean-typed-clarification-planner-lock.json"
    output_root = PROJECT_ROOT / "outputs/v187-clean-typed-clarification-planner/evaluation"
    if output_root.exists():
        raise RuntimeError("V187 clean development evaluation may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V187 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V187 dependency drifted: {key}")
    problem, evaluation, audit = reconstruct(lock)
    config = lock["config_payload"]
    decision = (
        config["decisionRule"]["ifEverySafetyExactnessCostAdaptivityAndDominanceGatePasses"]
        if audit["passed"] else config["decisionRule"]["otherwise"]
    )
    problem_summary = {
        "contract_ids": list(problem["contract_ids"]),
        "prior_counts": problem["prior_counts"],
        "prior": {key: float(value) for key, value in problem["prior"].items()},
        "raw_question_count": len(config.get("_raw_questions", [])),
        "partition_distinct_questions": [
            {"question_id": q.question_id, "family": q.family, "value": q.value, "column": list(q.column)}
            for q in problem["questions"]
        ],
        "horizon": problem["horizon"],
        "typed_question_cost": float(problem["typed_cost"]),
        "generic_trusted_clarification_cost": float(problem["generic_cost"]),
        "safe_deferral_cost": float(problem["deferral_cost"]),
    }
    paths = {
        "problem_summary": output_root / "problem-summary.json",
        "policy_summary": output_root / "policy-summary.json",
        "target_paths": output_root / "target-paths.json",
        "development_record_results": output_root / "development-record-results.json",
    }
    payloads = {
        "problem_summary": problem_summary,
        "policy_summary": evaluation["summary"],
        "target_paths": evaluation["by_target"],
        "development_record_results": {"records": evaluation["record_rows"]},
    }
    for key, path in paths.items():
        write_json(path, payloads[key])
    integrity = {key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)} for key, path in paths.items()}
    access = {
        "policy_build_count": 7,
        "policy_score_count": 7,
        "development_target_path_evaluation_count": 14 * 7,
        "protected_utterance_language_read_count": 0,
        "utterance_or_dialogue_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    result = {
        "schema_version": "187-clean-typed-clarification-planner-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "summary": evaluation["summary"],
        "development_gates": audit["checks"],
        "access": access,
        "output_integrity": integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
