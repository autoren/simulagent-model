#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v189_multiway_typed_channel_feasibility import audit, build_problem, evaluate


DEPENDENCY_KEYS = (
    "config", "parent_V188_outcome", "source_V188_lock", "source_V187_lock",
    "source_V186_outcome", "source_V186_lock", "contract_catalog", "development_bindings", "source_V187_result",
    "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    catalog = json.loads((PROJECT_ROOT / lock["contract_catalog"]).read_text())
    bindings = json.loads((PROJECT_ROOT / lock["development_bindings"]).read_text())
    problem = build_problem(catalog, bindings, lock["config_payload"])
    result = evaluate(problem, lock["config_payload"])
    return problem, result, audit(result, lock["config_payload"])


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v189-multiway-typed-channel-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v189-multiway-typed-channel-feasibility/feasibility"
    if output_root.exists():
        raise RuntimeError("V189 feasibility census may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V189 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V189 dependency drifted: {key}")
    problem, evaluation, audited = reconstruct(lock)
    config = lock["config_payload"]
    if audited["passed"] and evaluation["summary"]["robust_multiway_value"]:
        decision = config["decisionRule"]["ifRobustMultiwayValuePasses"]
    elif audited["passed"] and evaluation["summary"]["conditional_multiway_value"]:
        decision = config["decisionRule"]["ifOnlyConditionalMultiwayValuePasses"]
    else:
        decision = config["decisionRule"]["otherwise"]
    question_definitions = {
        "questions": [
            {"question_id": q.question_id, "field": q.field, "outcomes_by_contract": dict(zip(problem["contract_ids"], q.outcomes))}
            for q in problem["questions"]
        ],
        "coarse_question_ids": list(problem["coarse_question_ids"]),
    }
    paths = {
        "question_definitions": output_root / "question-definitions.json",
        "pricing_scenarios": output_root / "pricing-scenarios.json",
        "target_paths": output_root / "target-paths.json",
        "feasibility_summary": output_root / "feasibility-summary.json",
    }
    payloads = {
        "question_definitions": question_definitions,
        "pricing_scenarios": {"scenarios": evaluation["scenarios"]},
        "target_paths": evaluation["paths"],
        "feasibility_summary": evaluation["summary"],
    }
    for key, path in paths.items():
        write_json(path, payloads[key])
    integrity = {key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)} for key, path in paths.items()}
    access = {
        "multiway_question_build_count": 1,
        "pricing_scenario_score_count": len(evaluation["scenarios"]),
        "policy_score_count": len(evaluation["scenarios"]) * 4,
        "utterance_or_dialogue_language_read_count": 0,
        "protected_utterance_language_read_count": 0,
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
    output = {
        "schema_version": "189-multiway-typed-channel-feasibility-result",
        "experiment": config["experiment"],
        "passed": audited["passed"],
        "decision": decision,
        "summary": evaluation["summary"],
        "feasibility_gates": audited["checks"],
        "access": access,
        "output_integrity": integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
