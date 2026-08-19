#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v188_binary_clarification_channel_frontier import audit_frontier, build_frozen_problem, evaluate_frontier


DEPENDENCY_KEYS = (
    "config", "parent_V187r1_outcome", "source_V187_lock", "question_codebook",
    "contract_answer_vectors", "development_bindings", "source_V187_result",
    "source_V187_problem_summary", "plan", "protocol", "tests", "runner", "verifier",
    "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    question_payload = json.loads((PROJECT_ROOT / lock["question_codebook"]).read_text())
    vector_payload = json.loads((PROJECT_ROOT / lock["contract_answer_vectors"]).read_text())
    development_payload = json.loads((PROJECT_ROOT / lock["development_bindings"]).read_text())
    config = json.loads(json.dumps(lock["config_payload"]))
    config["_source_v187_result"] = json.loads((PROJECT_ROOT / lock["source_V187_result"]).read_text())
    problem = build_frozen_problem(question_payload, vector_payload, development_payload, config)
    result = evaluate_frontier(problem, config)
    return problem, result, audit_frontier(result, config)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v188-binary-clarification-channel-frontier-lock.json"
    output_root = PROJECT_ROOT / "outputs/v188-binary-clarification-channel-frontier/frontier"
    if output_root.exists():
        raise RuntimeError("V188 frontier census may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V188 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V188 dependency drifted: {key}")
    _, evaluation, audit = reconstruct(lock)
    config = lock["config_payload"]
    decision = (
        "freeze_V188_frontier_and_authorize_multiway_feasibility_protocol_design_only"
        if audit["passed"] and evaluation["summary"]["authorize_multiway_feasibility_design"]
        else "freeze_V188_frontier_without_multiway_successor"
    )
    paths = {
        "information_controls": output_root / "information-controls.json",
        "restricted_exact_tree": output_root / "restricted-exact-tree.json",
        "cost_frontier": output_root / "cost-frontier.json",
        "policy_breakpoints": output_root / "policy-breakpoints.json",
        "frontier_summary": output_root / "frontier-summary.json",
    }
    payloads = {
        "information_controls": evaluation["information"],
        "restricted_exact_tree": evaluation["restricted"],
        "cost_frontier": {"grid": evaluation["frontier"]["grid"]},
        "policy_breakpoints": {"breakpoints": evaluation["frontier"]["policy_breakpoints"]},
        "frontier_summary": evaluation["summary"],
    }
    for key, path in paths.items():
        write_json(path, payloads[key])
    integrity = {key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)} for key, path in paths.items()}
    access = {
        "entropy_computation_count": 1,
        "Huffman_build_count": 1,
        "restricted_exact_tree_build_count": 1,
        "cost_frontier_cell_score_count": len(evaluation["frontier"]["grid"]),
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
    output = {
        "schema_version": "188-binary-clarification-channel-frontier-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "summary": evaluation["summary"],
        "frontier_gates": audit["checks"],
        "access": access,
        "output_integrity": integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
