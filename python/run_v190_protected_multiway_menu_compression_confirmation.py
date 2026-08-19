#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v190_protected_multiway_menu_compression_confirmation import audit, build_confirmation_problem, evaluate


DEPENDENCY_KEYS = (
    "config", "parent_V189_outcome", "source_V189_lock", "contract_catalog",
    "protected_bindings", "source_V189_result", "plan", "protocol", "tests",
    "runner", "verifier", "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    catalog = json.loads((PROJECT_ROOT / lock["contract_catalog"]).read_text())
    protected = json.loads((PROJECT_ROOT / lock["protected_bindings"]).read_text())
    source = json.loads((PROJECT_ROOT / lock["source_V189_result"]).read_text())
    problem = build_confirmation_problem(catalog, protected, lock["config_payload"])
    result = evaluate(problem, lock["config_payload"], source)
    return problem, result, audit(result, lock["config_payload"])


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v190-protected-multiway-menu-compression-confirmation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v190-protected-multiway-menu-compression-confirmation/confirmation"
    if output_root.exists():
        raise RuntimeError("V190 confirmation may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V190 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V190 dependency drifted: {key}")
    _, evaluation, audited = reconstruct(lock)
    config = lock["config_payload"]
    decision = (
        config["decisionRule"]["ifEveryFreshCompressionSafetyAndCostGatePasses"]
        if audited["passed"] else config["decisionRule"]["otherwise"]
    )
    paths = {
        "confirmation_summary": output_root / "confirmation-summary.json",
        "target_paths": output_root / "target-paths.json",
        "protected_record_results": output_root / "protected-record-results.json",
    }
    payloads = {
        "confirmation_summary": evaluation["summary"],
        "target_paths": evaluation["paths"],
        "protected_record_results": {"records": evaluation["records"]},
    }
    for key, path in paths.items():
        write_json(path, payloads[key])
    integrity = {key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)} for key, path in paths.items()}
    access = {
        "protected_binding_record_read_count": evaluation["summary"]["protected_binding_count"],
        "protected_target_path_score_count": evaluation["summary"]["contract_count"] * 4,
        "policy_optimization_count": 0,
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
        "schema_version": "190-protected-multiway-menu-compression-confirmation-result",
        "experiment": config["experiment"],
        "passed": audited["passed"],
        "decision": decision,
        "summary": evaluation["summary"],
        "confirmation_gates": audited["checks"],
        "access": access,
        "output_integrity": integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
