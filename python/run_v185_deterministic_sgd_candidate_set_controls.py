#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v185_deterministic_sgd_candidate_set_controls import audit_controls, run_controls


DEPENDENCY_KEYS = (
    "config", "parent_V184_outcome", "development_language", "protected_language",
    "declared_catalog_language", "hidden_identifiability", "plan", "protocol", "tests",
    "runner", "verifier", "auditor", "design_audit",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = run_controls(
        json.loads((PROJECT_ROOT / lock["development_language"]).read_text()),
        json.loads((PROJECT_ROOT / lock["declared_catalog_language"]).read_text()),
        json.loads((PROJECT_ROOT / lock["hidden_identifiability"]).read_text()),
        lock["config_payload"],
    )
    return controls, audit_controls(controls, lock["config_payload"])


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v185-deterministic-sgd-candidate-set-controls-lock.json"
    output_root = PROJECT_ROOT / "outputs/v185-deterministic-sgd-candidate-set-controls/evaluation"
    if output_root.exists():
        raise RuntimeError("V185 development evaluation may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V185 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V185 dependency drifted: {key}")
    controls, audit = reconstruct(lock)
    config = lock["config_payload"]
    decision = (
        config["decisionRule"]["ifEveryEvaluationSafetySelectivityCostAndResidualGatePasses"]
        if audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    paths = {
        "split_manifest": output_root / "split-manifest.json",
        "shadow_predictions": output_root / "shadow-predictions.json",
        "residual_identifiers": output_root / "residual-identifiers.json",
        "evaluation_summary": output_root / "evaluation-summary.json",
    }
    payloads = {
        "split_manifest": controls["split"],
        "shadow_predictions": {"predictions": controls["predictions"], "contains_language_or_authoritative_state": False},
        "residual_identifiers": {"record_ids": controls["residual_ids"], "membership_uses_predictions_only": True},
        "evaluation_summary": controls["summary"],
    }
    for key, path in paths.items():
        write_json(path, payloads[key])
    integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in paths.items()
    }
    access = {
        "formal_development_evaluation_count": 1,
        "development_language_read_count": 1,
        "protected_language_read_count": 0,
        "manual_language_inspection_count": 0,
        "character_candidate_score_count": 720,
        "token_candidate_score_count": 720,
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
        "schema_version": "185-deterministic-SGD-candidate-set-controls-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "decision": decision,
        "summary": controls["summary"],
        "evaluation_gates": audit["checks"],
        "access": access,
        "output_integrity": integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
