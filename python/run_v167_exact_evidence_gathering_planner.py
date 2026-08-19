#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v167_exact_evidence_gathering_planner import build_planner_evaluation, evaluate_gates


DEPENDENCY_KEYS = (
    "config", "parent_V166_outcome", "roadmap", "plan", "protocol", "tests",
    "runner", "verifier", "auditor", "design_audit", "baseline_predictions", "hidden_records",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> dict[str, Any]:
    predictions = json.loads((PROJECT_ROOT / lock["baseline_predictions"]).read_text())
    hidden = json.loads((PROJECT_ROOT / lock["hidden_records"]).read_text())
    return build_planner_evaluation(predictions, hidden, lock["config_payload"])


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    output_root = PROJECT_ROOT / "outputs/v167-exact-evidence-gathering/planner"
    if output_root.exists():
        raise RuntimeError("V167 formal planner may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V167 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V167 dependency drifted: {key}")

    evaluation = reconstruct(lock)
    config = lock["config_payload"]
    access = {
        "frozen_prediction_read_count": 1,
        "hidden_development_truth_read_count": 1,
        "formal_policy_score_count": len(evaluation["cases"]),
        "evaluation_record_count": 0,
        "manual_judgment_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    gates = evaluate_gates(evaluation, access, config)
    passed = all(gates.values())
    decision = config["decisionRule"]["ifEveryPlannerGatePasses"] if passed else config["decisionRule"]["otherwise"]
    policies_path = output_root / "case-policy-trees.json"
    metrics_path = output_root / "planner-evaluation.json"
    write_json(policies_path, {"cases": evaluation["cases"], "contains_language": False, "shadow_only": True})
    write_json(metrics_path, {"summary": evaluation["summary"], "contains_language": False})
    output_integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in {"case_policy_trees": policies_path, "planner_evaluation": metrics_path}.items()
    }
    result = {
        "schema_version": "167-exact-evidence-gathering-planner-result",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": decision,
        "summary": evaluation["summary"],
        "gates": gates,
        "access": access,
        "development_informed_not_confirmatory": True,
        "output_integrity": output_integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
