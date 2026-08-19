#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v170_unchanged_planner_fresh_confirmation import evaluate_fresh_population, evaluate_integrity_gates, evaluate_strong_thresholds


DEPENDENCY_KEYS = ("config", "parent_V169r1_outcome", "source_V167_planner_lock", "source_V167r1_outcome", "constraint_states", "eligible_state_ids", "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit")


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> dict[str, Any]:
    states = json.loads((PROJECT_ROOT / lock["constraint_states"]).read_text())
    eligible = json.loads((PROJECT_ROOT / lock["eligible_state_ids"]).read_text())
    return evaluate_fresh_population(states, eligible, lock["V167_config_payload"])


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v170-unchanged-planner-fresh-confirmation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v170-unchanged-planner-fresh-confirmation/scoring"
    if output_root.exists(): raise RuntimeError("V170 may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]: raise RuntimeError("V170 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V170 dependency drifted: {key}")
    evaluation = reconstruct(lock); config = lock["config_payload"]
    access = {"formal_fresh_policy_score_count": len(evaluation["cases"]), "evaluation_record_count": 0, "manual_judgment_count": 0, "model_load_count": 0, "model_generation_count": 0, "API_call_count": 0, "training_run_count": 0, "ontology_registration_count": 0, "trusted_state_mutation_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0, "actual_execution_count": 0}
    integrity = evaluate_integrity_gates(evaluation, access, config); strong = evaluate_strong_thresholds(evaluation, config)
    passed = all(integrity.values()); strong_passed = all(strong.values())
    if passed and strong_passed: decision = config["decisionRule"]["ifIntegrityAndStrongThresholdsPass"]
    elif passed: decision = config["decisionRule"]["ifIntegrityPassesButStrongThresholdsFail"]
    else: decision = config["decisionRule"]["otherwise"]
    cases_path = output_root / "case-policy-results.json"; summary_path = output_root / "confirmation-summary.json"
    write_json(cases_path, {"cases": evaluation["cases"], "contains_language": False, "shadow_only": True})
    write_json(summary_path, evaluation["summary"])
    integrity_outputs = {key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)} for key, path in {"case_policy_results": cases_path, "confirmation_summary": summary_path}.items()}
    result = {"schema_version": "170-unchanged-planner-fresh-confirmation-result", "experiment": config["experiment"], "passed": passed, "strong_confirmation": strong_passed, "decision": decision, "summary": evaluation["summary"], "integrity_gates": integrity, "strong_thresholds": strong, "access": access, "output_integrity": integrity_outputs, "claim_boundary": config["claimBoundary"]}
    write_json(output_root / "result.json", result); print(json.dumps(result, indent=2, sort_keys=True))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
