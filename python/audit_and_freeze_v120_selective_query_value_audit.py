#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json
from pathlib import Path
from typing import Any
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def valid_lock(value: dict[str, Any]) -> bool: return payload_hash({k: v for k, v in value.items() if k != "lock_payload_sha256"}) == value.get("lock_payload_sha256")
def write_json(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v120-selective-query-value-audit.json"; plan_path = PROJECT_ROOT / "docs/v120-selective-query-value-audit-plan.md"; protocol_path = PROJECT_ROOT / "python/v120_selective_query_value_audit.py"; tests_path = PROJECT_ROOT / "python/test_v120_selective_query_value_audit.py"; runner_path = PROJECT_ROOT / "python/run_v120_selective_query_value_audit.py"; verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v120_selective_query_value_audit_outcome.py"; auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v120_selective_query_value_audit.py"; audit_path = PROJECT_ROOT / "outputs/v120-selective-query-value-audit/design-audit.json"; lock_path = PROJECT_ROOT / "configs/v120-selective-query-value-audit-lock.json"; result_path = PROJECT_ROOT / "outputs/v120-selective-query-value-audit/audit/result.json"
    if any(p.exists() for p in (audit_path, lock_path, result_path)): raise RuntimeError("V120 already frozen or evaluated")
    config = json.loads(config_path.read_text()); parent_path = PROJECT_ROOT / config["parentV119OutcomeLock"]; parent = json.loads(parent_path.read_text()); parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]; parent_lock = json.loads(parent_lock_path.read_text()); parent_result_path = PROJECT_ROOT / parent["result"]
    checks = {
        "V119_is_valid_negative_with_only_regret_gate_failed": bool(valid_lock(parent) and valid_lock(parent_lock) and parent["outcome"]["passed"] and not parent["outcome"]["simulator_pass"] and parent["outcome"]["decision"] == "adaptive_causal_feasibility_fails_keep_language_and_model_closed" and not any(parent["authorization"].values()) and file_sha256(parent_result_path) == parent["result_sha256"]),
        "aggregate_decomposition_only": config["outcomeGates"]["maximumIndividualRecordReadCount"] == 0 and config["outcomeGates"]["maximumIndividualRecordEmissionCount"] == 0 and config["decomposition"]["fixedClarificationCost"] == parent_lock["config_payload"]["adaptiveTree"]["totalCostEveryPath"],
        "no_retune_language_model_protected_or_execution_authorized": bool(all(v == 0 for v in config["accessGates"].values()) and config["decisionRule"]["passAuthorizesOnlyPreregisterModelFreePrequeryTriggerFeasibilityAudit"] and not config["decisionRule"]["passAuthorizesLanguageModelOrProtectedRun"] and not config["decisionRule"]["passAuthorizesRetuneV119OrDiscountCost"] and not config["decisionRule"]["passAuthorizesInductionAPITrainingActionOrExecution"]),
        "all_code_exists_and_outputs_absent": all(p.is_file() for p in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)) and not result_path.exists(),
    }
    passed = all(checks.values()); audit = {"schema_version": "120-selective-query-value-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "decision": "freeze_and_authorize_one_aggregate_cost_decomposition" if passed else "reject_V120_design"}; write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path, "parent_result": parent_result_path, "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "120-selective-query-value-audit-lock", "experiment": config["experiment"], "config_payload": config, "authorization": {"run_one_aggregate_cost_decomposition": True, "modify_parent_cost_metrics_gates_or_decision": False, "read_individual_records_language_or_raw_responses": False, "load_or_generate_with_model": False, "grant_any_authority_or_execution": False}}
    for key, path in deps.items(): lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock); print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
