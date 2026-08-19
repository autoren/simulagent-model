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
    config_path = PROJECT_ROOT / "configs/v121-prequery-value-selectivity-envelope.json"; plan_path = PROJECT_ROOT / "docs/v121-prequery-value-selectivity-envelope-plan.md"; protocol_path = PROJECT_ROOT / "python/v121_prequery_value_selectivity_envelope.py"; tests_path = PROJECT_ROOT / "python/test_v121_prequery_value_selectivity_envelope.py"; runner_path = PROJECT_ROOT / "python/run_v121_prequery_value_selectivity_envelope.py"; verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v121_prequery_value_selectivity_envelope_outcome.py"; auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v121_prequery_value_selectivity_envelope.py"; audit_path = PROJECT_ROOT / "outputs/v121-prequery-value-selectivity-envelope/design-audit.json"; lock_path = PROJECT_ROOT / "configs/v121-prequery-value-selectivity-envelope-lock.json"; result_path = PROJECT_ROOT / "outputs/v121-prequery-value-selectivity-envelope/audit/result.json"
    if any(p.exists() for p in (audit_path, lock_path, result_path)): raise RuntimeError("V121 already frozen or evaluated")
    config = json.loads(config_path.read_text()); parent_path = PROJECT_ROOT / config["parentV120OutcomeLock"]; parent = json.loads(parent_path.read_text()); parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]; parent_lock = json.loads(parent_lock_path.read_text()); parent_result_path = PROJECT_ROOT / parent["result"]
    auth = parent["authorization"]
    checks = {
        "V120_is_valid_and_authorizes_only_model_free_prequery_audit": bool(valid_lock(parent) and valid_lock(parent_lock) and parent["outcome"]["passed"] and parent["outcome"]["audit_pass"] and auth["preregister_model_free_prequery_trigger_feasibility_audit"] and not auth["run_language_model_or_protected"] and not auth["retune_V119_or_discount_cost"] and not auth["begin_induction_or_richer_planning"] and not auth["run_API_training_action_or_execution"] and file_sha256(parent_result_path) == parent["result_sha256"]),
        "envelope_is_aggregate_and_trigger_cannot_be_certified": config["outcomeGates"]["maximumIndividualRecordReadCount"] == 0 and config["outcomeGates"]["maximumIndividualRecordEmissionCount"] == 0 and config["outcomeGates"]["requireAggregateMetricsInsufficientToCertifyAnyTrigger"],
        "no_evaluation_retune_language_model_protected_or_execution_authorized": bool(all(v == 0 for v in config["accessGates"].values()) and config["decisionRule"]["passAuthorizesOnlyPrequerySignalInventory"] and not config["decisionRule"]["passAuthorizesTriggerEvaluation"] and not config["decisionRule"]["passAuthorizesLanguageModelOrProtectedRun"] and not config["decisionRule"]["passAuthorizesRetuneCostInductionAPITrainingActionOrExecution"]),
        "all_code_exists_and_outputs_absent": all(p.is_file() for p in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)) and not result_path.exists(),
    }
    passed = all(checks.values()); audit = {"schema_version": "121-prequery-value-selectivity-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "decision": "freeze_and_authorize_one_aggregate_value_envelope" if passed else "reject_V121_design"}; write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path, "parent_result": parent_result_path, "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "121-prequery-value-selectivity-envelope-lock", "experiment": config["experiment"], "config_payload": config, "authorization": {"run_one_aggregate_value_envelope": True, "modify_parent_cost_metrics_fractions_gates_or_decision": False, "read_individual_records_language_or_raw_responses": False, "evaluate_or_fit_trigger": False, "load_or_generate_with_model": False, "grant_any_authority_or_execution": False}}
    for key, path in deps.items(): lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock); print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
