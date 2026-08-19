#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def valid_lock(value: dict[str, Any]) -> bool:
    payload = {key: item for key, item in value.items() if key != "lock_payload_sha256"}
    return payload_hash(payload) == value.get("lock_payload_sha256")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v122-prequery-signal-inventory.json"
    plan_path = PROJECT_ROOT / "docs/v122-prequery-signal-inventory-plan.md"
    protocol_path = PROJECT_ROOT / "python/v122_prequery_signal_inventory.py"
    tests_path = PROJECT_ROOT / "python/test_v122_prequery_signal_inventory.py"
    runner_path = PROJECT_ROOT / "python/run_v122_prequery_signal_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v122_prequery_signal_inventory_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v122_prequery_signal_inventory.py"
    audit_path = PROJECT_ROOT / "outputs/v122-prequery-signal-inventory/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v122-prequery-signal-inventory-lock.json"
    result_path = PROJECT_ROOT / "outputs/v122-prequery-signal-inventory/audit/result.json"
    if any(path.exists() for path in (audit_path, lock_path, result_path)):
        raise RuntimeError("V122 already frozen or evaluated")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV121OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    parent_result_path = PROJECT_ROOT / parent["result"]
    auth = parent["authorization"]
    sources = [PROJECT_ROOT / path for path in config["sourceDefinitions"]]

    checks = {
        "V121_is_valid_and_authorizes_only_inventory": bool(
            valid_lock(parent)
            and valid_lock(parent_lock)
            and parent["outcome"]["passed"]
            and parent["outcome"]["audit_pass"]
            and auth["inventory_independently_available_prequery_signals"]
            and not auth["evaluate_or_fit_trigger"]
            and not auth["run_language_model_or_protected"]
            and not auth["begin_induction_or_richer_planning"]
            and not auth["run_API_training_action_or_execution"]
            and file_sha256(parent_result_path) == parent["result_sha256"]
        ),
        "inventory_is_definition_only_and_records_mutability": bool(
            config["outcomeGates"]["maximumIndividualRecordReadCount"] == 0
            and config["outcomeGates"]["maximumActualExecutionCount"] == 0
            and config["outcomeGates"]["requireMutabilityRecorded"]
        ),
        "no_evaluation_language_model_protected_or_execution_authorized": bool(
            all(value == 0 for value in config["accessGates"].values())
            and config["decisionRule"]["passAuthorizesOnlyPreregisterFreshModelFreeRetrievalGeometryDesign"]
            and not config["decisionRule"]["passAuthorizesSignalEvaluationOrTriggerFit"]
            and not config["decisionRule"]["passAuthorizesLanguageModelOrProtectedRun"]
            and not config["decisionRule"]["passAuthorizesInductionAPITrainingActionOrExecution"]
        ),
        "all_sources_and_code_exist_and_outputs_absent": bool(
            all(path.is_file() for path in sources)
            and all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path))
            and not result_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "122-prequery-signal-inventory-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "decision": "freeze_and_authorize_one_static_inventory" if passed else "reject_V122_design",
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    deps = {
        "config": config_path,
        "parent_outcome": parent_path,
        "parent_analysis_lock": parent_lock_path,
        "parent_result": parent_result_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    for index, source in enumerate(sources):
        deps[f"source_definition_{index}"] = source
    lock: dict[str, Any] = {
        "schema_version": "122-prequery-signal-inventory-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_one_static_definition_inventory": True,
            "modify_sources_inventory_gates_or_decision": False,
            "read_individual_records_language_or_raw_responses": False,
            "evaluate_signals_or_fit_trigger": False,
            "load_or_generate_with_model": False,
            "grant_any_authority_or_execution": False,
        },
    }
    for key, path in deps.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
