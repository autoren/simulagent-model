#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v123-fresh-population-availability-audit.json"
    plan_path = PROJECT_ROOT / "docs/v123-fresh-population-availability-audit-plan.md"
    protocol_path = PROJECT_ROOT / "python/v123_fresh_population_availability_audit.py"
    tests_path = PROJECT_ROOT / "python/test_v123_fresh_population_availability_audit.py"
    runner_path = PROJECT_ROOT / "python/run_v123_fresh_population_availability_audit.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v123_fresh_population_availability_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v123_fresh_population_availability.py"
    audit_path = PROJECT_ROOT / "outputs/v123-fresh-population-availability-audit/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v123-fresh-population-availability-audit-lock.json"
    result_path = PROJECT_ROOT / "outputs/v123-fresh-population-availability-audit/audit/result.json"
    if any(path.exists() for path in (audit_path, lock_path, result_path)):
        raise RuntimeError("V123 already frozen or evaluated")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV122OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    excluded_paths = [PROJECT_ROOT / path for path in config["excludedPopulations"]]
    auth = parent["authorization"]
    checks = {
        "V122_is_valid_and_authorizes_fresh_design_only": bool(
            valid_lock(parent) and valid_lock(parent_lock)
            and parent["outcome"]["passed"] and parent["outcome"]["audit_pass"]
            and auth["preregister_fresh_model_free_retrieval_geometry_design"]
            and not auth["evaluate_signals_or_fit_trigger"]
            and not auth["run_language_model_or_protected"]
            and not auth["run_API_training_action_or_execution"]
        ),
        "availability_audit_is_text_free_aggregate_only": bool(
            config["outcomeGates"]["maximumIndividualCandidateEmissionCount"] == 0
            and config["outcomeGates"]["maximumLanguageReadCount"] == 0
            and config["outcomeGates"]["maximumActualExecutionCount"] == 0
            and all(value == 0 for value in config["accessGates"].values())
        ),
        "success_authorizes_only_external_source_feasibility": bool(
            config["decisionRule"]["passAuthorizesOnlyExternalControlledOpenSetSourceFeasibilityAudit"]
            and not config["decisionRule"]["passAuthorizesShrinkReuseLanguageSignalTriggerOrModelEvaluation"]
            and not config["decisionRule"]["passAuthorizesProtectedInductionAPITrainingActionOrExecution"]
        ),
        "sources_code_and_output_absence_hold": bool(
            inventory_path.is_file() and all(path.is_file() for path in excluded_paths)
            and all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path))
            and not result_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "123-fresh-population-availability-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "decision": "freeze_and_authorize_one_text_free_availability_audit" if passed else "reject_V123_design",
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    deps: dict[str, Path] = {
        "config": config_path,
        "parent_outcome": parent_path,
        "parent_analysis_lock": parent_lock_path,
        "source_inventory": inventory_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    for index, path in enumerate(excluded_paths):
        deps[f"excluded_population_{index}"] = path
    lock: dict[str, Any] = {
        "schema_version": "123-fresh-population-availability-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_one_text_free_aggregate_availability_audit": True,
            "modify_counts_requirements_exclusions_gates_or_decision": False,
            "read_language_or_individual_candidates": False,
            "evaluate_signal_trigger_or_model": False,
            "grant_protected_induction_authority_or_execution": False,
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
