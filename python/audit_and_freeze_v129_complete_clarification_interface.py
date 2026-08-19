#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v129-complete-clarification-interface.json"
    plan_path = PROJECT_ROOT / "docs/v129-complete-clarification-interface-plan.md"
    protocol_path = PROJECT_ROOT / "python/v129_complete_clarification_interface.py"
    tests_path = PROJECT_ROOT / "python/test_v129_complete_clarification_interface.py"
    runner_path = PROJECT_ROOT / "python/run_v129_complete_clarification_interface.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v129_complete_clarification_interface_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v129_complete_clarification_interface.py"
    audit_path = PROJECT_ROOT / "outputs/v129-complete-clarification-interface/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v129-complete-clarification-interface-lock.json"
    result_path = PROJECT_ROOT / "outputs/v129-complete-clarification-interface/evaluation/result.json"
    if any(path.exists() for path in (audit_path, lock_path, result_path)): raise RuntimeError("V129 already frozen or evaluated")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV128OutcomeLock"]
    parent = json.loads(parent_path.read_text()); parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text()); auth = parent["authorization"]
    catalog_path = PROJECT_ROOT / config["choiceCatalog"]; baseline_path = PROJECT_ROOT / config["baselineConfig"]
    v119_path = PROJECT_ROOT / config["V119Config"]; catalog = json.loads(catalog_path.read_text())
    channel = config["completeClarificationChannel"]
    checks = {
        "V128_is_valid_negative_and_localizes_interface": bool(
            valid_lock(parent) and valid_lock(parent_lock) and parent["outcome"]["passed"]
            and parent["outcome"]["audit_pass"] and not parent["outcome"]["experimental_pass"]
            and auth["close_annotation_signature_family"] and not auth["run_language_model_or_protected"]
            and not auth["run_API_training_action_or_execution"]
        ),
        "complete_interface_and_census_are_exact": bool(
            catalog["choice_count"] == 11 and config["census"]["pairCount"] == 66
            and channel["requiredReliability"] == 0.95 and channel["totalCost"] == 0.3
            and channel["errorBiasFraction"] == 0.75 and channel["oneAnswerOnly"] and channel["noIndependenceClaim"]
            and set(channel["errorRegimes"]) == {"symmetric", "candidate_attraction", "abstention_attraction"}
        ),
        "candidate_comparator_is_same_cost_and_frozen": config["candidateSpecificComparator"]["sameTotalCost"] == channel["totalCost"] == 0.3 and config["candidateSpecificComparator"]["marginalCorrectness"] == 0.95,
        "model_language_authority_and_execution_closed": bool(
            all(value == 0 for value in config["accessGates"].values())
            and not config["decisionRule"]["passAuthorizesImmediateLanguageHumanOrModelRun"]
            and not config["decisionRule"]["passAuthorizesProtectedInductionOrRicherPlanning"]
            and not config["decisionRule"]["passAuthorizesAPITrainingActionOrExecution"]
            and config["frozenBayesPolicy"]["completeSafeHypothesisUniverseAlwaysRetained"]
            and config["frozenBayesPolicy"]["actualExecutionCount"] == 0
        ),
        "dependencies_code_and_output_absence_hold": all(path.is_file() for path in (catalog_path, baseline_path, v119_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)) and not result_path.exists(),
    }
    passed = all(checks.values())
    audit = {"schema_version": "129-complete-clarification-interface-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "decision": "freeze_and_authorize_one_model_free_complete_interface_audit" if passed else "reject_V129_design", "prelock_access": {key: 0 for key in config["accessGates"]} | {"actual_execution_count": 0}}
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path, "choice_catalog": catalog_path, "baseline_config": baseline_path, "V119_config": v119_path, "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "129-complete-clarification-interface-lock", "experiment": config["experiment"], "config_payload": config, "authorization": {"run_one_model_free_complete_interface_audit": True, "modify_census_channel_bias_cost_comparator_gates_or_decision": False, "read_language_records_or_load_model": False, "grant_protected_induction_authority_or_execution": False}}
    for key, path in deps.items(): lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
