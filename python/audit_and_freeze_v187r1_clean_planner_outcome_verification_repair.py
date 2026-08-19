#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v187_clean_typed_clarification_planner import DEPENDENCY_KEYS


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v187r1-clean-planner-outcome-verification-repair.json"
    plan_path = PROJECT_ROOT / "docs/v187r1-clean-planner-outcome-verification-repair-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v187r1_clean_planner_outcome_verification_repair.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v187r1_clean_planner_outcome_verification_repair.py"
    audit_path = PROJECT_ROOT / "outputs/v187r1-clean-planner-outcome-verification-repair/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v187r1-clean-planner-outcome-verification-repair-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v187r1-clean-planner-outcome-verification-repair-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, outcome_path)):
        raise RuntimeError("V187r1 is already preregistered or frozen")
    config = json.loads(config_path.read_text())
    source_lock_path = PROJECT_ROOT / config["sourceV187Lock"]
    result_path = PROJECT_ROOT / config["sourceV187Result"]
    failed_audit_path = PROJECT_ROOT / config["sourceV187FailedOutcomeAudit"]
    source_lock = json.loads(source_lock_path.read_text())
    result = json.loads(result_path.read_text())
    failed = json.loads(failed_audit_path.read_text())
    repair = config["repair"]
    decision = config["decisionRule"]
    checks = {
        "V187_lock_and_every_dependency_are_still_exact": bool(
            payload_hash({key: value for key, value in source_lock.items() if key != "lock_payload_sha256"}) == source_lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / source_lock[key]) == source_lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "V187_result_and_failed_audit_have_expected_boundary": bool(
            result["experiment"] == source_lock["experiment"]
            and not result["passed"]
            and result["decision"] == "freeze_V187_clean_boundary_result_without_error_or_model_successor"
            and not failed["passed"]
            and not failed["checks"]["all_policy_outputs_reconstruct_exactly"]
            and all(value for key, value in failed["checks"].items() if key != "all_policy_outputs_reconstruct_exactly")
        ),
        "repair_is_one_field_verification_only": bool(
            not repair["rerunPolicyEvaluation"]
            and not repair["modifySourceResultOrOutputs"]
            and repair["onlyCorrectExpectedProblemSummaryRawQuestionCount"]
            and repair["frozenRunnerSerializedRawQuestionCountFromUnaugmentedConfigAsZero"]
            and repair["policySummaryIndependentlyRetainsReconstructedRawQuestionCount164"]
            and repair["allOtherExpectedPayloadsUnchanged"]
        ),
        "successor_authority_is_closed": bool(
            not decision["authorizeCorrelatedErrorOrModelSuccessor"]
            and not decision["authorizeProtectedOrUtteranceLanguageAccess"]
            and not decision["authorizeRegistrationAuthorityActionOrExecution"]
            and all(value == 0 for value in config["preLockExposure"].values())
        ),
        "required_files_exist_and_outcomes_are_absent": bool(
            all(path.is_file() for path in (config_path, plan_path, auditor_path, verifier_path, source_lock_path, result_path, failed_audit_path))
            and not outcome_path.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "187r1-clean-planner-outcome-verification-repair-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_verification_only_repair" if passed else "reject_V187r1_design",
        "checks": checks,
        "additional_policy_score_count": 0,
        "protected_utterance_language_read_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "plan": plan_path, "auditor": auditor_path, "verifier": verifier_path,
        "source_V187_lock": source_lock_path, "source_V187_result": result_path,
        "source_V187_failed_outcome_audit": failed_audit_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "187r1-clean-planner-outcome-verification-repair-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_or_rerun_V187": False,
            "run_verification_repair_once": True,
            "read_protected_or_utterance_language_run_model_API_or_training": False,
            "register_mutate_call_service_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
