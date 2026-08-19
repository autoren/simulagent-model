#!/usr/bin/env python3
"""Audit and freeze the one V88r1 mechanical name-preservation repair."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v88r1-name-preservation-repair.json"
    execution_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-execution-lock.json"
    original_impl_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-implementation-lock.json"
    plan_path = PROJECT_ROOT / "docs/v88r1-name-preservation-repair-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v88r1_repair_design.py"
    prior_failed_audit_path = PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/design-audit.json"
    audit_path = PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/design-audit-r1a.json"
    lock_path = PROJECT_ROOT / "configs/v88r1-name-preservation-repair-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V88r1 repair design is already frozen")
    if (PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/evaluation").exists():
        raise RuntimeError("V88r1 evaluation exists before repair design lock")

    config = json.loads(design_path.read_text())
    execution = json.loads(execution_path.read_text())
    execution_payload = {key: value for key, value in execution.items() if key != "lock_payload_sha256"}
    original = json.loads(original_impl_path.read_text())
    original_payload = {key: value for key, value in original.items() if key != "lock_payload_sha256"}
    retry = config["retry"]
    stage = config["stageAuthorization"]
    checks = {
        "V88_failure_lock_exact_inconclusive_and_authorizes_one_repair": bool(
            payload_hash(execution_payload) == execution["lock_payload_sha256"]
            and execution["status"] == "execution_inconclusive"
            and execution["scientific_outcome"] is None
            and execution["authorization"]["preregister_one_mechanical_name_preservation_retry"]
            and execution["authorization"]["retry_may_only_preserve_registered_fixture_name"]
            and not execution["authorization"]["authorize_any_further_retry"]
        ),
        "original_implementation_exact_and_failed_attempt_had_no_result": bool(
            payload_hash(original_payload) == original["lock_payload_sha256"]
            and file_sha256(original_impl_path) == execution["implementation_lock_sha256"]
            and not (PROJECT_ROOT / "outputs/v88-external-intent-candidate/evaluation/result.json").exists()
            and execution["access"]["model_load_count"] == 1
            and execution["access"]["model_generation_count"] == 1
        ),
        "only_name_preservation_change_is_authorized": bool(
            config["onlyAuthorizedChange"].startswith("copy the harness-provided fixture name")
            and len(config["frozenUnchangedDependencies"]) == 9
        ),
        "retry_and_cumulative_budgets_are_exact_and_terminal": bool(
            retry["maximumRetryModelLoadCount"] == 1
            and retry["maximumRetryModelGenerationCount"] == 48
            and retry["priorFailedModelLoadCount"] == 1
            and retry["priorFailedModelGenerationCount"] == 1
            and retry["maximumCumulativeModelLoadCount"] == execution["authorization"]["maximum_cumulative_model_load_count_after_retry"] == 2
            and retry["maximumCumulativeModelGenerationCount"] == execution["authorization"]["maximum_cumulative_model_generation_count_after_retry"] == 49
            and not retry["retryOnMalformedOutput"]
            and not retry["anyFurtherRetryAuthorized"]
        ),
        "design_stage_has_no_inference_API_training_manual_language_or_execution_authority": bool(
            stage["auditAndFreezeRepair"]
            and not stage["modifyOriginalV88Files"]
            and not stage["implementNamePreservingRunner"]
            and not stage["runLocalModel"]
            and not stage["runAPIModel"]
            and not stage["trainAdapter"]
            and not stage["inspectSourceLanguageManually"]
            and not stage["performRealServiceCall"]
            and not stage["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "88r1-name-preservation-repair-design-audit",
        "experiment": "v88r1_name_preservation_repair_design_audit",
        "passed": passed,
        "decision": "freeze_repair_and_authorize_name_preserving_runner_implementation" if passed else "reject_V88r1_repair",
        "checks": checks,
        "prior_failed_audit": str(prior_failed_audit_path.relative_to(PROJECT_ROOT)),
        "prior_failed_audit_sha256": file_sha256(prior_failed_audit_path),
        "audit_correction": "expected frozen dependency category count corrected from 8 to the explicit design count 9",
        "access": {"model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "manual_utterance_inspection_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "88r1-name-preservation-repair-design-lock",
        "experiment": "v88r1_name_preservation_repair_design_lock",
        "repair_design": str(design_path.relative_to(PROJECT_ROOT)),
        "repair_design_sha256": file_sha256(design_path),
        "config_payload": config,
        "parent_execution_lock": str(execution_path.relative_to(PROJECT_ROOT)),
        "parent_execution_lock_sha256": file_sha256(execution_path),
        "original_implementation_lock": str(original_impl_path.relative_to(PROJECT_ROOT)),
        "original_implementation_lock_sha256": file_sha256(original_impl_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_repair_or_original_scientific_dependencies": False,
            "implement_and_audit_name_preserving_runner": True,
            "run_local_model": False,
            "run_API_model_or_train_adapter": False,
            "perform_real_service_call_or_external_side_effect": False,
            "authorize_further_retry": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
