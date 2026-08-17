#!/usr/bin/env python3
"""Freeze the failed V68r1 attempt before the all-point-control repair."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    evaluator_lock_path = PROJECT_ROOT / "configs/v68r1-development-evaluator-lock.json"
    attempt_path = PROJECT_ROOT / "outputs/v68r1-development-screening/evaluation/attempt.json"
    failure_path = PROJECT_ROOT / "outputs/v68r1-development-screening/evaluation/failure.json"
    audit_path = PROJECT_ROOT / "outputs/v68r1-development-screening/failed-attempt-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68r1-development-failed-attempt-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68r1 failed attempt already frozen")
    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    payload = {
        key: value for key, value in evaluator_lock.items() if key != "lock_payload_sha256"
    }
    attempt = json.loads(attempt_path.read_text())
    failure = json.loads(failure_path.read_text())
    errors: list[str] = []
    binding_ok = bool(
        payload_hash(payload) == evaluator_lock["lock_payload_sha256"]
        and evaluator_lock["authorization"]["run_repaired_development_screen_once"]
        and not evaluator_lock["authorization"]["score_confirmatory_models"]
        and attempt["evaluator_lock_sha256"] == file_sha256(evaluator_lock_path)
        and attempt["attempt_number"] == 1
        and attempt["confirmatory_models_scored"] == 0
    )
    evidence_ok = bool(
        failure["failure_type"] == "RuntimeError"
        and failure["failure_location"] == "map_model_policy -> evaluate_policy"
        and "MAP_point_model_policy_is_partial" in failure["failure_classification"]
        and failure["record_results_persisted"] == 0
        and not failure["aggregate_result_persisted"]
        and failure["confirmatory_models_scored"] == 0
        and failure["SMC2_runs"] == 0
    )
    results_absent = not any(
        path.exists()
        for path in (
            attempt_path.parent / "record-results.jsonl",
            attempt_path.parent / "result.json",
            PROJECT_ROOT / "configs/v68r1-development-outcome-lock.json",
        )
    )
    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v68r2-development-repair-design-lock.json",
            "python/v68r2_point_model_controls.py",
            "python/evaluate_v68r2_development_screen.py",
            "outputs/v68r2-development-screening/evaluation",
        )
    )
    if not binding_ok:
        errors.append("V68r1 evaluator or attempt binding failed")
    if not evidence_ok:
        errors.append("V68r1 MAP partial-policy failure evidence differs")
    if not results_absent:
        errors.append("V68r1 record or aggregate results unexpectedly persisted")
    if not downstream_absent:
        errors.append("V68r2 artifacts exist before V68r1 failure lock")
    checks = {
        "V68r1_evaluator_and_attempt_binding": binding_ok,
        "MAP_partial_policy_failure_evidence": evidence_ok,
        "no_record_or_aggregate_results": results_absent,
        "V68r2_artifacts_absent": downstream_absent,
    }
    audit = {
        "schema_version": "68r1-development-screening",
        "experiment": "v68r1_failed_attempt_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_V68r1_failure_and_authorize_all_point_model_control_totalization_preregistration_only"
            if not errors
            else "reject_V68r1_failure_evidence"
        ),
        "errors": errors,
        "checks": checks,
        "access": {
            "additional_development_records_evaluated": 0,
            "confirmatory_models_scored": 0,
            "SMC2_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "68r1-development-screening",
        "experiment": "v68r1_development_failed_attempt_lock",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "failure": str(failure_path.relative_to(PROJECT_ROOT)),
        "failure_sha256": file_sha256(failure_path),
        "failed_attempt_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "failed_attempt_audit_sha256": file_sha256(audit_path),
        "record_results_persisted": 0,
        "aggregate_result_persisted": False,
        "confirmatory_models_scored": 0,
        "authorization": {
            "modify_or_rerun_V68_or_V68r1_failed_evaluators": False,
            "preregister_all_point_model_control_totalization": True,
            "run_V68r2_before_new_locks": False,
            "score_confirmatory_models": False,
            "run_SMC2": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
