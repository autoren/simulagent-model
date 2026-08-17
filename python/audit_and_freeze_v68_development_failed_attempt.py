#!/usr/bin/env python3
"""Freeze the failed V68 development attempt before any repaired evaluator."""
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
    evaluator_lock_path = PROJECT_ROOT / "configs/v68-development-evaluator-lock.json"
    attempt_path = PROJECT_ROOT / "outputs/v68-development-screening/evaluation/attempt.json"
    failure_path = PROJECT_ROOT / "outputs/v68-development-screening/evaluation/failure.json"
    audit_path = PROJECT_ROOT / "outputs/v68-development-screening/failed-attempt-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68-development-failed-attempt-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68 failed attempt is already frozen")
    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    payload = {
        key: value for key, value in evaluator_lock.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    lock_ok = bool(
        payload_hash(payload) == evaluator_lock["lock_payload_sha256"]
        and evaluator_lock["authorization"]["run_development_screen_once"]
        and not evaluator_lock["authorization"]["score_confirmatory_models"]
    )
    if not lock_ok:
        errors.append("V68 evaluator lock binding failed")
    attempt = json.loads(attempt_path.read_text())
    failure = json.loads(failure_path.read_text())
    evidence_ok = bool(
        attempt["attempt_number"] == 1
        and attempt["evaluator_lock_sha256"] == file_sha256(evaluator_lock_path)
        and attempt["confirmatory_models_scored"] == 0
        and failure["attempt_number"] == 1
        and failure["failure_type"] == "RuntimeError"
        and failure["failure_message"]
        == "V66 policy omits an observation reachable under the evaluation belief"
        and "partial" in failure["failure_classification"]
        and failure["record_results_persisted"] == 0
        and not failure["aggregate_result_persisted"]
        and failure["confirmatory_models_scored"] == 0
        and failure["SMC2_runs"] == 0
    )
    if not evidence_ok:
        errors.append("V68 failed-attempt evidence differs from the observed partial-policy failure")
    result_absent = not any(
        path.exists()
        for path in (
            attempt_path.parent / "record-results.jsonl",
            attempt_path.parent / "result.json",
            PROJECT_ROOT / "configs/v68-development-outcome-lock.json",
        )
    )
    if not result_absent:
        errors.append("V68 result artifacts unexpectedly exist after the failed attempt")
    repair_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v68r1-development-repair-design-lock.json",
            "configs/v68r1-development-evaluator-lock.json",
            "python/v68r1_posterior_sampling.py",
            "python/evaluate_v68r1_development_screen.py",
            "outputs/v68r1-development-screening/evaluation",
        )
    )
    if not repair_absent:
        errors.append("V68r1 repair artifacts exist before the failed attempt lock")
    checks = {
        "V68_evaluator_lock_binding": lock_ok,
        "attempt_and_failure_evidence": evidence_ok,
        "no_record_or_aggregate_results_persisted": result_absent,
        "repair_artifacts_absent": repair_absent,
    }
    audit = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_failed_attempt_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_failed_attempt_and_authorize_preregistration_of_off_support_totalization_repair_only"
            if not errors
            else "reject_v68_failed_attempt_evidence"
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
        "schema_version": "68-development-screening",
        "experiment": "v68_development_failed_attempt_lock",
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
            "modify_or_rerun_failed_V68_evaluator": False,
            "preregister_off_support_totalization_repair": True,
            "run_repaired_development_screen_before_new_locks": False,
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
