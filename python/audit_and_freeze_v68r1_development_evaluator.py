#!/usr/bin/env python3
"""Audit and lock the repaired one-shot V68r1 development evaluator."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    implementation_lock_path = PROJECT_ROOT / "configs/v68r1-development-implementation-lock.json"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v68r1_development_screen.py"
    tests_path = PROJECT_ROOT / "python/test_v68r1_development_evaluator.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v68r1_development_evaluator.py"
    audit_path = PROJECT_ROOT / "outputs/v68r1-development-screening/evaluator-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68r1-development-evaluator-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68r1 evaluator already frozen")
    implementation = json.loads(implementation_lock_path.read_text())
    payload = {
        key: value for key, value in implementation.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    implementation_ok = bool(
        payload_hash(payload) == implementation["lock_payload_sha256"]
        and implementation["authorization"]["write_and_audit_repaired_durable_evaluator"]
        and not implementation["authorization"]["run_repaired_development_screen"]
        and not implementation["authorization"]["score_confirmatory_models"]
    )
    if not implementation_ok:
        errors.append("V68r1 implementation lock or evaluator-only authorization failed")

    repair_design_path = PROJECT_ROOT / implementation["repair_design_lock"]
    repair_design = json.loads(repair_design_path.read_text())
    failed_path = PROJECT_ROOT / repair_design["failed_attempt_lock"]
    census_path = PROJECT_ROOT / repair_design["unchanged_artifacts"]["census"]["path"]
    old_evaluator_lock_path = PROJECT_ROOT / "configs/v68-development-evaluator-lock.json"
    old_evaluator_lock = json.loads(old_evaluator_lock_path.read_text())
    old_evaluator_path = PROJECT_ROOT / old_evaluator_lock["evaluator"]
    unchanged_ok = bool(
        file_sha256(failed_path) == repair_design["failed_attempt_lock_sha256"]
        and file_sha256(census_path)
        == repair_design["unchanged_artifacts"]["census"]["sha256"]
        and file_sha256(old_evaluator_path) == old_evaluator_lock["evaluator_sha256"]
    )
    if not unchanged_ok:
        errors.append("failed attempt, census, or original evaluator changed before repair lock")

    completed = subprocess.run(
        [
            sys.executable, "-m", "unittest", "discover", "-s", "python",
            "-p", "test_v68r1_development_evaluator.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 3 tests" in combined
    if not tests_ok:
        errors.append("V68r1 repaired evaluator tests failed")

    source = evaluator_path.read_text()
    source_checks = {
        "only_control_replacement_is_totalized_posterior_sampling": (
            "base.persistent_posterior_sampling_mixture = repaired_control" in source
            and "totalized_persistent_posterior_sampling_mixture" in source
        ),
        "original_control_restored_in_finally": (
            "finally:" in source
            and "base.persistent_posterior_sampling_mixture = original" in source
        ),
        "off_support_diagnostics_persisted": all(
            token in source
            for token in (
                "off_support_branch_count",
                "expected_off_support_entry_probability",
                "fallback_action_name",
            )
        ),
        "atomic_new_attempt_before_rows": source.index("attempt_path.write_text") < source.index("rows = []"),
        "separate_V68r1_output_directory": "outputs/v68r1-development-screening/evaluation" in source,
        "repair_and_original_hash_checks": all(
            token in source
            for token in (
                "repair implementation hash mismatch",
                "unchanged V68 evaluator hash mismatch",
            )
        ),
        "confirmatory_firewall": "score_confirmatory_models" in source,
        "no_hardcoded_confirmatory_filename": not any(
            name in source
            for name in (
                "cheese.95.POMDP", "fully_observable_tmaze2.POMDP", "hallway.POMDP",
                "heavenhell.POMDP", "network.POMDP", "shuttle.POMDP", "paint.POMDP",
            )
        ),
    }
    source_ok = all(source_checks.values())
    if not source_ok:
        errors.append("V68r1 evaluator expands beyond or fails to durably bind the repair")

    evaluation_absent = not (
        PROJECT_ROOT / "outputs/v68r1-development-screening/evaluation"
    ).exists()
    if not evaluation_absent:
        errors.append("V68r1 evaluation exists before repaired evaluator lock")

    census = json.loads(census_path.read_text())
    failed = json.loads(failed_path.read_text())
    locks_ok = bool(
        census["record_count"] == 59
        and failed["record_results_persisted"] == 0
        and not failed["aggregate_result_persisted"]
        and failed["confirmatory_models_scored"] == 0
    )
    if not locks_ok:
        errors.append("unchanged census or failed-attempt facts differ")

    checks = {
        "repair_implementation_binding_and_evaluator_only_authorization": implementation_ok,
        "failed_attempt_census_and_original_evaluator_unchanged": unchanged_ok,
        "three_repaired_evaluator_tests": tests_ok,
        "repair_only_durable_evaluator_source": source_ok,
        "unchanged_59_record_and_failed_attempt_facts": locks_ok,
        "repaired_evaluation_absent_before_lock": evaluation_absent,
    }
    audit = {
        "schema_version": "68r1-development-screening",
        "experiment": "v68r1_development_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_repaired_evaluator_and_authorize_one_repaired_development_screen"
            if not errors
            else "reject_v68r1_development_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "synthetic_repaired_evaluator_records": 3,
            "additional_sealed_development_records_evaluated": 0,
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
    repair_implementation_path = PROJECT_ROOT / implementation["implementation"]
    lock = {
        "schema_version": "68r1-development-screening",
        "experiment": "v68r1_development_evaluator_lock",
        "repair_implementation_lock": str(implementation_lock_path.relative_to(PROJECT_ROOT)),
        "repair_implementation_lock_sha256": file_sha256(implementation_lock_path),
        "repair_implementation": str(repair_implementation_path.relative_to(PROJECT_ROOT)),
        "repair_implementation_sha256": file_sha256(repair_implementation_path),
        "development_census_seal": str(census_path.relative_to(PROJECT_ROOT)),
        "development_census_seal_sha256": file_sha256(census_path),
        "unchanged_V68_evaluator": str(old_evaluator_path.relative_to(PROJECT_ROOT)),
        "unchanged_V68_evaluator_sha256": file_sha256(old_evaluator_path),
        "source_failed_attempt": str(failed_path.relative_to(PROJECT_ROOT)),
        "source_failed_attempt_sha256": file_sha256(failed_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "evaluator_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "evaluator_tests_sha256": file_sha256(tests_path),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "evaluator_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "evaluator_auditor_sha256": file_sha256(auditor_path),
        "attempt_path": "outputs/v68r1-development-screening/evaluation/attempt.json",
        "expected_attempt_number": 1,
        "expected_records": 59,
        "expected_confirmatory_models_scored": 0,
        "authorization": {
            "modify_failed_V68_or_V68r1_locks_code_census_or_gates": False,
            "run_repaired_development_screen_once": True,
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
