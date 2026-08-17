#!/usr/bin/env python3
"""Audit and lock the one-shot V69 development evaluator."""
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
    seal_path = PROJECT_ROOT / "configs/v69-development-census-seal.json"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v69_development_screen.py"
    tests_path = PROJECT_ROOT / "python/test_v69_development_evaluator.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v69_development_evaluator.py"
    audit_path = PROJECT_ROOT / "outputs/v69-development-screening/evaluator-audit.json"
    lock_path = PROJECT_ROOT / "configs/v69-development-evaluator-lock.json"
    if lock_path.exists():
        raise RuntimeError("V69 evaluator already frozen")
    seal = json.loads(seal_path.read_text())
    seal_payload = {key: value for key, value in seal.items() if key != "lock_payload_sha256"}
    errors: list[str] = []
    seal_ok = bool(
        payload_hash(seal_payload) == seal["lock_payload_sha256"]
        and seal["authorization"]["write_and_audit_durable_development_evaluator"]
        and not seal["authorization"]["run_development_screen"]
        and not seal["authorization"]["score_confirmatory_models"]
        and seal["record_count"] == 59
        and seal["selection_rejection_or_replacement_count"] == 0
        and seal["confirmatory_models_scored"] == 0
    )
    if not seal_ok:
        errors.append("V69 census seal or evaluator-only authorization failed")

    implementation_lock_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation_lock = json.loads(implementation_lock_path.read_text())
    family_path = PROJECT_ROOT / implementation_lock["implementation"]
    point_lock_path = PROJECT_ROOT / "configs/v68r2-development-implementation-lock.json"
    point_lock = json.loads(point_lock_path.read_text())
    point_path = PROJECT_ROOT / point_lock["implementation"]
    base_lock_path = PROJECT_ROOT / "configs/v68-development-evaluator-lock.json"
    base_lock = json.loads(base_lock_path.read_text())
    base_path = PROJECT_ROOT / base_lock["evaluator"]
    dependencies_ok = bool(
        file_sha256(implementation_lock_path) == seal["implementation_lock_sha256"]
        and file_sha256(family_path) == implementation_lock["implementation_sha256"]
        and file_sha256(point_path) == point_lock["implementation_sha256"]
        and file_sha256(base_path) == base_lock["evaluator_sha256"]
    )
    if not dependencies_ok:
        errors.append("V69 family, point controls, or base evaluator drifted")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "python",
            "-p",
            "test_v69_development_evaluator.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 3 tests" in combined
    if not tests_ok:
        errors.append(f"V69 evaluator tests failed: {combined[-1200:]}")

    source = evaluator_path.read_text()
    source_checks = {
        "uses_dominant_remapping_family": "build_dominant_remapping_family" in source,
        "uses_totalized_evaluate_record": (
            "from evaluate_v68r2_development_screen import evaluate_record" in source
        ),
        "durable_attempt_before_rows": source.index("attempt_path.write_text") < source.index("rows = []"),
        "separate_V69_output_directory": "outputs/v69-development-screening/evaluation" in source,
        "confirmatory_firewall": "score_confirmatory_models" in source,
        "no_hardcoded_confirmatory_filename": not any(
            name in source
            for name in (
                "cheese.95.POMDP",
                "fully_observable_tmaze2.POMDP",
                "hallway.POMDP",
                "heavenhell.POMDP",
                "network.POMDP",
                "shuttle.POMDP",
                "paint.POMDP",
            )
        ),
    }
    source_ok = all(source_checks.values())
    if not source_ok:
        errors.append("V69 evaluator expands beyond frozen development protocol")
    evaluation_absent = not (
        PROJECT_ROOT / "outputs/v69-development-screening/evaluation"
    ).exists()
    if not evaluation_absent:
        errors.append("V69 evaluation exists before evaluator lock")

    checks = {
        "census_seal_and_evaluator_only_authorization": seal_ok,
        "locked_family_point_controls_and_base_evaluator": dependencies_ok,
        "three_synthetic_evaluator_tests": tests_ok,
        "durable_development_only_evaluator_source": source_ok,
        "evaluation_absent_before_lock": evaluation_absent,
    }
    audit = {
        "schema_version": "69-development-screening",
        "experiment": "v69_development_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_evaluator_and_authorize_one_V69_development_screen"
            if not errors
            else "reject_v69_development_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "synthetic_evaluator_records": 3,
            "sealed_development_records_evaluated": 0,
            "confirmatory_models_scored": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "69-development-screening",
        "experiment": "v69_development_evaluator_lock",
        "development_census_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "development_census_seal_sha256": file_sha256(seal_path),
        "family_implementation": str(family_path.relative_to(PROJECT_ROOT)),
        "family_implementation_sha256": file_sha256(family_path),
        "point_control_implementation": str(point_path.relative_to(PROJECT_ROOT)),
        "point_control_implementation_sha256": file_sha256(point_path),
        "unchanged_V68_evaluator": str(base_path.relative_to(PROJECT_ROOT)),
        "unchanged_V68_evaluator_sha256": file_sha256(base_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "evaluator_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "evaluator_tests_sha256": file_sha256(tests_path),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "evaluator_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "evaluator_auditor_sha256": file_sha256(auditor_path),
        "attempt_path": "outputs/v69-development-screening/evaluation/attempt.json",
        "expected_attempt_number": 1,
        "expected_records": 59,
        "expected_confirmatory_models_scored": 0,
        "authorization": {
            "modify_design_implementation_census_evaluator_or_gates": False,
            "run_development_screen_once": True,
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
