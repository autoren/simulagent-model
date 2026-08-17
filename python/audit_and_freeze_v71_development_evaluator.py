#!/usr/bin/env python3
"""Audit and freeze the one-shot exact V71 development evaluator."""
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
    seal_path = PROJECT_ROOT / "configs/v71-sensor-codebook-development-census-seal.json"
    evaluator_path = (
        PROJECT_ROOT / "python/evaluate_v71_sensor_codebook_development.py"
    )
    planning_path = PROJECT_ROOT / "python/v71_exact_planning.py"
    planning_tests_path = PROJECT_ROOT / "python/test_v71_exact_planning.py"
    belief_path = PROJECT_ROOT / "python/v71_sensor_codebook.py"
    parser_path = PROJECT_ROOT / "python/v71_cassandra_pomdp.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v71_development_evaluator.py"
    audit_path = PROJECT_ROOT / "outputs/v71-sensor-codebook/development-evaluator-audit.json"
    lock_path = (
        PROJECT_ROOT / "configs/v71-sensor-codebook-development-evaluator-lock.json"
    )
    if lock_path.exists():
        raise RuntimeError("V71 development evaluator is already frozen")

    seal = json.loads(seal_path.read_text())
    seal_payload = {key: value for key, value in seal.items() if key != "lock_payload_sha256"}
    errors: list[str] = []
    seal_ok = bool(
        payload_hash(seal_payload) == seal["lock_payload_sha256"]
        and seal["authorization"]["write_and_audit_development_evaluator"]
        and not seal["authorization"]["run_development_outcomes"]
        and not seal["authorization"][
            "read_protected_confirmation_histories_or_outcomes"
        ]
        and not seal["authorization"][
            "select_filter_drop_or_replace_records_or_models"
        ]
        and seal["record_count"] == 21
    )
    if not seal_ok:
        errors.append("V71 census seal or evaluator-only authorization failed")

    dependencies_ok = bool(
        file_sha256(PROJECT_ROOT / seal["belief_core"]) == seal["belief_core_sha256"]
        and file_sha256(PROJECT_ROOT / seal["source_lock"])
        == seal["source_lock_sha256"]
        and file_sha256(PROJECT_ROOT / seal["census"]) == seal["census_sha256"]
    )
    if not dependencies_ok:
        errors.append("V71 belief core, source lock, or sealed census drifted")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "python",
            "-p",
            "test_v71_exact_planning.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 5 tests" in combined
    if not tests_ok:
        errors.append(f"V71 exact evaluator tests failed: {combined[-1600:]}")

    source = evaluator_path.read_text()
    source_checks = {
        "uses_frozen_exact_planning_core": "from v71_exact_planning import" in source,
        "uses_sealed_joint_belief": "joint_belief_latent_by_state" in source,
        "durable_attempt_before_source_parse": source.index("attempt_path.write_text")
        < source.index("parsed = parse_cassandra_pomdp_file"),
        "separate_development_output": (
            'outputs/v71-sensor-codebook/development-evaluation' in source
        ),
        "protected_firewall": (
            "read_protected_confirmation_histories_or_outcomes" in source
            and "protected_confirmation_policy_value_count" in source
        ),
        "no_hardcoded_protected_filename": not any(
            name in source
            for name in (
                "bridge-repair.POMDP",
                "ejs2.POMDP",
                "ejs3.POMDP",
                "parr95.95.POMDP",
                "uav-search.raissa-bravo.orig.POMDP",
            )
        ),
        "no_fallback_path": "fallback_count\": 0" in source
        and "off-support branch" not in source,
    }
    source_ok = all(source_checks.values())
    if not source_ok:
        errors.append("V71 evaluator expands beyond the frozen development protocol")

    evaluation_absent = not (
        PROJECT_ROOT / "outputs/v71-sensor-codebook/development-evaluation"
    ).exists()
    if not evaluation_absent:
        errors.append("V71 evaluation exists before evaluator lock")

    checks = {
        "census_seal_and_evaluator_only_authorization": seal_ok,
        "locked_belief_source_and_census_dependencies": dependencies_ok,
        "five_synthetic_exact_planning_tests": tests_ok,
        "durable_fallback_free_development_only_evaluator": source_ok,
        "evaluation_absent_before_lock": evaluation_absent,
    }
    audit = {
        "schema_version": "71-sensor-codebook-development",
        "experiment": "v71_development_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_evaluator_and_authorize_one_complete_development_screen"
            if not errors
            else "reject_v71_development_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "synthetic_test_cases": 5,
            "sealed_development_records_evaluated": 0,
            "protected_confirmation_policy_value_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "71-sensor-codebook-development",
        "experiment": "v71_sensor_codebook_development_evaluator_lock",
        "development_census_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "development_census_seal_sha256": file_sha256(seal_path),
        "source_parser": str(parser_path.relative_to(PROJECT_ROOT)),
        "source_parser_sha256": file_sha256(parser_path),
        "belief_core": str(belief_path.relative_to(PROJECT_ROOT)),
        "belief_core_sha256": file_sha256(belief_path),
        "planning_core": str(planning_path.relative_to(PROJECT_ROOT)),
        "planning_core_sha256": file_sha256(planning_path),
        "planning_tests": str(planning_tests_path.relative_to(PROJECT_ROOT)),
        "planning_tests_sha256": file_sha256(planning_tests_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "evaluator_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "evaluator_auditor_sha256": file_sha256(auditor_path),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "attempt_path": "outputs/v71-sensor-codebook/development-evaluation/attempt.json",
        "expected_attempt_number": 1,
        "expected_records": 21,
        "expected_development_models": 3,
        "expected_protected_confirmation_policy_value_count": 0,
        "authorization": {
            "modify_source_family_partition_resource_census_evaluator_or_gates": False,
            "run_development_outcomes_once": True,
            "read_protected_confirmation_histories_or_outcomes": False,
            "select_filter_drop_or_replace_records_or_models": False,
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
