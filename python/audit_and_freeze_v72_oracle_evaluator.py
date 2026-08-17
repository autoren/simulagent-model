#!/usr/bin/env python3
"""Audit and freeze the one-shot V72 engineered-oracle evaluator."""
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
    design_lock_path = PROJECT_ROOT / "configs/v72-active-sensing-oracle-design-lock.json"
    fixture_path = PROJECT_ROOT / "python/v72_active_sensing_oracles.py"
    tests_path = PROJECT_ROOT / "python/test_v72_active_sensing_oracles.py"
    planning_path = PROJECT_ROOT / "python/v71_exact_planning.py"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v72_active_sensing_oracle.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v72_oracle_evaluator.py"
    audit_path = PROJECT_ROOT / "outputs/v72-active-sensing/oracle-evaluator-audit.json"
    lock_path = PROJECT_ROOT / "configs/v72-active-sensing-oracle-evaluator-lock.json"
    if lock_path.exists():
        raise RuntimeError("V72 oracle evaluator is already frozen")

    design_lock = json.loads(design_lock_path.read_text())
    design_payload = {
        key: value for key, value in design_lock.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    design_ok = bool(
        payload_hash(design_payload) == design_lock["lock_payload_sha256"]
        and design_lock["authorization"]["implement_and_audit_oracle_evaluator"]
        and not design_lock["authorization"]["run_oracle_outcomes"]
        and not design_lock["authorization"]["inspect_external_candidate_metadata"]
        and not design_lock["authorization"][
            "compute_external_candidate_policy_values_actions_regrets_or_EIG"
        ]
    )
    if not design_ok:
        errors.append("V72 design lock or evaluator-only authorization failed")

    v71_outcome_path = PROJECT_ROOT / design_lock["V71_outcome_lock"]
    v71_outcome = json.loads(v71_outcome_path.read_text())
    v71_evaluator_path = PROJECT_ROOT / v71_outcome["evaluator_lock"]
    v71_evaluator = json.loads(v71_evaluator_path.read_text())
    v71_chain_ok = bool(
        file_sha256(v71_outcome_path) == design_lock["V71_outcome_lock_sha256"]
        and file_sha256(v71_evaluator_path) == v71_outcome["evaluator_lock_sha256"]
        and file_sha256(planning_path) == v71_evaluator["planning_core_sha256"]
    )
    if not v71_chain_ok:
        errors.append("V71 outcome-to-planning lock chain drifted")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "python",
            "-p",
            "test_v72_active_sensing_oracles.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 8 tests" in combined
    if not tests_ok:
        errors.append(f"V72 structural tests failed: {combined[-1600:]}")

    source = evaluator_path.read_text()
    source_checks = {
        "uses_locked_v71_planner": "from v71_exact_planning import" in source,
        "uses_v72_fixture_core": "from v72_active_sensing_oracles import" in source,
        "durable_attempt_before_fixture_evaluation": source.index("attempt_path.write_text")
        < source.index('evaluate_fixture("positive"'),
        "separate_oracle_output": "outputs/v72-active-sensing/oracle-evaluation" in source,
        "external_outcome_firewall": "external_candidate_policy_values_computed" in source,
        "V71_protected_firewall": "V71_protected_access_count" in source,
        "non_evidence_claim": "not scientific evidence" in source,
        "exactly_two_fixed_fixtures": source.count('evaluate_fixture("positive"') == 1
        and source.count('evaluate_fixture("negative_control"') == 1,
    }
    if not all(source_checks.values()):
        errors.append("V72 evaluator expands beyond the frozen oracle protocol")

    outcome_absent = not (
        PROJECT_ROOT / "outputs/v72-active-sensing/oracle-evaluation"
    ).exists()
    external_absent = not (
        PROJECT_ROOT / "outputs/v72-active-sensing/external-source-inventory.json"
    ).exists()
    if not outcome_absent or not external_absent:
        errors.append("V72 oracle outcome or external inventory predates evaluator lock")

    checks = {
        "design_lock_and_evaluator_only_authorization": design_ok,
        "eight_outcome_free_structural_tests": tests_ok,
        "one_shot_oracle_only_evaluator": all(source_checks.values()),
        "oracle_outcome_absent_before_lock": outcome_absent,
        "external_inventory_absent_before_lock": external_absent,
        "V71_outcome_to_planning_lock_chain_unchanged": v71_chain_ok,
    }
    audit = {
        "schema_version": "72-active-sensing-oracle",
        "experiment": "v72_oracle_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_evaluator_and_authorize_one_engineered_oracle_run"
            if not errors
            else "reject_v72_oracle_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "structural_test_cases": 8,
            "oracle_fixture_outcomes_evaluated": 0,
            "external_candidate_metadata_records_read": 0,
            "external_candidate_policy_values_computed": 0,
            "V71_protected_access_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "72-active-sensing-oracle",
        "experiment": "v72_active_sensing_oracle_evaluator_lock",
        "design_lock": str(design_lock_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_lock_path),
        "fixture_core": str(fixture_path.relative_to(PROJECT_ROOT)),
        "fixture_core_sha256": file_sha256(fixture_path),
        "fixture_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "fixture_tests_sha256": file_sha256(tests_path),
        "planning_core": str(planning_path.relative_to(PROJECT_ROOT)),
        "planning_core_sha256": file_sha256(planning_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "evaluator_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "evaluator_auditor_sha256": file_sha256(auditor_path),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "attempt_path": "outputs/v72-active-sensing/oracle-evaluation/attempt.json",
        "result_path": "outputs/v72-active-sensing/oracle-evaluation/result.json",
        "expected_attempt_number": 1,
        "expected_fixture_count": 2,
        "authorization": {
            "modify_V71_or_V72_design_fixture_evaluator_or_gates": False,
            "run_engineered_oracle_outcomes_once": True,
            "inspect_external_candidate_metadata": False,
            "compute_external_candidate_policy_values_actions_regrets_or_EIG": False,
            "read_V71_protected_models_histories_or_outcomes": False,
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
