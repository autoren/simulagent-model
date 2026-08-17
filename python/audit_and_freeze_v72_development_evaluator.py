#!/usr/bin/env python3
"""Audit and freeze the one-shot V72 RockSample development evaluator."""
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
    resource_lock_path = (
        PROJECT_ROOT / "configs/v72-active-sensing-development-resource-lock.json"
    )
    config_path = (
        PROJECT_ROOT / "configs/v72-active-sensing-development-evaluation.json"
    )
    plan_path = (
        PROJECT_ROOT / "docs/v72-active-sensing-development-evaluation-plan.md"
    )
    exporter_path = PROJECT_ROOT / "python/v72_rocksample_source.py"
    tests_path = PROJECT_ROOT / "python/test_v72_rocksample_source.py"
    planning_path = PROJECT_ROOT / "python/v71_exact_planning.py"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v72_rocksample_development.py"
    auditor_path = (
        PROJECT_ROOT / "python/audit_and_freeze_v72_development_evaluator.py"
    )
    audit_path = (
        PROJECT_ROOT / "outputs/v72-active-sensing/development-evaluator-audit.json"
    )
    lock_path = (
        PROJECT_ROOT / "configs/v72-active-sensing-development-evaluator-lock.json"
    )
    if lock_path.exists():
        raise RuntimeError("V72 development evaluator is already frozen")

    resource_lock = json.loads(resource_lock_path.read_text())
    resource_payload = {
        key: value for key, value in resource_lock.items() if key != "lock_payload_sha256"
    }
    config = json.loads(config_path.read_text())
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(resource_payload) == resource_lock["lock_payload_sha256"]
        and resource_lock["authorization"]["write_and_audit_development_evaluator"]
        and not resource_lock["authorization"]["run_development_outcomes"]
        and not resource_lock["authorization"][
            "compute_candidate_policy_values_actions_regrets_or_EIG"
        ]
        and not resource_lock["authorization"][
            "select_or_inspect_protected_confirmation_models"
        ]
    )
    if not authorization_ok:
        errors.append("V72 resource lock does not authorize evaluator-only work")

    gates = config["gates"]
    config_ok = bool(
        config["modelCount"] == 1
        and config["horizonActions"] == 4
        and gates["requiredExactRootAction"] == "check_reference"
        and gates["requiredMAPRootAction"] == "check_target"
        and gates["requiredFinalControlActions"] == ["sample", "east"]
        and gates["minimumNormalizedMAPRegret"] == 0.02
        and gates["minimumNormalizedPosteriorSamplingRegret"] == 0.02
        and gates["minimumNormalizedExactOverOpenLoopAdvantage"] == 0.01
        and config["decisionRule"]["ifAnyGateFails"].startswith(
            "freeze a negative development result"
        )
        and not config["claimBoundary"]["externalConfirmationEvidence"]
    )
    if not config_ok:
        errors.append("V72 development gates or stop rule drifted")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "python",
            "-p",
            "test_v72_rocksample_source.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 10 tests" in combined
    if not tests_ok:
        errors.append(f"V72 source tests failed: {combined[-1600:]}")

    source = evaluator_path.read_text()
    source_checks = {
        "uses_locked_exact_planner": "from v71_exact_planning import" in source,
        "uses_locked_RockSample_exporter": "from v72_rocksample_source import" in source,
        "durable_attempt_before_candidate_evaluation": source.index(
            '(output_dir / "attempt.json").write_text'
        )
        < source.index("row = evaluate(config)"),
        "separate_development_output": "outputs/v72-active-sensing/development-evaluation"
        in source,
        "protected_firewall": "protected_confirmation_policy_value_count" in source,
        "EIG_firewall": "candidate_EIG_value_count" in source,
        "V71_protected_firewall": "V71_protected_access_count" in source,
        "stop_rule_present": "freeze_negative_result_and_stop_V72_before_protected_discovery"
        in source,
        "all_five_controls_present": all(
            name in source
            for name in (
                "plan_exact",
                "map_control",
                "posterior_sampling_control",
                "best_open_loop_sequence",
                "plan_myopic",
            )
        ),
    }
    if not all(source_checks.values()):
        errors.append("V72 development evaluator expands beyond the registered screen")

    outcome_absent = not (
        PROJECT_ROOT / "outputs/v72-active-sensing/development-evaluation"
    ).exists()
    if not outcome_absent:
        errors.append("V72 development outcome predates evaluator lock")

    checks = {
        "resource_lock_and_evaluator_only_authorization": authorization_ok,
        "fixed_noncompensatory_development_gates": config_ok,
        "ten_source_exporter_tests": tests_ok,
        "durable_one_model_development_evaluator": all(source_checks.values()),
        "development_outcome_absent_before_lock": outcome_absent,
    }
    audit = {
        "schema_version": "72-active-sensing-development-evaluation",
        "experiment": "v72_development_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_evaluator_and_authorize_one_development_run"
            if not errors
            else "reject_v72_development_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "structural_test_cases": 10,
            "development_models_evaluated": 0,
            "protected_confirmation_policy_value_count": 0,
            "candidate_EIG_value_count": 0,
            "V71_protected_access_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "72-active-sensing-development-evaluation",
        "experiment": "v72_rocksample_development_evaluator_lock",
        "resource_lock": str(resource_lock_path.relative_to(PROJECT_ROOT)),
        "resource_lock_sha256": file_sha256(resource_lock_path),
        "evaluation_config": str(config_path.relative_to(PROJECT_ROOT)),
        "evaluation_config_sha256": file_sha256(config_path),
        "evaluation_plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "evaluation_plan_sha256": file_sha256(plan_path),
        "exporter": str(exporter_path.relative_to(PROJECT_ROOT)),
        "exporter_sha256": file_sha256(exporter_path),
        "exporter_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "exporter_tests_sha256": file_sha256(tests_path),
        "planning_core": str(planning_path.relative_to(PROJECT_ROOT)),
        "planning_core_sha256": file_sha256(planning_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "evaluator_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "evaluator_auditor_sha256": file_sha256(auditor_path),
        "evaluator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluator_audit_sha256": file_sha256(audit_path),
        "attempt_path": "outputs/v72-active-sensing/development-evaluation/attempt.json",
        "result_path": "outputs/v72-active-sensing/development-evaluation/result.json",
        "expected_attempt_number": 1,
        "expected_model_count": 1,
        "expected_protected_confirmation_policy_value_count": 0,
        "authorization": {
            "modify_source_blueprint_exporter_evaluator_controls_or_gates": False,
            "run_development_outcomes_once": True,
            "inspect_or_select_protected_confirmation_sources": False,
            "compute_protected_confirmation_policy_values": False,
            "compute_candidate_EIG": False,
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
