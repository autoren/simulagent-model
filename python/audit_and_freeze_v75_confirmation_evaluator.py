#!/usr/bin/env python3
"""Audit and freeze the one-shot V75 replication evaluator."""
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
    structural_lock_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-structural-lock.json"
    config_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-evaluation.json"
    plan_path = PROJECT_ROOT / "docs/v75-active-sensing-confirmation-design-plan.md"
    exporter_path = PROJECT_ROOT / "python/v75_nova_paint_source.py"
    tests_path = PROJECT_ROOT / "python/test_v75_nova_paint_source.py"
    planning_path = PROJECT_ROOT / "python/v71_exact_planning.py"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v75_nova_paint_confirmation.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v75_confirmation_evaluator.py"
    audit_path = PROJECT_ROOT / "outputs/v75-active-sensing-confirmation/evaluator-audit.json"
    lock_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-evaluator-lock.json"
    if lock_path.exists():
        raise RuntimeError("V75 replication evaluator is already frozen")

    structural_lock = json.loads(structural_lock_path.read_text())
    structural_payload = {
        key: value for key, value in structural_lock.items() if key != "lock_payload_sha256"
    }
    config = json.loads(config_path.read_text())
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(structural_payload) == structural_lock["lock_payload_sha256"]
        and structural_lock["outcome"]["passed_all_structural_gates"]
        and structural_lock["authorization"]["implement_and_lock_replication_evaluator"]
        and not structural_lock["authorization"]["run_exact_replication"]
    )
    if not authorization_ok:
        errors.append("V75 structural lock does not authorize evaluator-only work")

    gates = config["gates"]
    config_ok = bool(
        config["modelCount"] == 1
        and config["horizonActions"] == 4
        and gates["requiredExactRootAction"] == "calibrate_beacon"
        and gates["requiredExactRootOptimalActions"]
        == ["calibrate_beacon", "inspect_target"]
        and gates["requiredMAPRootAction"] == "inspect_target"
        and gates["minimumNormalizedMAPRegret"] == 0.015
        and gates["minimumNormalizedPosteriorSamplingRegret"] == 0.015
        and gates["maximumEvaluationAttempts"] == 1
        and config["decisionRule"]["noRetrospectiveChanges"] is True
        and not config["claimBoundary"]["sourceDiscoveryCleanConfirmation"]
    )
    if not config_ok:
        errors.append("V75 replication gates or stop rule drifted")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "python",
            "-p",
            "test_v75_nova_paint_source.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 10 tests" in combined
    if not tests_ok:
        errors.append(f"V75 source tests failed: {combined[-1800:]}")

    evaluator_source = evaluator_path.read_text()
    source_checks = {
        "uses_locked_exact_planner": "from v71_exact_planning import" in evaluator_source,
        "uses_locked_paint_exporter": "from v75_nova_paint_source import" in evaluator_source,
        "durable_attempt_before_evaluation": evaluator_source.index(
            '(output_dir / "attempt.json").write_text'
        )
        < evaluator_source.index("row = evaluate(config)"),
        "separate_replication_output": "outputs/v75-active-sensing-confirmation/evaluation"
        in evaluator_source,
        "prior_outcome_firewall": "prior_paint_policy_outcome_access_count"
        in evaluator_source,
        "EIG_firewall": "candidate_EIG_value_count" in evaluator_source,
        "one_shot_negative_stop": "freeze_negative_replication_without_tuning_source_replacement_or_rerun"
        in evaluator_source,
        "all_five_controls_present": all(
            name in evaluator_source
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
        errors.append("V75 evaluator expands beyond the registered one-shot screen")

    outcome_absent = not (
        PROJECT_ROOT / "outputs/v75-active-sensing-confirmation/evaluation"
    ).exists()
    if not outcome_absent:
        errors.append("V75 replication outcome predates evaluator lock")

    checks = {
        "structural_lock_and_evaluator_only_authorization": authorization_ok,
        "fixed_noncompensatory_replication_gates": config_ok,
        "ten_source_exporter_tests": tests_ok,
        "durable_one_model_replication_evaluator": all(source_checks.values()),
        "replication_outcome_absent_before_lock": outcome_absent,
    }
    audit = {
        "schema_version": "75-active-sensing-confirmation-evaluation",
        "experiment": "v75_replication_evaluator_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_evaluator_and_authorize_one_replication_run"
            if not errors
            else "reject_v75_replication_evaluator"
        ),
        "errors": errors,
        "checks": checks,
        "source_checks": source_checks,
        "access": {
            "structural_test_cases": 10,
            "replication_models_evaluated": 0,
            "prior_paint_policy_outcome_access_count": 0,
            "candidate_EIG_value_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "75-active-sensing-confirmation-evaluation",
        "experiment": "v75_nova_paint_replication_evaluator_lock",
        "structural_lock": str(structural_lock_path.relative_to(PROJECT_ROOT)),
        "structural_lock_sha256": file_sha256(structural_lock_path),
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
        "attempt_path": "outputs/v75-active-sensing-confirmation/evaluation/attempt.json",
        "result_path": "outputs/v75-active-sensing-confirmation/evaluation/result.json",
        "expected_attempt_number": 1,
        "expected_model_count": 1,
        "authorization": {
            "modify_source_blueprint_exporter_evaluator_controls_or_gates": False,
            "run_replication_outcomes_once": True,
            "compute_candidate_EIG": False,
            "read_prior_paint_policy_outcomes": False,
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
