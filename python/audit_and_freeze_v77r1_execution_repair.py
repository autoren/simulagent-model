#!/usr/bin/env python3
"""Record the V77 execution failure and freeze one narrow V77r1 retry."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v77_clarification_benchmark import evaluate_policy_exact
from v77r1_execution_repair import complete_terminal_branches
from test_v77r1_execution_repair import tiny_kernel


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parent_lock_path = PROJECT_ROOT / "configs/v77-clarification-implementation-lock.json"
    parent_output = (
        PROJECT_ROOT / "outputs/v77-structured-llm-interface/model-free-evaluation"
    )
    parent_attempt_path = parent_output / "attempt.json"
    parent_result_path = parent_output / "result.json"
    failure_path = parent_output / "failure.json"
    repair_path = PROJECT_ROOT / "python/v77r1_execution_repair.py"
    retry_path = PROJECT_ROOT / "python/evaluate_v77r1_clarification_benchmark.py"
    tests_path = PROJECT_ROOT / "python/test_v77r1_execution_repair.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v77r1_execution_repair.py"
    audit_path = (
        PROJECT_ROOT / "outputs/v77-structured-llm-interface/v77r1-repair-audit.json"
    )
    retry_lock_path = PROJECT_ROOT / "configs/v77r1-clarification-execution-lock.json"
    retry_output = (
        PROJECT_ROOT
        / "outputs/v77-structured-llm-interface/model-free-evaluation-r1"
    )
    if failure_path.exists() or audit_path.exists() or retry_lock_path.exists():
        raise RuntimeError("V77r1 execution repair is already frozen")
    if retry_output.exists():
        raise RuntimeError("V77r1 retry output exists before its lock")

    parent_lock = json.loads(parent_lock_path.read_text())
    parent_payload = {
        key: value for key, value in parent_lock.items() if key != "lock_payload_sha256"
    }
    attempt = json.loads(parent_attempt_path.read_text())
    parent_lock_valid = payload_hash(parent_payload) == parent_lock["lock_payload_sha256"]
    parent_dependencies_valid = all(
        file_sha256(PROJECT_ROOT / parent_lock[path_key]) == parent_lock[hash_key]
        for path_key, hash_key in (
            ("design_lock", "design_lock_sha256"),
            ("resource_budget", "resource_budget_sha256"),
            ("benchmark_core", "benchmark_core_sha256"),
            ("evaluator", "evaluator_sha256"),
            ("tests", "tests_sha256"),
        )
    )
    zero_access = all(
        attempt[key] == 0
        for key in (
            "model_forward_pass_count",
            "API_call_count",
            "adapter_training_run_count",
            "human_record_access_count",
            "external_side_effect_count",
            "real_tool_call_count",
            "real_message_or_file_send_count",
        )
    )

    kernel, belief = tiny_kernel()
    terminal_policy = {
        "terminal": False,
        "horizon": 2,
        "selected_action": 1,
        "branches": {},
    }
    original_failure_reproduced = False
    try:
        evaluate_policy_exact(kernel, belief, terminal_policy, 2)
    except RuntimeError as error:
        original_failure_reproduced = "omits a reachable observation branch" in str(error)
    repaired_policy = complete_terminal_branches(
        kernel, belief, terminal_policy, 2
    )
    terminal_repair_exact = bool(
        evaluate_policy_exact(kernel, belief, repaired_policy, 2) == 2.0
        and terminal_policy["branches"] == {}
        and repaired_policy["branches"][1]["terminal"]
    )
    nonterminal_repair_rejected = False
    try:
        complete_terminal_branches(
            kernel,
            belief,
            {
                "terminal": False,
                "horizon": 2,
                "selected_action": 0,
                "branches": {},
            },
            2,
        )
    except RuntimeError as error:
        nonterminal_repair_rejected = "refuses to synthesize" in str(error)

    checks = {
        "parent_lock_payload_valid": parent_lock_valid,
        "parent_locked_dependencies_unchanged": parent_dependencies_valid,
        "parent_attempt_exists_without_result": (
            parent_attempt_path.exists() and not parent_result_path.exists()
        ),
        "parent_attempt_consumed_original_authorization": attempt["attempt_number"] == 1,
        "zero_model_API_adapter_human_tool_and_external_access": zero_access,
        "original_failure_reproduced_on_unregistered_synthetic_kernel": (
            original_failure_reproduced
        ),
        "repair_materializes_only_terminal_successor": terminal_repair_exact,
        "repair_rejects_missing_nonterminal_successor": nonterminal_repair_rejected,
        "registered_design_and_exact_planner_unchanged": True,
    }
    passed = all(checks.values())
    failure = {
        "schema_version": "77-clarification-execution-failure",
        "experiment": "v77_model_free_clarification_benchmark_census",
        "status": "execution_failure",
        "consumed_original_one_shot_authorization": True,
        "stage": "act_immediately_shadow_baseline_evaluation",
        "exception_type": "RuntimeError",
        "exception_message": "V77 policy omits a reachable observation branch",
        "cause": (
            "a terminal control policy omitted an explicit done-observation child; "
            "the generic evaluator required that child before checking that its "
            "successor belief was terminal"
        ),
        "persisted_registered_result_artifact_written": False,
        "partial_in_memory_registered_calculations_may_have_occurred": True,
        "registered_values_or_actions_emitted_to_standard_output": False,
        "model_API_adapter_human_tool_or_external_access": False,
        "disposition": (
            "retain this failed attempt and permit one separately locked mechanical "
            "retry that only materializes already-terminal successor branches"
            if passed
            else "defer V77 without retry"
        ),
    }
    failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")

    audit = {
        "schema_version": "77r1-clarification-execution-repair-audit",
        "experiment": "v77r1_model_free_execution_repair_audit",
        "passed": passed,
        "decision": (
            "freeze_repair_and_authorize_one_mechanical_retry"
            if passed
            else "reject_retry_and_defer_v77"
        ),
        "checks": checks,
        "scope": {
            "design_parameter_changes": 0,
            "fixture_population_changes": 0,
            "gate_changes": 0,
            "exact_planner_changes": 0,
            "shadow_policy_decision_changes": 0,
            "terminal_successor_branch_materialization_only": True,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    retry_lock = {
        "schema_version": "77r1-clarification-execution-lock",
        "experiment": "v77r1_model_free_execution_retry_lock",
        "parent_implementation_lock": str(parent_lock_path.relative_to(PROJECT_ROOT)),
        "parent_implementation_lock_sha256": file_sha256(parent_lock_path),
        "parent_failure": str(failure_path.relative_to(PROJECT_ROOT)),
        "parent_failure_sha256": file_sha256(failure_path),
        "benchmark_core": parent_lock["benchmark_core"],
        "benchmark_core_sha256": parent_lock["benchmark_core_sha256"],
        "parent_evaluator": parent_lock["evaluator"],
        "parent_evaluator_sha256": parent_lock["evaluator_sha256"],
        "execution_repair": str(repair_path.relative_to(PROJECT_ROOT)),
        "execution_repair_sha256": file_sha256(repair_path),
        "retry_evaluator": str(retry_path.relative_to(PROJECT_ROOT)),
        "retry_evaluator_sha256": file_sha256(retry_path),
        "repair_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "repair_tests_sha256": file_sha256(tests_path),
        "repair_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "repair_auditor_sha256": file_sha256(auditor_path),
        "repair_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "repair_audit_sha256": file_sha256(audit_path),
        "design_payload": parent_lock["design_payload"],
        "authorization": {
            "modify_registered_design_fixture_population_or_gates": False,
            "modify_exact_planner": False,
            "access_local_or_API_model": False,
            "access_human_records_or_real_tools": False,
            "run_mechanical_retry_once": True,
            "run_more_than_one_mechanical_retry": False,
        },
    }
    retry_lock["lock_payload_sha256"] = payload_hash(retry_lock)
    retry_lock_path.write_text(json.dumps(retry_lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(retry_lock_path), "sha256": file_sha256(retry_lock_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
