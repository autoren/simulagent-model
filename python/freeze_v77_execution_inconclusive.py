#!/usr/bin/env python3
"""Freeze V77/V77r1 as execution-inconclusive without another retry."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import evaluate_v77_clarification_benchmark as evaluator
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fake_fixture(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "structural": {
            "hypothesis_count": 5,
            "observation_normalization_rate": 1.0,
            "identical_hypothesis_support_rate": 1.0,
            "belief_normalizes": True,
        },
        "resource": {"belief_normalization_rate": 1.0},
        "exact": {
            "root_action": "ask_report",
            "reachable_information_actions": ["ask_report", "ask_recipient"],
            "safe_unknown_continuation_count": 1,
            "unknown_branch_irreversible_send_count": 0,
        },
        "map": {"normalized_regret": 1.0, "off_support_fallback_count": 0},
        "posterior_sampling": {
            "normalized_regret": 0.0,
            "off_support_fallback_count": 0,
        },
        "act_immediately": {"normalized_regret": 1.0},
        "ask_always": {"normalized_regret": 1.0},
    }


def main() -> None:
    parent_lock_path = PROJECT_ROOT / "configs/v77-clarification-implementation-lock.json"
    retry_lock_path = PROJECT_ROOT / "configs/v77r1-clarification-execution-lock.json"
    first_dir = PROJECT_ROOT / "outputs/v77-structured-llm-interface/model-free-evaluation"
    retry_dir = PROJECT_ROOT / "outputs/v77-structured-llm-interface/model-free-evaluation-r1"
    retry_failure_path = retry_dir / "failure.json"
    closure_path = PROJECT_ROOT / "outputs/v77-structured-llm-interface/execution-closure.json"
    closure_lock_path = PROJECT_ROOT / "configs/v77-execution-closure-lock.json"
    auditor_path = PROJECT_ROOT / "python/freeze_v77_execution_inconclusive.py"
    if retry_failure_path.exists() or closure_path.exists() or closure_lock_path.exists():
        raise RuntimeError("V77 execution closure is already frozen")

    parent_lock = json.loads(parent_lock_path.read_text())
    retry_lock = json.loads(retry_lock_path.read_text())
    retry_payload = {
        key: value for key, value in retry_lock.items() if key != "lock_payload_sha256"
    }
    attempts = [
        json.loads((first_dir / "attempt.json").read_text()),
        json.loads((retry_dir / "attempt.json").read_text()),
    ]
    zero_access = all(
        row[key] == 0
        for row in attempts
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
    config = retry_lock["design_payload"]
    fake = {row["name"]: _fake_fixture(row["name"]) for row in config["fixtures"]}
    aggregation_failure_reproduced = False
    try:
        evaluator.evaluate_gates(fake, config, attempts[-1])
    except KeyError as error:
        aggregation_failure_reproduced = error.args == ("name",)

    checks = {
        "retry_lock_payload_valid": payload_hash(retry_payload)
        == retry_lock["lock_payload_sha256"],
        "parent_core_still_matches_both_locks": bool(
            file_sha256(PROJECT_ROOT / parent_lock["benchmark_core"])
            == parent_lock["benchmark_core_sha256"]
            == retry_lock["benchmark_core_sha256"]
        ),
        "parent_evaluator_still_matches_both_locks": bool(
            file_sha256(PROJECT_ROOT / parent_lock["evaluator"])
            == parent_lock["evaluator_sha256"]
            == retry_lock["parent_evaluator_sha256"]
        ),
        "both_attempts_exist_without_result_artifacts": bool(
            (first_dir / "attempt.json").exists()
            and (retry_dir / "attempt.json").exists()
            and not (first_dir / "result.json").exists()
            and not (retry_dir / "result.json").exists()
        ),
        "gate_aggregation_failure_reproduced_without_registered_planning": (
            aggregation_failure_reproduced
        ),
        "zero_model_API_adapter_human_tool_and_external_access": zero_access,
        "retry_limit_is_explicitly_exhausted": bool(
            attempts[-1]["execution_retry_number"] == 1
            and not retry_lock["authorization"]["run_more_than_one_mechanical_retry"]
        ),
    }
    passed = all(checks.values())
    retry_failure = {
        "schema_version": "77r1-clarification-execution-failure",
        "experiment": "v77r1_model_free_clarification_benchmark_census",
        "status": "execution_failure",
        "consumed_only_authorized_mechanical_retry": True,
        "stage": "gate_aggregation_after_fixture_evaluation",
        "exception_type": "KeyError",
        "exception_message": "name",
        "cause": (
            "the frozen gate aggregator extracted structural subrecords, then "
            "attempted to use a fixture-name field that those subrecords do not carry"
        ),
        "persisted_registered_result_artifact_written": False,
        "partial_in_memory_registered_calculations_occurred": True,
        "registered_values_or_actions_emitted_to_standard_output": False,
        "model_API_adapter_human_tool_or_external_access": False,
        "disposition": "no further V77 retry; close as execution-inconclusive",
    }
    retry_failure_path.write_text(
        json.dumps(retry_failure, indent=2, sort_keys=True) + "\n"
    )
    closure = {
        "schema_version": "77-clarification-execution-closure",
        "experiment": "v77_clarification_execution_closure",
        "status": "execution_inconclusive",
        "scientific_hypothesis_supported": False,
        "scientific_hypothesis_refuted": False,
        "registered_outcome_available": False,
        "checks": checks,
        "passed_as_execution_audit": passed,
        "decision": (
            "freeze_v77_without_further_retry_and_require_full_path_harness_tests_before_a_fresh_successor"
            if passed
            else "defer_clarification_benchmark_branch"
        ),
        "attempts": [
            {
                "attempt_number": 1,
                "status": "execution_failure",
                "failure": str((first_dir / "failure.json").relative_to(PROJECT_ROOT)),
            },
            {
                "attempt_number": 2,
                "status": "execution_failure",
                "failure": str(retry_failure_path.relative_to(PROJECT_ROOT)),
            },
        ],
        "authorization": {
            "rerun_v77_or_v77r1": False,
            "claim_v77_scientific_outcome": False,
            "access_local_or_API_model": False,
            "start_outcome_blind_successor_harness_hardening": passed,
            "preregister_fresh_successor_after_harness_passes": passed,
        },
    }
    closure_path.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(closure, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "77-clarification-execution-closure-lock",
        "experiment": "v77_clarification_execution_closure_lock",
        "parent_implementation_lock": str(parent_lock_path.relative_to(PROJECT_ROOT)),
        "parent_implementation_lock_sha256": file_sha256(parent_lock_path),
        "retry_lock": str(retry_lock_path.relative_to(PROJECT_ROOT)),
        "retry_lock_sha256": file_sha256(retry_lock_path),
        "first_failure": str((first_dir / "failure.json").relative_to(PROJECT_ROOT)),
        "first_failure_sha256": file_sha256(first_dir / "failure.json"),
        "retry_failure": str(retry_failure_path.relative_to(PROJECT_ROOT)),
        "retry_failure_sha256": file_sha256(retry_failure_path),
        "closure": str(closure_path.relative_to(PROJECT_ROOT)),
        "closure_sha256": file_sha256(closure_path),
        "closure_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "closure_auditor_sha256": file_sha256(auditor_path),
        "authorization": closure["authorization"],
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    closure_lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(closure, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(closure_lock_path), "sha256": file_sha256(closure_lock_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
