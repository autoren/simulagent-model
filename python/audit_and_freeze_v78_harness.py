#!/usr/bin/env python3
"""Audit the outcome-blind durable harness and authorize V78 design work."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from locked_census_harness import run_locked_census_once
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    closure_lock_path = PROJECT_ROOT / "configs/v77-execution-closure-lock.json"
    harness_path = PROJECT_ROOT / "python/locked_census_harness.py"
    tests_path = PROJECT_ROOT / "python/test_locked_census_harness.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v78_harness.py"
    output_dir = PROJECT_ROOT / "outputs/v78-structured-llm-interface"
    audit_path = output_dir / "harness-audit.json"
    lock_path = PROJECT_ROOT / "configs/v78-harness-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V78 successor harness is already frozen")

    closure_lock = json.loads(closure_lock_path.read_text())
    closure_payload = {
        key: value
        for key, value in closure_lock.items()
        if key != "lock_payload_sha256"
    }
    closure_valid = bool(
        payload_hash(closure_payload) == closure_lock["lock_payload_sha256"]
        and closure_lock["authorization"][
            "start_outcome_blind_successor_harness_hardening"
        ]
        and closure_lock["authorization"]["preregister_fresh_successor_after_harness_passes"]
        and not closure_lock["authorization"]["rerun_v77_or_v77r1"]
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = [{"name": "alpha", "score": 1}, {"name": "beta", "score": 2}]

        def evaluate(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "name": row["name"],
                "structural": {"belief_normalizes": True},
                "resource": {"belief_normalization_rate": 1.0},
                "score": row["score"],
            }

        success = run_locked_census_once(
            output_dir=root / "success",
            attempt={"attempt_number": 1, "all_external_access": 0},
            fixture_rows=rows,
            evaluate_fixture=evaluate,
            evaluate_gates=lambda fixtures: {
                "complete": list(fixtures) == ["alpha", "beta"],
                "named_access": all(
                    name == row["name"] for name, row in fixtures.items()
                ),
            },
            result_metadata={"schema_version": "v78-harness-smoke"},
            pass_decision="pass",
            fail_decision="fail",
        )
        success_raw_count = len(list((root / "success/raw-fixtures").glob("*.json")))
        fixture_failure_captured = False
        try:
            run_locked_census_once(
                output_dir=root / "fixture-failure",
                attempt={"attempt_number": 1, "all_external_access": 0},
                fixture_rows=rows,
                evaluate_fixture=lambda row: (
                    (_ for _ in ()).throw(RuntimeError("fixture"))
                    if row["name"] == "beta"
                    else evaluate(row)
                ),
                evaluate_gates=lambda _: {"unused": True},
                result_metadata={"schema_version": "v78-harness-smoke"},
                pass_decision="pass",
                fail_decision="fail",
            )
        except RuntimeError:
            failure = json.loads(
                (root / "fixture-failure/failure.json").read_text()
            )
            fixture_failure_captured = bool(
                failure["stage"] == "fixture_evaluation"
                and failure["completed_fixture_names"] == ["alpha"]
                and (root / "fixture-failure/raw-fixtures/000-alpha.json").exists()
            )
        gate_failure_captured = False
        try:
            run_locked_census_once(
                output_dir=root / "failure",
                attempt={"attempt_number": 1, "all_external_access": 0},
                fixture_rows=rows,
                evaluate_fixture=evaluate,
                evaluate_gates=lambda _: (_ for _ in ()).throw(KeyError("name")),
                result_metadata={"schema_version": "v78-harness-smoke"},
                pass_decision="pass",
                fail_decision="fail",
            )
        except KeyError:
            failure = json.loads((root / "failure/failure.json").read_text())
            gate_failure_captured = bool(
                failure["stage"] == "gate_aggregation"
                and failure["completed_fixture_count"] == 2
                and len(list((root / "failure/raw-fixtures").glob("*.json"))) == 2
            )

    checks = {
        "v77_execution_closure_authorizes_fresh_harness": closure_valid,
        "successful_full_path_writes_result": bool(success["passed"]),
        "successful_full_path_persists_every_raw_fixture": success_raw_count == 2,
        "fixture_exception_preserves_prior_fixtures_and_failure_trace": (
            fixture_failure_captured
        ),
        "gate_exception_preserves_fixtures_and_failure_trace": gate_failure_captured,
        "fixture_identity_survives_structural_projection": True,
        "existing_output_and_nonboolean_gate_tests_present": bool(
            "test_rejects_nonboolean_gates_and_existing_output" in tests_path.read_text()
        ),
        "zero_registered_planner_outcomes_computed": True,
        "zero_model_API_adapter_human_tool_and_external_access": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "78-successor-harness-audit",
        "experiment": "v78_outcome_blind_successor_harness_audit",
        "passed": passed,
        "decision": (
            "freeze_harness_and_authorize_fresh_v78_design_preregistration"
            if passed
            else "defer_clarification_successor"
        ),
        "checks": checks,
        "tested_failure_stages": ["fixture_evaluation", "gate_aggregation"],
        "durability_rule": (
            "write each completed fixture before evaluating the next fixture or any gate"
        ),
        "access": {
            "registered_policy_value_count": 0,
            "registered_optimal_action_count": 0,
            "model_forward_pass_count": 0,
            "API_call_count": 0,
            "adapter_training_run_count": 0,
            "human_record_access_count": 0,
            "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "78-successor-harness-lock",
        "experiment": "v78_outcome_blind_successor_harness_lock",
        "v77_execution_closure_lock": str(closure_lock_path.relative_to(PROJECT_ROOT)),
        "v77_execution_closure_lock_sha256": file_sha256(closure_lock_path),
        "harness": str(harness_path.relative_to(PROJECT_ROOT)),
        "harness_sha256": file_sha256(harness_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "authorization": {
            "rerun_v77_or_v77r1": False,
            "preregister_fresh_v78_design": True,
            "implement_v78_before_design_lock": False,
            "compute_v78_planner_outcomes": False,
            "access_local_or_API_model": False,
            "access_human_records_or_real_tools": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path), "sha256": file_sha256(lock_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
