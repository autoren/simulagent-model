#!/usr/bin/env python3
"""Audit the V72 RockSample exporter and freeze its resource envelope."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v72_rocksample_source import build_family, structural_resource_metrics


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    source_lock_path = PROJECT_ROOT / "configs/v72-active-sensing-external-source-lock.json"
    budget_path = (
        PROJECT_ROOT / "configs/v72-active-sensing-development-resource-budget.json"
    )
    plan_path = PROJECT_ROOT / "docs/v72-active-sensing-development-resource-plan.md"
    exporter_path = PROJECT_ROOT / "python/v72_rocksample_source.py"
    tests_path = PROJECT_ROOT / "python/test_v72_rocksample_source.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v72_development_resource.py"
    audit_path = PROJECT_ROOT / "outputs/v72-active-sensing/development-resource-audit.json"
    lock_path = PROJECT_ROOT / "configs/v72-active-sensing-development-resource-lock.json"
    if lock_path.exists():
        raise RuntimeError("V72 development resource envelope is already frozen")

    source_lock = json.loads(source_lock_path.read_text())
    source_payload = {
        key: value for key, value in source_lock.items() if key != "lock_payload_sha256"
    }
    budget = json.loads(budget_path.read_text())
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(source_payload) == source_lock["lock_payload_sha256"]
        and source_lock["authorization"]["implement_and_test_deterministic_source_exporter"]
        and source_lock["authorization"]["run_structural_resource_census"]
        and not source_lock["authorization"][
            "compute_candidate_policy_values_actions_regrets_or_EIG"
        ]
        and source_lock["selected_development_candidate"]
        == "rocksample_jl_configurable_small"
    )
    if not authorization_ok:
        errors.append("V72 source lock does not authorize exporter/resource work")

    selected_source = (
        PROJECT_ROOT
        / "data/v72-active-sensing/source-checkouts/RockSample.jl/src/RockSample.jl"
    )
    inventory = json.loads((PROJECT_ROOT / source_lock["inventory"]).read_text())
    source_integrity_ok = bool(
        file_sha256(PROJECT_ROOT / source_lock["inventory"])
        == source_lock["inventory_sha256"]
        and file_sha256(selected_source)
        == inventory["source_file_sha256"]["RockSample.jl/src/RockSample.jl"]
        and source_lock["repository_commits"][
            "https://github.com/JuliaPOMDP/RockSample.jl"
        ]
        == "c8b3566d30c5dd7be6c7790b4b9a54ebfcdeecde"
    )
    if not source_integrity_ok:
        errors.append("V72 selected source or inventory drifted")

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
        errors.append(f"V72 RockSample structural tests failed: {combined[-1600:]}")

    family = build_family()
    metrics = structural_resource_metrics(family, horizon=4)
    resource_ok = bool(
        metrics["states"] <= budget["maximumStates"]
        and metrics["actions"] <= budget["maximumActions"]
        and metrics["observations"] <= budget["maximumObservations"]
        and metrics["dense_kernel_bytes"] <= budget["maximumDenseKernelBytes"]
        and metrics["exact_bellman_node_upper_bound"]
        <= budget["maximumExactBellmanNodesAtLockedHorizon"]
    )
    if not resource_ok:
        errors.append("V72 RockSample blueprint exceeds its resource envelope")

    exporter_source = exporter_path.read_text()
    outcome_firewall_ok = bool(
        "plan_exact" not in exporter_source
        and "map_control" not in exporter_source
        and "posterior_sampling_control" not in exporter_source
        and "best_open_loop_sequence" not in exporter_source
        and "plan_myopic" not in exporter_source
    )
    if not outcome_firewall_ok:
        errors.append("V72 exporter imported or called candidate planning code")

    access = {
        "implementation_structural_test_invocations": 3,
        "failed_structural_test_invocations": 2,
        "failed_test_causes": [
            "observation rows were initially undefined for unreachable action-successor pairs",
            "full-support assertion initially included the absorbing terminal state's none-only observation",
        ],
        "candidate_simulator_runs": 0,
        "candidate_policy_values_computed": 0,
        "candidate_optimal_actions_computed": 0,
        "candidate_regrets_computed": 0,
        "candidate_EIG_values_computed": 0,
        "V71_protected_access_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    checks = {
        "source_lock_authorizes_exporter_and_resource_only": authorization_ok,
        "selected_source_and_inventory_integrity": source_integrity_ok,
        "ten_structural_source_export_tests": tests_ok,
        "resource_envelope_passes": resource_ok,
        "zero_candidate_planning_or_outcome_code": outcome_firewall_ok,
    }
    audit = {
        "schema_version": "72-active-sensing-development-resource",
        "experiment": "v72_rocksample_development_resource_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_exporter_and_resource_envelope_then_authorize_evaluator_implementation"
            if not errors
            else "defer_v72_development_before_candidate_outcomes"
        ),
        "errors": errors,
        "checks": checks,
        "metrics": metrics,
        "access": access,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "72-active-sensing-development-resource",
        "experiment": "v72_rocksample_development_resource_lock",
        "source_lock": str(source_lock_path.relative_to(PROJECT_ROOT)),
        "source_lock_sha256": file_sha256(source_lock_path),
        "budget": str(budget_path.relative_to(PROJECT_ROOT)),
        "budget_sha256": file_sha256(budget_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "exporter": str(exporter_path.relative_to(PROJECT_ROOT)),
        "exporter_sha256": file_sha256(exporter_path),
        "exporter_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "exporter_tests_sha256": file_sha256(tests_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "metrics": metrics,
        "expected_development_model_count": 1,
        "expected_horizon": 4,
        "authorization": {
            "modify_source_inventory_blueprint_exporter_or_resource_budget": False,
            "write_and_audit_development_evaluator": True,
            "run_development_outcomes": False,
            "compute_candidate_policy_values_actions_regrets_or_EIG": False,
            "select_or_inspect_protected_confirmation_models": False,
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
