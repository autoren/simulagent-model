#!/usr/bin/env python3
"""Audit and freeze V74's source-level economic screen before adapter code."""
from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v74-active-sensing-economic-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v74-active-sensing-economic-feasibility-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v74_economic_feasibility.py"
    prior_lock_path = (
        PROJECT_ROOT / "configs/v73-active-sensing-structural-outcome-lock.json"
    )
    checkout = PROJECT_ROOT / "data/v74-active-sensing/source-checkouts/pomdp-py"
    tiger_path = checkout / "pomdp_py/problems/tiger/tiger_problem.py"
    value_path = checkout / "pomdp_py/algorithms/value_function.py"
    conversion_path = checkout / "tests/test_conversion_pomdp-solve.py"
    license_path = checkout / "LICENSE"
    inventory_path = PROJECT_ROOT / "outputs/v74-active-sensing/source-economic-inventory.json"
    audit_path = PROJECT_ROOT / "outputs/v74-active-sensing/source-economic-audit.json"
    lock_path = PROJECT_ROOT / "configs/v74-active-sensing-economic-lock.json"
    if lock_path.exists():
        raise RuntimeError("V74 economic feasibility is already frozen")

    forbidden_future_paths = (
        PROJECT_ROOT / "python/v74_pomdppy_tiger_source.py",
        PROJECT_ROOT / "python/test_v74_pomdppy_tiger_source.py",
        PROJECT_ROOT / "python/evaluate_v74_pomdppy_tiger_development.py",
    )
    config = json.loads(config_path.read_text())
    prior_lock = json.loads(prior_lock_path.read_text())
    prior_payload = {
        key: value for key, value in prior_lock.items() if key != "lock_payload_sha256"
    }
    errors: list[str] = []
    prior_authorization_ok = bool(
        payload_hash(prior_payload) == prior_lock["lock_payload_sha256"]
        and not prior_lock["outcome"]["passed_all_structural_gates"]
        and prior_lock["authorization"][
            "design_materially_new_successor_after_fresh_preregistration"
        ]
        and not prior_lock["authorization"]["modify_or_rerun_V73"]
        and not prior_lock["authorization"][
            "reuse_V73_source_component_adapter_or_parameters_for_successor_outcomes"
        ]
    )
    if not prior_authorization_ok:
        errors.append("V73 outcome lock does not authorize the fresh V74 successor")

    preimplementation_ok = not any(path.exists() for path in forbidden_future_paths)
    if not preimplementation_ok:
        errors.append("V74 economic screen was not run before adapter implementation")

    actual_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    selected = config["selectedDevelopmentSource"]
    commit_ok = actual_commit == selected["commit"]
    if not commit_ok:
        errors.append("V74 pomdp-py checkout commit drifted")

    tiger_source = tiger_path.read_text()
    value_source = value_path.read_text()
    conversion_source = conversion_path.read_text()
    license_text = license_path.read_text()
    source_structure_ok = all(
        fragment in tiger_source
        for fragment in (
            'States: tiger-left, tiger-right',
            'Actions: open-left, open-right, listen',
            '+10 for opening treasure door. -100 for opening tiger door.',
            '-1 for listening.',
            'def __init__(self, noise=0.15):',
            'return 1.0 - self.noise',
            'return 0.5',
            'return 1.0 - 1e-9',
            'return 1e-9',
        )
    ) and "The MIT License (MIT)" in license_text
    discount_ok = "gamma = 0.95" in value_source and "discount_factor=0.95" in conversion_source
    if not source_structure_ok or not discount_ok:
        errors.append("V74 Tiger source, license, or discount declarations drifted")

    source = config["sourceGrounding"]
    layer = config["prospectiveProjectLayer"]
    gates = config["economicGates"]
    p = float(source["configuredObservationAccuracy"])
    discount = float(source["discount"])
    safe_reward = float(source["safeOpenReward"])
    tiger_reward = float(source["tigerOpenReward"])
    listen_reward = float(source["targetListenReward"])
    beacon_reward = float(layer["beaconReward"])
    paired_correct = p * p + (1.0 - p) * (1.0 - p)
    safe_open_threshold = (-tiger_reward) / (safe_reward - tiger_reward)
    expected_open = paired_correct * safe_reward + (1.0 - paired_correct) * tiger_reward
    fixed_value = beacon_reward + discount * listen_reward + discount**2 * expected_open
    open_loop_value = beacon_reward * (1.0 + discount + discount**2)
    raw_advantage = fixed_value - open_loop_value
    return_scale = (safe_reward - tiger_reward) * (1.0 + discount + discount**2)
    normalized_advantage = raw_advantage / return_scale
    normalized_margin = (
        normalized_advantage
        - float(gates["minimumNormalizedFixedPolicyOverBestOpenLoopAdvantage"])
    )

    checks = {
        "v73_authorizes_fresh_successor": prior_authorization_ok,
        "economic_screen_precedes_adapter": preimplementation_ok,
        "pinned_commit_matches": commit_ok,
        "source_structure_license_and_discount_match": source_structure_ok and discount_ok,
        "source_is_fresh_relative_to_v72_and_v73": bool(selected["freshRelativeToV72AndV73"]),
        "configured_accuracy_passes": p >= float(gates["minimumObservationAccuracy"]),
        "paired_correct_probability_crosses_safe_open_threshold": paired_correct > safe_open_threshold,
        "fixed_policy_value_strictly_positive": fixed_value > 0.0,
        "raw_economic_advantage_passes": raw_advantage >= float(gates["minimumRawFixedPolicyOverBestOpenLoopAdvantage"]),
        "normalized_economic_advantage_passes": normalized_advantage >= float(gates["minimumNormalizedFixedPolicyOverBestOpenLoopAdvantage"]),
        "normalized_margin_passes": normalized_margin >= float(gates["minimumNormalizedMarginAboveThreshold"]),
        "beacon_is_nonharvestable": layer["beaconHarvestable"] is gates["requiredCalibrationBeaconHarvestable"],
        "open_loop_comparator_is_repeated_beacon": gates["requiredOpenLoopOptimalAction"] == "calibrate-beacon",
        "adapter_and_optimal_planner_calls_are_zero": gates["maximumAdapterImplementationCount"] == 0 and gates["maximumOptimalPlannerCalls"] == 0,
        "protected_source_access_is_zero": gates["maximumProtectedSourceAccessCount"] == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    access = {
        "repository_landing_pages_previewed": 1,
        "pinned_repositories_cloned": 1,
        "source_files_inspected": 4,
        "closed_form_economic_screen_count": 1,
        "adapter_implementation_count": 0,
        "candidate_simulator_runs": 0,
        "optimal_planner_calls": 0,
        "candidate_regrets_computed": 0,
        "protected_source_access_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    inventory = {
        "schema_version": "74-active-sensing-economic-feasibility",
        "experiment": "v74_pomdppy_tiger_source_economic_inventory",
        "repository": selected["repository"],
        "commit": actual_commit,
        "license": selected["license"],
        "source_files": {
            str(path.relative_to(checkout)): file_sha256(path)
            for path in (tiger_path, value_path, conversion_path, license_path)
        },
        "frozen_source_grounding": source,
        "frozen_project_layer": layer,
        "access": access,
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    audit = {
        "schema_version": "74-active-sensing-economic-feasibility",
        "experiment": "v74_pomdppy_tiger_prospective_economic_screen",
        "passed": not errors and not failed,
        "decision": (
            "freeze_economic_pass_and_authorize_adapter_preregistration"
            if not errors and not failed
            else "freeze_negative_economic_result_and_stop_before_adapter"
        ),
        "errors": errors,
        "failed_gates": failed,
        "checks": checks,
        "metrics": {
            "observation_accuracy": p,
            "paired_correct_probability": paired_correct,
            "safe_open_probability_threshold": safe_open_threshold,
            "expected_terminal_open_reward": expected_open,
            "fixed_economic_policy_value": fixed_value,
            "best_open_loop_value": open_loop_value,
            "raw_fixed_policy_over_open_loop_advantage": raw_advantage,
            "return_scale": return_scale,
            "normalized_fixed_policy_over_open_loop_advantage": normalized_advantage,
            "normalized_margin_above_threshold": normalized_margin,
        },
        "access": access,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lock = {
        "schema_version": "74-active-sensing-economic-feasibility",
        "experiment": "v74_pomdppy_tiger_economic_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "prior_lock": str(prior_lock_path.relative_to(PROJECT_ROOT)),
        "prior_lock_sha256": file_sha256(prior_lock_path),
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "repository_commit": actual_commit,
        "outcome": {
            "passed_all_economic_gates": audit["passed"],
            "failed_gate_count": len(failed) + len(errors),
            "fixed_policy_value": fixed_value,
            "best_open_loop_value": open_loop_value,
            "normalized_advantage": normalized_advantage,
            "normalized_margin": normalized_margin,
        },
        "authorization": {
            "modify_source_economic_blueprint_or_gates": False,
            "preregister_adapter_structural_and_development_design": audit["passed"],
            "implement_adapter_and_structural_tests": False,
            "run_optimal_planners": False,
            "select_or_inspect_confirmation_sources": False,
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
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
