#!/usr/bin/env python3
"""Run and freeze the single authorized V74 adapter structural audit."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v71_exact_planning import (
    best_open_loop_sequence,
    evaluate_policy_exact,
    finite_horizon_return_scale,
)
from v74_pomdppy_tiger_source import (
    ACTION_NAMES,
    build_family,
    fixed_structural_policy,
    structural_diagnostics,
    structural_resource_metrics,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_lock_path = PROJECT_ROOT / "configs/v74-active-sensing-design-lock.json"
    config_path = PROJECT_ROOT / "configs/v74-active-sensing-structural-design.json"
    exporter_path = PROJECT_ROOT / "python/v74_pomdppy_tiger_source.py"
    tests_path = PROJECT_ROOT / "python/test_v74_pomdppy_tiger_source.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v74_structural_feasibility.py"
    attempt_path = PROJECT_ROOT / "outputs/v74-active-sensing/structural-attempt.json"
    audit_path = PROJECT_ROOT / "outputs/v74-active-sensing/structural-audit.json"
    lock_path = PROJECT_ROOT / "configs/v74-active-sensing-structural-lock.json"
    if attempt_path.exists() or lock_path.exists():
        raise RuntimeError("V74 structural audit already attempted")

    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(
        json.dumps(
            {
                "schema_version": "74-active-sensing-structural-attempt",
                "experiment": "v74_tiger_structural_audit",
                "attempt_number": 1,
                "maximum_attempts": 1,
                "status": "claimed_before_structural_values",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    design_lock = json.loads(design_lock_path.read_text())
    design_payload = {
        key: value for key, value in design_lock.items() if key != "lock_payload_sha256"
    }
    config = json.loads(config_path.read_text())
    gates = config["structuralGates"]
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(design_payload) == design_lock["lock_payload_sha256"]
        and design_lock["authorization"]["implement_and_test_source_grounded_adapter"]
        and design_lock["authorization"]["run_structural_audit_once"]
        and not design_lock["authorization"]["run_optimal_planner_outcomes"]
    )
    if not authorization_ok:
        errors.append("V74 design lock does not authorize structural audit")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "python",
            "-p",
            "test_v74_pomdppy_tiger_source.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 10 tests" in combined
    if not tests_ok:
        errors.append(f"V74 structural tests failed: {combined[-1600:]}")

    family = build_family()
    horizon = int(config["horizonActions"])
    metrics = structural_resource_metrics(horizon)
    diagnostics = structural_diagnostics()
    fixed_value = evaluate_policy_exact(
        family.kernel, family.initial_belief, fixed_structural_policy(horizon), horizon
    )
    open_loop = best_open_loop_sequence(
        family.kernel,
        family.initial_belief,
        horizon,
        tie_tolerance=float(config["tieTolerance"]),
    )
    scale = finite_horizon_return_scale(family.kernel, horizon)
    raw_advantage = fixed_value - float(open_loop["value"])
    normalized_advantage = raw_advantage / scale
    normalized_margin = (
        normalized_advantage
        - float(gates["minimumFixedAdaptivePolicyOverBestOpenLoopNormalizedAdvantage"])
    )

    checks = {
        "design_lock_authorizes_structural_only": authorization_ok,
        "required_structural_unit_tests_pass": tests_ok,
        "resource_envelope_passes": bool(
            metrics["states"] <= gates["maximumStates"]
            and metrics["actions"] <= gates["maximumActions"]
            and metrics["observations"] <= gates["maximumObservations"]
            and metrics["dense_kernel_bytes"] <= gates["maximumDenseKernelBytes"]
            and metrics["exact_bellman_node_upper_bound"]
            <= gates["maximumExactBellmanNodeUpperBound"]
        ),
        "initial_unconditioned_route_is_nonrewarding_beacon": bool(
            diagnostics["initial_best_unconditioned_action"]
            == gates["requiredInitialBestUnconditionedAction"]
            and diagnostics["initial_best_unconditioned_expected_reward"]
            <= gates["maximumInitialBestUnconditionedExpectedReward"]
        ),
        "calibration_beacon_is_nonharvestable": diagnostics[
            "calibration_beacon_harvestable"
        ]
        is gates["requiredCalibrationBeaconHarvestable"],
        "calibration_uses_source_listen_transition": bool(
            diagnostics["calibration_transition_matches_source_listen"]
        ),
        "point_model_supports_identical": diagnostics[
            "point_model_supports_identical"
        ]
        is gates["requiredPointModelSupportIdentity"],
        "calibration_mutual_information_passes": diagnostics[
            "calibration_mutual_information_nats"
        ]
        >= gates["minimumCalibrationMutualInformationNats"],
        "target_listen_TV_passes": diagnostics["target_listen_total_variation"]
        >= gates["minimumTargetListenTotalVariation"],
        "paired_decision_correct_probability_passes": diagnostics[
            "paired_decision_correct_probability"
        ]
        >= gates["minimumPairedDecisionCorrectProbability"],
        "fixed_policy_raw_advantage_passes": raw_advantage
        >= gates["minimumFixedAdaptivePolicyOverBestOpenLoopRawAdvantage"],
        "fixed_policy_normalized_advantage_passes": normalized_advantage
        >= gates["minimumFixedAdaptivePolicyOverBestOpenLoopNormalizedAdvantage"],
        "normalized_margin_passes": normalized_margin
        >= gates["minimumNormalizedMarginAboveEconomicThreshold"],
        "forbidden_optimal_planner_calls_are_zero": bool(
            gates["maximumExactBayesAdaptiveCalls"] == 0
            and gates["maximumMAPCalls"] == 0
            and gates["maximumPosteriorSamplingCalls"] == 0
            and gates["maximumMyopicCalls"] == 0
        ),
        "protected_source_access_is_zero": gates[
            "maximumProtectedSourceAccessCount"
        ]
        == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    access = {
        "structural_audit_attempts": 1,
        "implementation_structural_test_invocations": 1,
        "fixed_policy_value_count": 1,
        "open_loop_sequence_values_computed": int(open_loop["sequence_count"]),
        "exact_Bayes_adaptive_calls": 0,
        "MAP_calls": 0,
        "posterior_sampling_calls": 0,
        "myopic_calls": 0,
        "candidate_EIG_values_computed": 0,
        "protected_source_access_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    audit = {
        "schema_version": "74-active-sensing-structural-feasibility",
        "experiment": "v74_pomdppy_tiger_structural_audit",
        "passed": not errors and not failed,
        "decision": (
            "freeze_structural_pass_and_authorize_evaluator_implementation"
            if not errors and not failed
            else "freeze_negative_structural_result_and_stop_before_planner_outcomes"
        ),
        "errors": errors,
        "failed_gates": failed,
        "checks": checks,
        "metrics": metrics,
        "structural_diagnostics": diagnostics,
        "fixed_adaptive_policy_value": fixed_value,
        "best_open_loop_value": float(open_loop["value"]),
        "best_open_loop_actions": [
            ACTION_NAMES[int(action)] for action in open_loop["selected_actions"]
        ],
        "raw_fixed_policy_over_open_loop_advantage": raw_advantage,
        "return_scale": scale,
        "normalized_fixed_policy_over_open_loop_advantage": normalized_advantage,
        "normalized_margin_above_threshold": normalized_margin,
        "access": access,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    lock = {
        "schema_version": "74-active-sensing-structural-feasibility",
        "experiment": "v74_pomdppy_tiger_structural_lock",
        "design_lock": str(design_lock_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_lock_path),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "exporter": str(exporter_path.relative_to(PROJECT_ROOT)),
        "exporter_sha256": file_sha256(exporter_path),
        "exporter_tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "exporter_tests_sha256": file_sha256(tests_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "outcome": {
            "passed_all_structural_gates": audit["passed"],
            "failed_gate_count": len(failed) + len(errors),
            "fixed_policy_value": fixed_value,
            "best_open_loop_value": float(open_loop["value"]),
            "normalized_advantage": normalized_advantage,
            "point_model_supports_identical": diagnostics[
                "point_model_supports_identical"
            ],
        },
        "authorization": {
            "modify_source_adapter_structural_or_evaluation_design": False,
            "implement_and_lock_development_evaluator": audit["passed"],
            "run_exact_development_evaluation": False,
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
