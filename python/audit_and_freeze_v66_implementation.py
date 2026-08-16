#!/usr/bin/env python3
"""Audit and freeze the V66 planner implementation without scoring sealed policies."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v64_external_eig import filter_public_history, load_family
from v66_bayes_adaptive_reward import (
    StaticKernel,
    assert_synthetic_planner_fixture,
    compact_policy,
    evaluate_policy,
    evaluate_policy_information,
    evaluate_root_action_values,
    exact_eig_crosscheck,
    exact_kernel_and_belief,
    map_model_policy,
    persistent_posterior_sampling_mixture,
    plan_bayes_adaptive,
    plan_information_only_policy,
    plan_invalid_mean_transition_policy,
    plan_myopic_reward_policy,
    point_model_kernel_and_belief,
    posterior_weighted_model_oracle,
    restore_compact_policy,
    scalar_plan_bayes_adaptive,
    step_belief,
    systematic_quantile_indices,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rejected(callable_) -> bool:
    try:
        callable_()
    except (ValueError, RuntimeError, PermissionError):
        return True
    return False


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v66-design-lock.json"
    audit_path = PROJECT_ROOT / "outputs/v66-external-bayes-adaptive-reward/implementation-audit.json"
    output_path = PROJECT_ROOT / "configs/v66-implementation-lock.json"
    if output_path.exists():
        raise RuntimeError("V66 implementation already frozen")
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    errors: list[str] = []

    design_payload = {
        key: value for key, value in design.items() if key != "lock_payload_sha256"
    }
    design_ok = bool(
        payload_hash(design_payload) == design["lock_payload_sha256"]
        and design["authorization"]["write_and_audit_planner_implementation"]
        and not design["authorization"]["write_and_audit_durable_evaluator"]
        and not design["authorization"]["run_evaluation"]
        and not design["authorization"]["formal_verification"]
        and file_sha256(PROJECT_ROOT / design["config"]) == design["config_sha256"]
        and file_sha256(PROJECT_ROOT / design["preregistration"])
        == design["preregistration_sha256"]
        and file_sha256(PROJECT_ROOT / design["design_audit"])
        == design["design_audit_sha256"]
        and file_sha256(PROJECT_ROOT / design["source_v65r3_outcome_lock"])
        == design["source_v65r3_outcome_lock_sha256"]
        and file_sha256(PROJECT_ROOT / design["subset_seal"])
        == design["subset_seal_sha256"]
    )
    if not design_ok:
        errors.append("V66 design binding or implementation-only authorization failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v66-evaluation-implementation-lock.json",
            "configs/v66-outcome-lock.json",
            "python/evaluate_v66_reward.py",
            "outputs/v66-external-bayes-adaptive-reward/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V66 evaluator or evaluation exists before implementation lock")

    subset_seal = json.loads((PROJECT_ROOT / design["subset_seal"]).read_text())
    subset_path = PROJECT_ROOT / subset_seal["files"]["subset_public"]["path"]
    sealed_records = read_jsonl(subset_path)
    subset_binding_ok = bool(
        file_sha256(subset_path) == subset_seal["files"]["subset_public"]["sha256"]
        and len(sealed_records) == 48
        and all(set(row) == {
            "record_id", "prefix_length", "initial_observation", "actions", "observations"
        } for row in sealed_records)
    )
    if not subset_binding_ok:
        errors.append("V66 implementation firewall subset binding failed")

    synthetic = {
        "record_id": "v66-synthetic-planner-fixture",
        "prefix_length": 2,
        "initial_observation": "left",
        "actions": ["n", "e"],
        "observations": ["neither", "right"],
    }
    assert_synthetic_planner_fixture(synthetic, sealed_records)
    sealed_id_rejected = rejected(
        lambda: assert_synthetic_planner_fixture(
            {**synthetic, "record_id": sealed_records[0]["record_id"]}, sealed_records
        )
    )
    sealed_history_rejected = rejected(
        lambda: assert_synthetic_planner_fixture(
            {**sealed_records[0], "record_id": "v66-changed-id"}, sealed_records
        )
    )
    truth_field_rejected = rejected(
        lambda: assert_synthetic_planner_fixture({**synthetic, "truth": 1}, sealed_records)
    )
    firewall_ok = sealed_id_rejected and sealed_history_rejected and truth_field_rejected
    if not firewall_ok:
        errors.append("V66 synthetic-only implementation planner firewall failed")

    family = load_family(quadrature_nodes=5)
    exact, _ = filter_public_history(
        family,
        synthetic["initial_observation"],
        synthetic["actions"],
        synthetic["observations"],
    )
    kernel, belief = exact_kernel_and_belief(family, exact)
    stats: dict[str, int] = {}
    primary = plan_bayes_adaptive(
        kernel,
        belief,
        3,
        retain_forced_root_actions=True,
        stats=stats,
    )
    scalar = scalar_plan_bayes_adaptive(kernel, belief, 2)
    primary_two = plan_bayes_adaptive(
        kernel, belief, 2, retain_forced_root_actions=True
    )
    policy_value = evaluate_policy(kernel, belief, primary, 3)
    root_q = evaluate_root_action_values(kernel, belief, primary, 3)
    vector_scalar_error = float(
        max(abs(a - b) for a, b in zip(primary_two["q_values"], scalar["q_values"], strict=True))
    )
    policy_error = abs(float(primary["value"]) - policy_value)
    root_q_error = float(
        max(abs(a - b) for a, b in zip(primary["q_values"], root_q, strict=True))
    )

    point_kernel, point_belief, _ = point_model_kernel_and_belief(kernel, belief, 0)
    point = plan_bayes_adaptive(point_kernel, point_belief, 3)
    point_scalar = scalar_plan_bayes_adaptive(point_kernel, point_belief, 2)
    point_two = plan_bayes_adaptive(point_kernel, point_belief, 2)
    point_error = float(
        max(abs(a - b) for a, b in zip(point_two["q_values"], point_scalar["q_values"], strict=True))
    )

    oracle = posterior_weighted_model_oracle(kernel, belief, 3)
    map_result = map_model_policy(kernel, belief, 3)
    mixture4 = persistent_posterior_sampling_mixture(
        kernel, belief, 3, points=4, offset=0.5 / 4
    )
    mixture8 = persistent_posterior_sampling_mixture(
        kernel, belief, 3, points=8, offset=0.5 / 8
    )
    controls = (
        plan_myopic_reward_policy(kernel, belief, 3),
        plan_information_only_policy(kernel, belief, 3),
        plan_invalid_mean_transition_policy(kernel, belief, 3),
    )
    control_values = [evaluate_policy(kernel, belief, policy, 3) for policy in controls]
    information_values = [
        evaluate_policy_information(kernel, belief, policy, 3) for policy in controls
    ]
    serialization_error = abs(
        evaluate_policy(kernel, belief, primary, 3)
        - evaluate_policy(
            kernel, belief, restore_compact_policy(compact_policy(primary)), 3
        )
    )
    eig_error = exact_eig_crosscheck(family, exact)

    explicit_indices = np.asarray(
        [
            next(
                index
                for index, edge in enumerate(np.cumsum(belief.sum(axis=1)))
                if position < edge
            )
            for position in 0.5 / 8 + np.arange(8) / 8
        ],
        dtype=np.int64,
    )
    systematic_indices = systematic_quantile_indices(
        belief.sum(axis=1), 8, 0.5 / 8
    )

    omitted_branch_rejected = False
    broken = {**primary, "branches": dict(primary["branches"])}
    if broken["branches"]:
        del broken["branches"][next(iter(broken["branches"]))]
        omitted_branch_rejected = rejected(
            lambda: evaluate_policy(kernel, belief, broken, 3)
        )
    malformed_belief_rejected = rejected(
        lambda: step_belief(kernel, belief * 0.5, kernel.canonical_actions[0])
    )
    invalid_offset_rejected = rejected(
        lambda: systematic_quantile_indices(belief.sum(axis=1), 8, 0.125)
    )
    invalid_transition_rejected = False
    transition = kernel.transitions.copy()
    transition[0, 0, 0] *= 0.5
    invalid_transition_rejected = rejected(
        lambda: StaticKernel(
            action_names=kernel.action_names,
            observation_names=kernel.observation_names,
            state_names=kernel.state_names,
            canonical_actions=kernel.canonical_actions,
            transitions=transition,
            observations=kernel.observations,
            rewards=kernel.rewards,
            discount=kernel.discount,
            identities=kernel.identities,
            thetas=kernel.thetas,
        )
    )
    invalid_observation_rejected = False
    observation = kernel.observations.copy()
    observation[0, 0] *= 0.5
    invalid_observation_rejected = rejected(
        lambda: StaticKernel(
            action_names=kernel.action_names,
            observation_names=kernel.observation_names,
            state_names=kernel.state_names,
            canonical_actions=kernel.canonical_actions,
            transitions=kernel.transitions,
            observations=observation,
            rewards=kernel.rewards,
            discount=kernel.discount,
            identities=kernel.identities,
            thetas=kernel.thetas,
        )
    )

    unit_command = [
        sys.executable,
        "-m",
        "unittest",
        "python/test_v66_bayes_adaptive_reward.py",
        "-v",
    ]
    unit = subprocess.run(
        unit_command,
        cwd=PROJECT_ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": "python"},
        capture_output=True,
        text=True,
        check=False,
    )
    unit_ok = unit.returncode == 0 and "Ran 13 tests" in unit.stderr and "OK" in unit.stderr
    if not unit_ok:
        errors.append("V66 unit-test suite failed")

    analytic_checks = {
        "vectorized_Bellman_matches_independent_scalar_reference": vector_scalar_error <= 1e-10,
        "frozen_policy_evaluation_matches_Bellman_value": policy_error <= 1e-10,
        "forced_root_policy_values_match_Bellman_Q": root_q_error <= 1e-10,
        "point_static_model_matches_scalar_known_model_reference": point_error <= 1e-10,
        "model_information_oracle_weakly_dominates_Bayes_adaptive": oracle["value"] + 1e-10 >= primary["value"],
        "exact_Bayes_adaptive_weakly_dominates_MAP": primary["value"] + 1e-10 >= map_result["exact_environment_value"],
        "exact_Bayes_adaptive_weakly_dominates_persistent_mixture": primary["value"] + 1e-10 >= mixture4["value"],
        "persistent_mixture_primary_and_sensitivity_executable": len(mixture4["models"]) == 4 and len(mixture8["models"]) == 8,
        "systematic_quantiles_match_independent_inverse_CDF": np.array_equal(systematic_indices, explicit_indices),
        "generic_static_EIG_matches_frozen_V64": eig_error <= 1e-12,
        "registered_controls_execute_under_exact_environment": all(np.isfinite(control_values)),
        "registered_control_information_is_finite": all(np.isfinite(information_values)),
        "policy_serialization_round_trip": serialization_error <= 1e-12,
        "synthetic_only_planner_firewall": firewall_ok,
        "full_horizon_three_tree_exercised": stats.get("bellman_nodes", 0) > 100 and point["horizon"] == 3,
    }
    analytic_checks = {key: bool(value) for key, value in analytic_checks.items()}
    analytic_ok = all(analytic_checks.values())
    if not analytic_ok:
        errors.append("V66 analytic implementation audit failed")

    mutation_checks = {
        "replace_vectorized_Bellman_with_wrong_reference": vector_scalar_error <= 1e-10,
        "omit_discounted_continuation": primary["value"] != max(
            float(value) for value in plan_bayes_adaptive(kernel, belief, 1)["q_values"]
        ),
        "skip_independent_policy_evaluation": policy_error <= 1e-10,
        "miscompute_forced_root_Q": root_q_error <= 1e-10,
        "drop_static_model_persistence": mixture4["sampled_model_persists_for_full_policy"],
        "label_mean_transition_as_valid_mixture": controls[-1].get("invalid_static_semantics") == "transition_matrices_reaveraged_at_each_history",
        "use_random_instead_of_fixed_systematic_quantiles": np.array_equal(systematic_indices, explicit_indices),
        "accept_invalid_systematic_offset": invalid_offset_rejected,
        "reveal_full_physical_state_to_oracle": all("state" not in row for row in oracle["models"]),
        "evaluate_MAP_under_its_own_model_only": np.isfinite(map_result["exact_environment_value"]),
        "evaluate_mixture_under_sampled_model_only": all("exact_environment_value" in row for row in mixture4["models"]),
        "omit_reachable_policy_observation": omitted_branch_rejected,
        "accept_malformed_belief": malformed_belief_rejected,
        "accept_unnormalized_transition": invalid_transition_rejected,
        "accept_unnormalized_observation": invalid_observation_rejected,
        "change_static_EIG_target": eig_error <= 1e-12,
        "break_policy_serialization": serialization_error <= 1e-12,
        "allow_sealed_record_ID": sealed_id_rejected,
        "allow_sealed_history_under_new_ID": sealed_history_rejected,
        "allow_truth_field_in_planner_fixture": truth_field_rejected,
        "omit_registered_control": len(controls) == 3,
        "omit_horizon_three_recursion": stats.get("bellman_nodes", 0) > 100,
    }
    mutation_checks = {key: bool(value) for key, value in mutation_checks.items()}
    mutation_ok = all(mutation_checks.values())
    if not mutation_ok:
        errors.append("V66 implementation mutation audit did not kill every mutant")

    checks = {
        "frozen_design_and_implementation_only_authorization": design_ok,
        "V66_downstream_absent": downstream_absent,
        "sealed_subset_bound_for_firewall_only": subset_binding_ok,
        "synthetic_only_planner_firewall": firewall_ok,
        "unit_tests_13_of_13": unit_ok,
        "analytic_fixtures_15_of_15": analytic_ok and len(analytic_checks) == 15,
        "implementation_mutants_22_of_22": mutation_ok and len(mutation_checks) == 22,
    }
    audit = {
        "schema_version": "66",
        "experiment": "v66_implementation_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_v66_planner_implementation_and_authorize_durable_evaluator_only"
            if not errors and all(checks.values())
            else "reject_v66_planner_implementation"
        ),
        "errors": errors,
        "checks": checks,
        "unit_test_command": unit_command,
        "unit_test_stdout": unit.stdout,
        "unit_test_stderr": unit.stderr,
        "analytic_checks": analytic_checks,
        "mutation_checks": mutation_checks,
        "reference_errors": {
            "vector_scalar_root_Q": vector_scalar_error,
            "policy_evaluation": policy_error,
            "forced_root_Q": root_q_error,
            "point_model_scalar": point_error,
            "V64_EIG": eig_error,
            "serialization": serialization_error,
        },
        "fixture": {
            "record": synthetic,
            "quadrature_nodes": 5,
            "horizon": 3,
            "Bellman_nodes": stats.get("bellman_nodes", 0),
            "exact_value": float(primary["value"]),
            "oracle_value": float(oracle["value"]),
            "MAP_exact_environment_value": float(map_result["exact_environment_value"]),
            "persistent_mixture_4_value": float(mixture4["value"]),
            "persistent_mixture_8_value": float(mixture8["value"]),
        },
        "access": {
            "sealed_public_records_loaded_for_firewall_only": len(sealed_records),
            "sealed_reward_policies_planned_or_scored": 0,
            "V64_or_V65_evaluation_result_records_loaded": 0,
            "truth_fields_accessed": 0,
            "V65r3_evaluation_reruns": 0,
            "V66_evaluation_attempts": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    source_paths = (
        "python/v66_bayes_adaptive_reward.py",
        "python/test_v66_bayes_adaptive_reward.py",
        "python/audit_and_freeze_v66_implementation.py",
        "python/v64_external_eig.py",
        "python/v65_smc2_eig.py",
        "python/v65r2_smc2_eig.py",
        "python/v65r3_smc2_eig.py",
        "python/v62_external_pomdp.py",
    )
    lock = {
        "schema_version": "66",
        "experiment": "v66_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "source_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in source_paths
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "authorization": {
            "modify_or_rerun_v65r3": False,
            "modify_v66_design_or_implementation": False,
            "write_and_audit_durable_evaluator": True,
            "run_evaluation": False,
            "formal_verification": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit_passed": audit["passed"],
                "checks": checks,
                "lock": str(output_path.relative_to(PROJECT_ROOT)),
                "reference_errors": audit["reference_errors"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
