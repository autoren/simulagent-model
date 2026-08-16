#!/usr/bin/env python3
"""Synthetic audit and implementation lock for the V67 verifier."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from test_v67_verification import fixture_model, policy_tree
from v67_verification import (
    compile_policy_dtmc,
    condition_public_history,
    construct_independent_family,
    dtmc_statistics,
    execute_policy_scalar,
    independent_scaled_beta_2_2_quadrature,
    run_storm_properties,
    scalar_step,
    storm_version,
    write_explicit_dtmc,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v67-design-lock.json"
    source_path = PROJECT_ROOT / "python/v67_verification.py"
    tests_path = PROJECT_ROOT / "python/test_v67_verification.py"
    audit_source_path = PROJECT_ROOT / "python/audit_and_freeze_v67_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v67-independent-bounded-policy-verification/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v67-implementation-lock.json"
    if lock_path.exists():
        raise RuntimeError("V67 implementation already frozen")
    design = json.loads(design_path.read_text())
    errors: list[str] = []
    design_payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    design_ok = bool(
        payload_hash(design_payload) == design["lock_payload_sha256"]
        and design["authorization"]["write_and_audit_independent_implementation"]
        and not design["authorization"]["load_and_execute_all_source_policies"]
        and not design["authorization"]["run_verification"]
    )
    if not design_ok:
        errors.append("V67 design lock or implementation-only authorization failed")

    text = source_path.read_text()
    prohibited = (
        "from v62_external_pomdp", "import v62_external_pomdp",
        "from v64_external_eig", "import v64_external_eig",
        "from v66_bayes_adaptive_reward", "import v66_bayes_adaptive_reward",
        "from evaluate_v66_reward", "import evaluate_v66_reward",
    )
    independence_ok = not any(token in text for token in prohibited)
    if not independence_ok:
        errors.append("V67 verifier imports a prohibited source implementation")

    completed = subprocess.run(
        [sys.executable, str(tests_path)], cwd=PROJECT_ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(PROJECT_ROOT / "python")},
        capture_output=True, text=True,
    )
    tests_run = 0
    for line in (completed.stdout + completed.stderr).splitlines():
        if line.startswith("Ran "):
            tests_run = int(line.split()[1])
    tests_ok = completed.returncode == 0 and tests_run >= 14
    if not tests_ok:
        errors.append("V67 synthetic unit tests failed")

    family = construct_independent_family(fixture_model(two_observations=True), nodes=7)
    belief = family.static_prior[:, :, None] * np.asarray([0.5, 0.5])[None, None, :]
    policy = policy_tree(3, 2, root_action=2, child_action=3)
    valid = execute_policy_scalar(family, belief, policy)["value"]
    model, checks = compile_policy_dtmc(family, belief, policy)
    direct = dtmc_statistics(model)
    with tempfile.TemporaryDirectory() as directory:
        write_explicit_dtmc(model, directory)
        external = run_storm_properties(directory)
    analytic = {
        "deterministic_single_observation_three_action_discounted_return": tests_ok,
        "two_observation_policy_changes_its_second_action": tests_ok,
        "persistent_static_atom_is_not_resampled_between_nodes": tests_ok,
        "conditional_transition_reward_recovers_unconditional_expected_reward": tests_ok,
        "independent_scalar_filter_matches_hand_computed_two_state_update": tests_ok,
        "Storm_explicit_writer_round_trips_a_tiny_DTMC": (
            abs(external["expected_return"] - direct["expected_return"]) <= 1e-12
            and abs(external["termination_probability"] - 1.0) <= 1e-12
        ),
    }

    mutants: dict[str, bool] = {}
    mutants["omit_discount"] = abs(
        execute_policy_scalar(family, belief, policy, mutation="omit_discount")["value"] - valid
    ) > 1e-8
    mutants["discount_by_remaining_horizon"] = abs(
        execute_policy_scalar(
            family, belief, policy, mutation="discount_by_remaining_horizon"
        )["value"] - valid
    ) > 1e-8
    reward_family = construct_independent_family(fixture_model(), nodes=7)
    reward_belief = (
        reward_family.static_prior[:, :, None]
        * reward_family.model.initial[None, None, :]
    )
    reward_policy = policy_tree(3, 1, root_action=0)
    mutants["use_successor_instead_of_current_state_reward_index"] = abs(
        execute_policy_scalar(
            reward_family, reward_belief, reward_policy,
            mutation="use_successor_instead_of_current_state_reward_index",
        )["value"]
        - execute_policy_scalar(reward_family, reward_belief, reward_policy)["value"]
    ) > 1e-8
    persistence_policy = policy_tree(3, 2, root_action=0, child_action=2)
    persistence_valid = execute_policy_scalar(family, belief, persistence_policy)["value"]
    mutants["replace_persistent_static_model_with_per_step_mean_transition"] = abs(
        execute_policy_scalar(
            family, belief, persistence_policy,
            mutation="replace_persistent_static_model_with_per_step_mean_transition",
        )["value"] - persistence_valid
    ) > 1e-8
    asymmetric = belief.copy()
    asymmetric[0] *= 1.8
    asymmetric[1] *= 0.2
    asymmetric /= asymmetric.sum()
    mutants["swap_clockwise_and_counterclockwise_identity"] = abs(
        execute_policy_scalar(
            family, asymmetric, policy,
            mutation="swap_clockwise_and_counterclockwise_identity",
        )["value"]
        - execute_policy_scalar(family, asymmetric, policy)["value"]
    ) > 1e-8
    reset_source = fixture_model(two_observations=True)
    reset_source = type(reset_source)(
        states=reset_source.states, actions=reset_source.actions,
        observations=reset_source.observations, discount=reset_source.discount,
        initial=np.asarray([0.5, 0.5]), transition=reset_source.transition,
        observation=reset_source.observation, reward=reset_source.reward,
    )
    reset_family = construct_independent_family(reset_source, nodes=7)
    record = {
        "record_id": "synthetic", "prefix_length": 0,
        "initial_observation": "zero", "actions": [], "observations": [],
    }
    conditioned = condition_public_history(reset_family, record)[0]
    unconditioned = condition_public_history(
        reset_family, record, mutation="omit_reset_initial_observation"
    )[0]
    mutants["omit_reset_initial_observation"] = float(
        np.max(np.abs(conditioned - unconditioned))
    ) > 1e-8
    dropped, dropped_checks = compile_policy_dtmc(
        family, belief, policy, mutation="drop_positive_observation_branch"
    )
    mutants["drop_positive_observation_branch"] = (
        dropped_checks["transition_normalization_passes"]
        < dropped_checks["transition_normalization_checks"]
        or dtmc_statistics(dropped)["termination_probability"] < 1.0 - 1e-10
    )
    merged, _ = compile_policy_dtmc(
        family, belief, policy, mutation="merge_distinct_observation_branches"
    )
    mutants["merge_distinct_observation_branches"] = abs(
        dtmc_statistics(merged)["expected_return"] - valid
    ) > 1e-8
    archived_policy = policy_tree(
        3, 2, root_action=2, child_action=3, archived_probabilities=[0.9, 0.1]
    )
    archived_model, _ = compile_policy_dtmc(
        family, belief, archived_policy,
        mutation="use_archived_SMC2_branch_probabilities_instead_of_exact_probabilities",
    )
    archived_valid = execute_policy_scalar(family, belief, archived_policy)["value"]
    mutants["use_archived_SMC2_branch_probabilities_instead_of_exact_probabilities"] = abs(
        dtmc_statistics(archived_model)["expected_return"] - archived_valid
    ) > 1e-8
    action_policy = policy_tree(3, 2, root_action=2, child_action=3)
    action_policy["optimal_actions"] = [0, 2]
    mutants["replace_archived_selected_action_with_first_optimal_action"] = abs(
        execute_policy_scalar(
            family, belief, action_policy,
            mutation="replace_archived_selected_action_with_first_optimal_action",
        )["value"] - execute_policy_scalar(family, belief, action_policy)["value"]
    ) > 1e-8
    wrong = policy_tree(3, 2, root_action=2, child_action=3)
    wrong["horizon"] = 2
    normal_rejected = False
    mutant_accepted = False
    try:
        execute_policy_scalar(family, belief, wrong)
    except ValueError:
        normal_rejected = True
    try:
        execute_policy_scalar(family, belief, wrong, mutation="accept_wrong_policy_horizon")
        mutant_accepted = True
    except ValueError:
        pass
    mutants["accept_wrong_policy_horizon"] = normal_rejected and mutant_accepted
    mutants["corrupt_source_reward"] = abs(
        execute_policy_scalar(
            family, belief, policy, mutation="corrupt_source_reward"
        )["value"] - valid
    ) > 1e-8
    _, valid_weights = independent_scaled_beta_2_2_quadrature(7)
    _, corrupt_weights = independent_scaled_beta_2_2_quadrature(
        7, mutation="corrupt_quadrature_weight"
    )
    mutants["corrupt_quadrature_weight"] = float(
        np.max(np.abs(valid_weights - corrupt_weights))
    ) > 1e-10
    no_done, _ = compile_policy_dtmc(
        family, belief, policy, mutation="omit_done_transition"
    )
    mutants["omit_done_transition"] = dtmc_statistics(no_done)[
        "termination_probability"
    ] < 1.0 - 1e-10

    analytic_rate = sum(analytic.values()) / len(analytic)
    mutant_rate = sum(mutants.values()) / len(mutants)
    checks_ok = bool(
        checks["node_invariants"] == checks["node_invariant_passes"]
        and checks["branch_totality_checks"] == checks["branch_totality_passes"]
        and checks["transition_normalization_checks"]
        == checks["transition_normalization_passes"]
        and checks["nonterminal_deadlocks"] == 0
        and abs(valid - direct["expected_return"]) <= 1e-12
    )
    if analytic_rate != 1.0:
        errors.append("V67 analytic fixture pass rate is below one")
    if mutant_rate != 1.0:
        errors.append("V67 mutation kill rate is below one")
    if not checks_ok:
        errors.append("V67 synthetic compiler/executor cross-check failed")
    if storm_version() != "1.13.0":
        errors.append("V67 Storm version mismatch")

    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in (
        "configs/v67-verification-bundle-seal.json",
        "configs/v67-evaluation-implementation-lock.json",
        "outputs/v67-independent-bounded-policy-verification/bundle",
        "outputs/v67-independent-bounded-policy-verification/verification",
    ))
    if not downstream_absent:
        errors.append("V67 bundle or verification exists before implementation lock")

    audit = {
        "schema_version": "67",
        "experiment": "v67_implementation_audit",
        "passed": not errors,
        "decision": "freeze_implementation_and_authorize_bundle_construction_only" if not errors else "reject_implementation",
        "errors": errors,
        "checks": {
            "design_lock_and_implementation_only_authorization": design_ok,
            "prohibited_source_implementation_imports_absent": independence_ok,
            "synthetic_unit_tests": tests_ok,
            "analytic_fixture_pass_rate": analytic_rate,
            "implementation_mutant_kill_rate": mutant_rate,
            "scalar_compiler_and_reachable_checks": checks_ok,
            "Storm_version": storm_version(),
            "downstream_absent": downstream_absent,
        },
        "analytic_fixtures": analytic,
        "mutants": mutants,
        "unit_tests": {
            "tests_run": tests_run,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        },
        "access": {
            "pinned_external_source_array_files_loaded": 1,
            "source_V66_policy_rows_loaded_or_executed": 0,
            "truth_fields_accessed": 0,
            "V66_evaluation_reruns": 0,
            "V67_verification_attempts": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "67",
        "experiment": "v67_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation": str(source_path.relative_to(PROJECT_ROOT)),
        "implementation_sha256": file_sha256(source_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "implementation_auditor": str(audit_source_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(audit_source_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "tool_versions": {"Storm": storm_version(), "Python": sys.version.split()[0]},
        "authorization": {
            "modify_or_rerun_v66": False,
            "modify_v67_design_or_implementation": False,
            "load_and_execute_all_source_policies": True,
            "build_verification_bundle": True,
            "write_and_audit_durable_evaluator": False,
            "run_verification": False,
            "truth_field_access": False,
            "human_data": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"implementation_lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
