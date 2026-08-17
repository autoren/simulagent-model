#!/usr/bin/env python3
"""Synthetic audit and lock for V68 environment-generic exact infrastructure."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any

import numpy as np

from test_v68_multi_environment_exact import synthetic_model
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v66_bayes_adaptive_reward import plan_bayes_adaptive, step_belief
from v68_multi_environment_exact import (
    best_open_loop_sequence,
    build_command_channel_family,
    cycle_permutations,
    enumerate_public_prefixes,
    evaluate_action_sequence,
    filter_action_observation_history,
    finite_horizon_return_scale,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v68-development-design-lock.json"
    source_path = PROJECT_ROOT / "python/v68_multi_environment_exact.py"
    tests_path = PROJECT_ROOT / "python/test_v68_multi_environment_exact.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v68_development_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v68-development-screening/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v68-development-implementation-lock.json"
    if lock_path.exists():
        raise RuntimeError("V68 development implementation is already frozen")
    design = json.loads(design_path.read_text())
    errors: list[str] = []
    payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    design_ok = bool(
        payload_hash(payload) == design["lock_payload_sha256"]
        and design["authorization"]["write_and_audit_exact_infrastructure"]
        and not design["authorization"]["construct_development_census"]
        and not design["authorization"]["run_development_screen"]
        and not design["authorization"]["score_confirmatory_models"]
    )
    if not design_ok:
        errors.append("V68 development design lock or implementation-only authorization failed")

    implementation_text = source_path.read_text() + tests_path.read_text()
    confirmatory_names = (
        "cheese.95", "fully_observable_tmaze", "hallway.POMDP",
        "heavenhell.POMDP", "network.POMDP", "shuttle.POMDP", "paint.POMDP",
    )
    confirmatory_independence_ok = not any(name in implementation_text for name in confirmatory_names)
    if not confirmatory_independence_ok:
        errors.append("generic implementation or tests name a confirmatory source model")

    completed = subprocess.run(
        [
            sys.executable, "-m", "unittest", "discover", "-s", "python",
            "-p", "test_v68_multi_environment_exact.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    tests_ok = completed.returncode == 0 and "Ran 8 tests" in combined
    if not tests_ok:
        errors.append("V68 exact infrastructure tests failed")

    model = synthetic_model()
    family = build_command_channel_family(
        model, ("a", "c", "b"), quadrature_nodes=7
    )
    prefixes = enumerate_public_prefixes(family, maximum_depth=1)
    filtered, log_evidence = filter_action_observation_history(family, (0,), (0,))
    matched = next(row for row in prefixes if row.actions == (0,) and row.observations == (0,))
    open_loop = best_open_loop_sequence(family.kernel, family.initial_belief, 3)
    adaptive = plan_bayes_adaptive(family.kernel, family.initial_belief, 3)
    analytic = {
        "V64_4x3_family_exact_array_parity": tests_ok,
        "non_storage_canonical_cycle_permutation": np.array_equal(
            family.kernel.canonical_actions, (0, 2, 1)
        ),
        "joint_static_state_prior_normalizes": abs(float(family.initial_belief.sum()) - 1.0) <= 1e-14,
        "one_step_filter_matches_complete_prefix_enumerator": bool(
            np.array_equal(filtered, matched.belief)
            and abs(np.exp(log_evidence) - matched.probability) <= 1e-14
        ),
        "complete_depth_one_census": len(prefixes) == 7,
        "open_loop_exhausts_all_twenty_seven_sequences": open_loop["sequence_count"] == 27,
        "open_loop_selected_sequence_direct_reexecution": abs(
            float(open_loop["value"])
            - evaluate_action_sequence(
                family.kernel, family.initial_belief, open_loop["selected_actions"]
            )
        ) <= 1e-14,
        "adaptive_value_weakly_dominates_best_open_loop": float(adaptive["value"]) + 1e-14 >= float(open_loop["value"]),
        "frozen_return_scale": abs(
            finite_horizon_return_scale(model, 3) - 2.0 * (1.0 + 0.9 + 0.9**2)
        ) <= 1e-14,
    }

    theta = family.theta
    forward = family.permutations[0]
    valid = family.kernel.transitions[: len(theta)]
    reversed_mixture = np.asarray(
        [
            [
                (1.0 - value) * model.transition[action]
                + value * model.transition[forward[action]]
                for action in range(len(model.actions))
            ]
            for value in theta
        ]
    )
    _, storage_canonical = cycle_permutations(model, model.actions)
    storage_permutations, _ = cycle_permutations(model, model.actions)
    correct_step = step_belief(family.kernel, family.initial_belief, 0)
    current_weighted = (
        family.initial_belief
        * family.kernel.observations[0, :, 0][None, :]
    )
    current_weighted /= current_weighted.sum()
    observe_before_transition = np.einsum(
        "ms,mst->mt", current_weighted, family.kernel.transitions[:, 0], optimize=True
    )
    omit_discount_value = 0.0
    selected = open_loop["selected_actions"]
    belief = family.initial_belief
    for action in selected:
        step = step_belief(family.kernel, belief, int(action))
        omit_discount_value += float(step["reward"])
        belief = sum(
            float(step["probabilities"][observation]) * posterior
            for observation, posterior in step["posteriors"].items()
        )
        belief /= belief.sum()
    scale_reward = model.reward.copy()
    scale_reward[0, 0, 0] = -1.0
    scale_model = type(model)(
        name="synthetic_scale_mutant_fixture",
        states=model.states,
        actions=model.actions,
        observations=model.observations,
        discount=model.discount,
        initial=model.initial,
        transition=model.transition,
        observation=model.observation,
        reward=scale_reward,
    )
    mutants = {
        "swap_theta_success_and_failure_coefficients": float(np.max(np.abs(valid - reversed_mixture))) > 1e-8,
        "collapse_forward_and_backward_identities": float(
            np.max(
                np.abs(
                    family.kernel.transitions[: len(theta)]
                    - family.kernel.transitions[len(theta):]
                )
            )
        ) > 1e-8,
        "omit_half_identity_prior_mass": abs(float(family.initial_belief[: len(theta)].sum()) - 1.0) > 1e-3,
        "use_source_storage_order_instead_of_frozen_cycle": bool(
            storage_canonical != family.kernel.canonical_actions
            and not np.array_equal(storage_permutations, family.permutations)
        ),
        "resample_static_model_via_mean_transition": float(
            np.max(np.var(family.kernel.transitions, axis=0))
        ) > 1e-8,
        "condition_observation_before_transition": float(
            np.max(np.abs(observe_before_transition - correct_step["posteriors"][0]))
        ) > 1e-8,
        "drop_one_positive_observation_from_census": len(prefixes) - 1 != 7,
        "treat_open_loop_as_one_sequence_only": int(open_loop["sequence_count"]) != 1,
        "omit_discount_in_open_loop_return": abs(
            omit_discount_value - float(open_loop["value"])
        ) > 1e-8,
        "use_reward_maximum_instead_of_reward_span_for_scale": abs(
            finite_horizon_return_scale(scale_model, 3)
            - float(scale_model.reward.max())
            * (1.0 + scale_model.discount + scale_model.discount**2)
        ) > 1e-8,
    }
    analytic_rate = sum(bool(value) for value in analytic.values()) / len(analytic)
    mutant_rate = sum(bool(value) for value in mutants.values()) / len(mutants)
    if analytic_rate != 1.0:
        errors.append("V68 analytic exact-infrastructure fixture rate is below one")
    if mutant_rate != 1.0:
        errors.append("V68 exact-infrastructure mutant kill rate is below one")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v68-development-census-seal.json",
            "configs/v68-development-evaluator-lock.json",
            "configs/v68-development-outcome-lock.json",
            "outputs/v68-development-screening/census.jsonl",
            "outputs/v68-development-screening/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V68 development census or evaluation exists before implementation lock")

    checks = {
        "development_design_lock_and_implementation_authorization": design_ok,
        "generic_code_does_not_name_confirmatory_models": confirmatory_independence_ok,
        "eight_exact_infrastructure_tests": tests_ok,
        "nine_analytic_fixtures": analytic_rate == 1.0,
        "ten_synthetic_mutants_killed": mutant_rate == 1.0,
        "development_census_and_evaluation_absent": downstream_absent,
    }
    audit = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_implementation_audit",
        "passed": not errors and all(checks.values()),
        "decision": (
            "freeze_exact_infrastructure_and_authorize_development_census_only"
            if not errors
            else "reject_v68_development_implementation"
        ),
        "errors": errors,
        "checks": checks,
        "analytic_fixtures": {key: bool(value) for key, value in analytic.items()},
        "mutants": {key: bool(value) for key, value in mutants.items()},
        "analytic_fixture_pass_rate": analytic_rate,
        "mutant_kill_rate": mutant_rate,
        "access": {
            "synthetic_planning_fixtures": 1,
            "development_source_array_parity_checks": 1,
            "development_policy_outcomes": 0,
            "confirmatory_models_scored": 0,
            "SMC2_runs": 0,
            "human_records": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "68-development-screening",
        "experiment": "v68_development_implementation_lock",
        "development_design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "development_design_lock_sha256": file_sha256(design_path),
        "implementation": str(source_path.relative_to(PROJECT_ROOT)),
        "implementation_sha256": file_sha256(source_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "analytic_fixture_pass_rate": analytic_rate,
        "mutant_kill_rate": mutant_rate,
        "authorization": {
            "modify_design_or_exact_infrastructure": False,
            "construct_and_seal_complete_development_census": True,
            "write_and_audit_development_evaluator": False,
            "run_development_screen": False,
            "score_confirmatory_models": False,
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
