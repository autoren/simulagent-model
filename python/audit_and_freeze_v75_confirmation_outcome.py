#!/usr/bin/env python3
"""Independently reconstruct and freeze the V75 replication outcome."""
from __future__ import annotations

import hashlib
from itertools import product
import json
from typing import Any, Sequence

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v75_nova_paint_source import ACTION_NAMES, build_family


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _select(values: Sequence[float], tolerance: float) -> tuple[int, tuple[int, ...], float]:
    maximum = max(float(value) for value in values)
    optimal = tuple(
        index for index, value in enumerate(values) if maximum - float(value) <= tolerance
    )
    return optimal[0], optimal, maximum


def _joint_step(kernel: Any, belief: np.ndarray, action: int):
    predicted = np.einsum("zs,sq->zq", belief, kernel.transition[action])
    joint = predicted[:, :, None] * kernel.observation[:, action]
    probabilities = joint.sum(axis=(0, 1))
    posteriors = {
        observation: joint[:, :, observation] / probability
        for observation, probability in enumerate(probabilities)
        if probability > 0.0
    }
    reward = float(
        np.einsum("zs,sq,sq->", belief, kernel.transition[action], kernel.reward[action])
    )
    return reward, probabilities, posteriors


def _point_step(kernel: Any, belief: np.ndarray, latent: int, action: int):
    predicted = belief @ kernel.transition[action]
    joint = predicted[:, None] * kernel.observation[latent, action]
    probabilities = joint.sum(axis=0)
    posteriors = {
        observation: joint[:, observation] / probability
        for observation, probability in enumerate(probabilities)
        if probability > 0.0
    }
    reward = float(
        np.einsum("s,sq,sq->", belief, kernel.transition[action], kernel.reward[action])
    )
    return reward, probabilities, posteriors


def _solve_joint(kernel: Any, belief: np.ndarray, horizon: int, tolerance: float):
    if horizon == 0:
        return {"terminal": True, "value": 0.0}
    rows = []
    for action in range(len(ACTION_NAMES)):
        reward, probabilities, posteriors = _joint_step(kernel, belief, action)
        branches = {}
        continuation = 0.0
        if horizon > 1:
            for observation, posterior in posteriors.items():
                child = _solve_joint(kernel, posterior, horizon - 1, tolerance)
                branches[observation] = child
                continuation += float(probabilities[observation]) * float(child["value"])
        rows.append(
            {
                "action": action,
                "value": float(reward + kernel.discount * continuation),
                "branches": branches,
            }
        )
    selected, optimal, maximum = _select([row["value"] for row in rows], tolerance)
    return {
        "terminal": False,
        "value": maximum,
        "selected_action": selected,
        "optimal_actions": optimal,
        "q_values": tuple(float(row["value"]) for row in rows),
        "branches": rows[selected]["branches"],
    }


def _solve_point(kernel: Any, belief: np.ndarray, latent: int, horizon: int, tolerance: float):
    if horizon == 0:
        return {"terminal": True, "value": 0.0}
    rows = []
    for action in range(len(ACTION_NAMES)):
        reward, probabilities, posteriors = _point_step(kernel, belief, latent, action)
        branches = {}
        continuation = 0.0
        if horizon > 1:
            for observation, posterior in posteriors.items():
                child = _solve_point(kernel, posterior, latent, horizon - 1, tolerance)
                branches[observation] = child
                continuation += float(probabilities[observation]) * float(child["value"])
        rows.append(
            {
                "action": action,
                "value": float(reward + kernel.discount * continuation),
                "branches": branches,
            }
        )
    selected, optimal, maximum = _select([row["value"] for row in rows], tolerance)
    return {
        "terminal": False,
        "value": maximum,
        "selected_action": selected,
        "optimal_actions": optimal,
        "branches": rows[selected]["branches"],
    }


def _evaluate_policy(kernel: Any, belief: np.ndarray, policy: dict[str, Any], horizon: int) -> float:
    if horizon == 0:
        return 0.0
    action = int(policy["selected_action"])
    reward, probabilities, posteriors = _joint_step(kernel, belief, action)
    continuation = 0.0
    if horizon > 1:
        continuation = sum(
            float(probabilities[observation])
            * _evaluate_policy(kernel, posterior, policy["branches"][observation], horizon - 1)
            for observation, posterior in posteriors.items()
        )
    return float(reward + kernel.discount * continuation)


def _sequence_value(kernel: Any, belief: np.ndarray, actions: Sequence[int]) -> float:
    if not actions:
        return 0.0
    reward, probabilities, posteriors = _joint_step(kernel, belief, int(actions[0]))
    continuation = 0.0
    if len(actions) > 1:
        continuation = sum(
            float(probabilities[observation])
            * _sequence_value(kernel, posterior, actions[1:])
            for observation, posterior in posteriors.items()
        )
    return float(reward + kernel.discount * continuation)


def _myopic_policy(kernel: Any, belief: np.ndarray, horizon: int, tolerance: float):
    if horizon == 0:
        return {"terminal": True, "value": 0.0}
    steps = [_joint_step(kernel, belief, action) for action in range(len(ACTION_NAMES))]
    selected, _, maximum = _select([step[0] for step in steps], tolerance)
    _, _, posteriors = steps[selected]
    return {
        "terminal": False,
        "value": maximum,
        "selected_action": selected,
        "branches": {
            observation: _myopic_policy(kernel, posterior, horizon - 1, tolerance)
            for observation, posterior in posteriors.items()
            if horizon > 1
        },
    }


def main() -> None:
    evaluator_lock_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-evaluator-lock.json"
    config_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-evaluation.json"
    result_path = PROJECT_ROOT / "outputs/v75-active-sensing-confirmation/evaluation/result.json"
    attempt_path = PROJECT_ROOT / "outputs/v75-active-sensing-confirmation/evaluation/attempt.json"
    report_path = PROJECT_ROOT / "docs/v75-active-sensing-confirmation-results.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v75_confirmation_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v75-active-sensing-confirmation/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v75-active-sensing-confirmation-outcome-lock.json"
    if lock_path.exists():
        raise RuntimeError("V75 replication outcome is already frozen")

    evaluator_lock = json.loads(evaluator_lock_path.read_text())
    evaluator_payload = {
        key: value for key, value in evaluator_lock.items() if key != "lock_payload_sha256"
    }
    config = json.loads(config_path.read_text())
    result = json.loads(result_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    errors: list[str] = []
    authorization_ok = bool(
        payload_hash(evaluator_payload) == evaluator_lock["lock_payload_sha256"]
        and evaluator_lock["authorization"]["run_replication_outcomes_once"]
        and file_sha256(result_path)
        == file_sha256(PROJECT_ROOT / evaluator_lock["result_path"])
    )
    if not authorization_ok:
        errors.append("V75 evaluator lock or result binding is invalid")

    one_shot_ok = bool(
        attempt["attempt_number"] == 1
        and attempt["maximum_attempts"] == 1
        and attempt["model_count"] == 1
        and attempt["prior_paint_policy_outcome_access_count"] == 0
        and attempt["candidate_EIG_value_count"] == 0
    )
    if not one_shot_ok:
        errors.append("V75 durable attempt or access firewall is invalid")

    family = build_family()
    kernel = family.kernel
    belief = family.initial_belief
    horizon = int(config["horizonActions"])
    tolerance = float(config["tieTolerance"])
    exact = _solve_joint(kernel, belief, horizon, tolerance)
    masses = belief.sum(axis=1)
    point_rows = []
    posterior_sampling_value = 0.0
    for latent, mass in enumerate(masses):
        state_belief = belief[latent] / mass
        policy = _solve_point(kernel, state_belief, latent, horizon, tolerance)
        exact_environment_value = _evaluate_policy(kernel, belief, policy, horizon)
        posterior_sampling_value += float(mass) * exact_environment_value
        point_rows.append(
            {
                "latent": latent,
                "root_action": ACTION_NAMES[int(policy["selected_action"])],
                "exact_environment_value": exact_environment_value,
            }
        )
    map_value = point_rows[0]["exact_environment_value"]
    sequence_rows = [
        (actions, _sequence_value(kernel, belief, actions))
        for actions in product(range(len(ACTION_NAMES)), repeat=horizon)
    ]
    open_loop_value = max(value for _, value in sequence_rows)
    open_loop_optimal = [
        actions
        for actions, value in sequence_rows
        if open_loop_value - value <= tolerance
    ]
    open_loop_actions = open_loop_optimal[0]
    myopic = _myopic_policy(kernel, belief, horizon, tolerance)
    myopic_value = _evaluate_policy(kernel, belief, myopic, horizon)
    scale = max(
        1.0,
        float(kernel.reward.max() - kernel.reward.min())
        * sum(kernel.discount**depth for depth in range(horizon)),
    )

    archived = result["model"]
    numeric_tolerance = 1e-12
    numeric_checks = {
        "exact_value": abs(float(exact["value"]) - archived["exact"]["value"])
        <= numeric_tolerance,
        "exact_root_q_values": all(
            abs(exact["q_values"][index] - archived["exact"]["root_q_values"][name])
            <= numeric_tolerance
            for index, name in enumerate(ACTION_NAMES)
        ),
        "map_value": abs(map_value - archived["map"]["exact_environment_value"])
        <= numeric_tolerance,
        "posterior_sampling_value": abs(
            posterior_sampling_value
            - archived["posterior_sampling"]["exact_environment_value"]
        )
        <= numeric_tolerance,
        "open_loop_value": abs(open_loop_value - archived["open_loop"]["value"])
        <= numeric_tolerance,
        "myopic_value": abs(myopic_value - archived["myopic"]["exact_environment_value"])
        <= numeric_tolerance,
        "return_scale": abs(scale - archived["return_scale"]) <= numeric_tolerance,
    }
    exact_branches_ok = all(
        ACTION_NAMES[int(child["selected_action"])] == "inspect_target"
        and all(
            ACTION_NAMES[int(grandchild["selected_action"])]
            == ("paint" if beacon == target else "reject")
            for target, grandchild in child["branches"].items()
        )
        for beacon, child in exact["branches"].items()
    )
    structural_checks = {
        "exact_root_action": ACTION_NAMES[int(exact["selected_action"])]
        == archived["exact"]["root_action"],
        "exact_optimal_set": [ACTION_NAMES[index] for index in exact["optimal_actions"]]
        == archived["exact"]["root_optimal_actions"],
        "exact_contingent_structure": exact_branches_ok,
        "both_point_roots_inspect": all(
            row["root_action"] == "inspect_target" for row in point_rows
        ),
        "open_loop_actions": [ACTION_NAMES[index] for index in open_loop_actions]
        == archived["open_loop"]["selected_actions"],
        "myopic_root": ACTION_NAMES[int(myopic["selected_action"])]
        == archived["myopic"]["root_action"],
        "archived_gates_all_pass": result["passed"] and all(result["gates"].values()),
        "common_support_and_zero_fallback": archived["integrity"][
            "point_model_on_support_rate"
        ]
        == 1.0
        and archived["integrity"]["fallback_count"] == 0,
    }
    checks = {
        "locked_one_shot_authorization": authorization_ok,
        "durable_attempt_and_firewalls": one_shot_ok,
        "independent_numeric_reconstruction": all(numeric_checks.values()),
        "independent_policy_and_gate_reconstruction": all(structural_checks.values()),
        "report_exists": report_path.exists(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    audit = {
        "schema_version": "75-active-sensing-confirmation-outcome",
        "experiment": "v75_replication_outcome_independent_audit",
        "passed": not errors and not failed,
        "errors": errors,
        "failed_checks": failed,
        "checks": checks,
        "numeric_checks": numeric_checks,
        "structural_checks": structural_checks,
        "independent_reconstruction": {
            "exact_value": float(exact["value"]),
            "exact_root_action": ACTION_NAMES[int(exact["selected_action"])],
            "exact_root_q_values": dict(zip(ACTION_NAMES, exact["q_values"], strict=True)),
            "point_rows": point_rows,
            "map_value": map_value,
            "posterior_sampling_value": posterior_sampling_value,
            "open_loop_value": open_loop_value,
            "open_loop_actions": [ACTION_NAMES[index] for index in open_loop_actions],
            "myopic_value": myopic_value,
            "return_scale": scale,
        },
        "access": {
            "independent_outcome_audit_count": 1,
            "replication_evaluation_attempt_count": 1,
            "prior_paint_policy_outcome_access_count": 0,
            "candidate_EIG_value_count": 0,
            "human_record_access_count": 0,
            "model_forward_pass_count": 0,
            "adapter_training_run_count": 0,
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "75-active-sensing-confirmation-outcome",
        "experiment": "v75_replication_outcome_lock",
        "evaluator_lock": str(evaluator_lock_path.relative_to(PROJECT_ROOT)),
        "evaluator_lock_sha256": file_sha256(evaluator_lock_path),
        "attempt": str(attempt_path.relative_to(PROJECT_ROOT)),
        "attempt_sha256": file_sha256(attempt_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "outcome_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "outcome_audit_sha256": file_sha256(audit_path),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "report_sha256": file_sha256(report_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "outcome": {
            "passed_all_replication_gates": bool(result["passed"]),
            "exact_value": archived["exact"]["value"],
            "map_value": archived["map"]["exact_environment_value"],
            "posterior_sampling_value": archived["posterior_sampling"][
                "exact_environment_value"
            ],
            "normalized_map_regret": archived["map"]["normalized_regret"],
            "normalized_posterior_sampling_regret": archived[
                "posterior_sampling"
            ]["normalized_regret"],
            "normalized_exact_over_open_loop_advantage": archived["open_loop"][
                "normalized_exact_advantage"
            ],
            "fallback_count": archived["integrity"]["fallback_count"],
            "source_discovery_clean": False,
            "source_policy_outcome_untouched": True,
        },
        "authorization": {
            "modify_or_rerun_V75": False,
            "tune_V75_source_beacon_horizon_controls_or_gates": False,
            "report_and_synthesize_V75": True,
            "design_discovery_clean_successor_after_fresh_preregistration": True,
            "compute_additional_V75_policy_values": False,
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
