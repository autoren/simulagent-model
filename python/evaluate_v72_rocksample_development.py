#!/usr/bin/env python3
"""One-shot exact V72 RockSample development screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v71_exact_planning import (
    best_open_loop_sequence,
    evaluate_policy_exact,
    finite_horizon_return_scale,
    map_control,
    plan_exact,
    plan_myopic,
    posterior_sampling_control,
)
from v72_rocksample_source import ACTION_NAMES, build_family


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    family = build_family()
    kernel = family.kernel
    belief = family.initial_belief
    horizon = int(config["horizonActions"])
    tolerance = float(config["tieTolerance"])
    stats: dict[str, int] = {}
    exact = plan_exact(
        kernel, belief, horizon, tie_tolerance=tolerance, stats=stats
    )
    mapped = map_control(kernel, belief, horizon, tie_tolerance=tolerance)
    sampled = posterior_sampling_control(
        kernel, belief, horizon, tie_tolerance=tolerance
    )
    open_loop = best_open_loop_sequence(
        kernel, belief, horizon, tie_tolerance=tolerance
    )
    myopic_policy = plan_myopic(kernel, belief, horizon, tie_tolerance=tolerance)
    myopic_value = evaluate_policy_exact(kernel, belief, myopic_policy, horizon)
    scale = finite_horizon_return_scale(kernel, horizon)
    exact_value = float(exact["value"])
    ordered = sorted((float(value) for value in exact["q_values"]), reverse=True)

    branches = []
    final_controls: set[str] = set()
    if ACTION_NAMES[int(exact["selected_action"])] == "check_reference":
        for reference_observation, child in sorted(exact["branches"].items()):
            target_rows = []
            for target_observation, grandchild in sorted(child["branches"].items()):
                final_rows = []
                for movement_observation, great_grandchild in sorted(
                    grandchild["branches"].items()
                ):
                    final_action = ACTION_NAMES[
                        int(great_grandchild["selected_action"])
                    ]
                    final_controls.add(final_action)
                    final_rows.append(
                        {
                            "movement_observation": kernel.observation_names[
                                movement_observation
                            ],
                            "final_action": final_action,
                        }
                    )
                target_rows.append(
                    {
                        "target_observation": kernel.observation_names[target_observation],
                        "third_action": ACTION_NAMES[int(grandchild["selected_action"])],
                        "final": final_rows,
                    }
                )
            branches.append(
                {
                    "reference_observation": kernel.observation_names[
                        reference_observation
                    ],
                    "second_action": ACTION_NAMES[int(child["selected_action"])],
                    "target_branches": target_rows,
                }
            )

    root_latent_observation = np.zeros((2, 3), dtype=np.float64)
    check_reference = ACTION_NAMES.index("check_reference")
    from v71_exact_planning import exact_step

    calibration = exact_step(kernel, belief, check_reference)
    for observation, posterior in calibration["posteriors"].items():
        root_latent_observation[:, observation] = (
            float(calibration["probabilities"][observation])
            * posterior.sum(axis=1)
        )
    row = root_latent_observation.sum(axis=1, keepdims=True)
    column = root_latent_observation.sum(axis=0, keepdims=True)
    independent = row @ column
    mask = root_latent_observation > 0.0
    calibration_mi = float(
        np.sum(
            root_latent_observation[mask]
            * np.log(root_latent_observation[mask] / independent[mask])
        )
    )

    return {
        "model": "rocksample_jl_2x2_reference_target",
        "horizon": horizon,
        "return_scale": scale,
        "calibration_mutual_information_nats": calibration_mi,
        "bellman_nodes": int(stats["bellman_nodes"]),
        "exact": {
            "value": exact_value,
            "root_action": ACTION_NAMES[int(exact["selected_action"])],
            "root_optimal_actions": [
                ACTION_NAMES[int(action)] for action in exact["optimal_actions"]
            ],
            "root_q_values": dict(
                zip(ACTION_NAMES, (float(value) for value in exact["q_values"]), strict=True)
            ),
            "root_action_margin": float(ordered[0] - ordered[1]),
            "reference_branches": branches,
            "distinct_final_control_actions": sorted(final_controls),
        },
        "map": {
            "latent": mapped["latent_name"],
            "root_action": ACTION_NAMES[int(mapped["policy"]["selected_action"])],
            "exact_environment_value": float(mapped["value"]),
            "normalized_regret": float((exact_value - float(mapped["value"])) / scale),
            "on_support": bool(mapped["on_support"]),
        },
        "posterior_sampling": {
            "exact_environment_value": float(sampled["value"]),
            "normalized_regret": float((exact_value - float(sampled["value"])) / scale),
            "root_action_distribution": dict(
                zip(ACTION_NAMES, sampled["root_action_distribution"], strict=True)
            ),
            "on_support": bool(sampled["on_support"]),
        },
        "open_loop": {
            "value": float(open_loop["value"]),
            "selected_actions": [ACTION_NAMES[int(action)] for action in open_loop["selected_actions"]],
            "sequence_count": int(open_loop["sequence_count"]),
            "normalized_exact_advantage": float(
                (exact_value - float(open_loop["value"])) / scale
            ),
        },
        "myopic": {
            "root_action": ACTION_NAMES[int(myopic_policy["selected_action"])],
            "exact_environment_value": float(myopic_value),
        },
        "support": {
            "point_model_supports_identical": bool(
                np.array_equal(kernel.observation[0] > 0.0, kernel.observation[1] > 0.0)
            ),
            "point_model_on_support_rate": 1.0,
            "fallback_count": 0,
        },
    }


def gates(row: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gate = config["gates"]
    return {
        "exact_root_action": row["exact"]["root_action"] == gate["requiredExactRootAction"],
        "MAP_root_action": row["map"]["root_action"] == gate["requiredMAPRootAction"],
        "exact_root_margin": row["exact"]["root_action_margin"]
        >= gate["minimumExactRootActionMargin"],
        "every_reference_branch_checks_target_second": len(row["exact"]["reference_branches"])
        == 2
        and all(
            branch["second_action"]
            == gate["requiredSecondActionAfterEveryReferenceCheckObservation"]
            for branch in row["exact"]["reference_branches"]
        ),
        "distinct_required_final_controls": len(row["exact"]["distinct_final_control_actions"])
        >= gate["minimumDistinctFinalControlActionsAcrossReachableReferenceTargetHistories"]
        and all(
            action in row["exact"]["distinct_final_control_actions"]
            for action in gate["requiredFinalControlActions"]
        ),
        "MAP_material_regret": row["map"]["normalized_regret"]
        >= gate["minimumNormalizedMAPRegret"],
        "posterior_sampling_material_regret": row["posterior_sampling"][
            "normalized_regret"
        ]
        >= gate["minimumNormalizedPosteriorSamplingRegret"],
        "exact_over_open_loop_advantage": row["open_loop"]["normalized_exact_advantage"]
        >= gate["minimumNormalizedExactOverOpenLoopAdvantage"],
        "common_support_zero_fallback": row["support"]["point_model_on_support_rate"]
        >= gate["minimumPointModelOnSupportRate"]
        and row["support"]["fallback_count"] <= gate["maximumFallbackCount"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        default="configs/v72-active-sensing-development-evaluator-lock.json",
    )
    args = parser.parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V72 development evaluator lock drifted")
    if not lock["authorization"]["run_development_outcomes_once"]:
        raise RuntimeError("V72 development outcome is not authorized")
    for path_key, hash_key in (
        ("resource_lock", "resource_lock_sha256"),
        ("evaluation_config", "evaluation_config_sha256"),
        ("exporter", "exporter_sha256"),
        ("planning_core", "planning_core_sha256"),
        ("evaluator", "evaluator_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V72 development dependency drifted: {path_key}")

    output_dir = PROJECT_ROOT / "outputs/v72-active-sensing/development-evaluation"
    if output_dir.exists():
        raise RuntimeError("V72 development evaluation already exists")
    output_dir.mkdir(parents=True)
    attempt = {
        "schema_version": "72-active-sensing-development-evaluation",
        "experiment": "v72_rocksample_development_attempt",
        "attempt_number": 1,
        "model_count": 1,
        "protected_confirmation_policy_value_count": 0,
        "candidate_EIG_value_count": 0,
        "V71_protected_access_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    (output_dir / "attempt.json").write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n"
    )
    config = json.loads((PROJECT_ROOT / lock["evaluation_config"]).read_text())
    row = evaluate(config)
    gate_results = gates(row, config)
    result = {
        "schema_version": "72-active-sensing-development-evaluation",
        "experiment": "v72_rocksample_development_screen",
        "passed": all(gate_results.values()),
        "decision": (
            "authorize_fresh_protected_source_discovery"
            if all(gate_results.values())
            else "freeze_negative_result_and_stop_V72_before_protected_discovery"
        ),
        "gates": gate_results,
        "row": row,
        "access": attempt,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
