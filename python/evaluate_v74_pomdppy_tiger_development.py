#!/usr/bin/env python3
"""Run the one authorized V74 pomdp-py Tiger development evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

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
from v74_pomdppy_tiger_source import (
    ACTION_NAMES,
    OBSERVATION_NAMES,
    build_family,
    structural_diagnostics,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def named_actions(indices: list[int] | tuple[int, ...]) -> list[str]:
    return [ACTION_NAMES[int(index)] for index in indices]


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    family = build_family()
    kernel = family.kernel
    horizon = int(config["horizonActions"])
    tolerance = float(config["tieTolerance"])
    exact_stats: dict[str, int] = {}
    exact = plan_exact(
        kernel,
        family.initial_belief,
        horizon,
        tie_tolerance=tolerance,
        stats=exact_stats,
    )
    mapped = map_control(
        kernel, family.initial_belief, horizon, tie_tolerance=tolerance
    )
    sampled = posterior_sampling_control(
        kernel, family.initial_belief, horizon, tie_tolerance=tolerance
    )
    open_loop = best_open_loop_sequence(
        kernel, family.initial_belief, horizon, tie_tolerance=tolerance
    )
    myopic_policy = plan_myopic(
        kernel, family.initial_belief, horizon, tie_tolerance=tolerance
    )
    myopic_value = evaluate_policy_exact(
        kernel, family.initial_belief, myopic_policy, horizon
    )
    scale = finite_horizon_return_scale(kernel, horizon)
    exact_value = float(exact["value"])
    q_values = [float(value) for value in exact["q_values"]]
    ordered = sorted(q_values, reverse=True)

    calibration_branches = []
    final_controls: set[str] = set()
    final_mapping_ok = True
    for beacon_observation, child in sorted(exact["branches"].items()):
        terminal = []
        for target_observation, grandchild in sorted(child["branches"].items()):
            action_name = ACTION_NAMES[int(grandchild["selected_action"])]
            final_controls.add(action_name)
            expected = (
                "open_right"
                if beacon_observation == target_observation
                else "open_left"
            )
            final_mapping_ok = final_mapping_ok and action_name == expected
            terminal.append(
                {
                    "target_observation": OBSERVATION_NAMES[target_observation],
                    "action": action_name,
                    "registered_expected_action": expected,
                }
            )
        calibration_branches.append(
            {
                "beacon_observation": OBSERVATION_NAMES[beacon_observation],
                "second_action": ACTION_NAMES[int(child["selected_action"])],
                "terminal_actions": terminal,
            }
        )

    map_value = float(mapped["value"])
    sampled_value = float(sampled["value"])
    open_loop_value = float(open_loop["value"])
    map_regret = (exact_value - map_value) / scale
    sampled_regret = (exact_value - sampled_value) / scale
    open_loop_advantage = (exact_value - open_loop_value) / scale
    myopic_regret = (exact_value - myopic_value) / scale
    return {
        "name": "pomdp_py_tiger_high_fidelity_codebook",
        "structural": structural_diagnostics(),
        "return_scale": scale,
        "exact": {
            "value": exact_value,
            "root_action": ACTION_NAMES[int(exact["selected_action"])],
            "root_optimal_actions": named_actions(exact["optimal_actions"]),
            "root_q_values": dict(zip(ACTION_NAMES, q_values, strict=True)),
            "root_action_margin": float(ordered[0] - ordered[1]),
            "normalized_root_action_margin": float((ordered[0] - ordered[1]) / scale),
            "calibration_branches": calibration_branches,
            "distinct_final_control_actions": sorted(final_controls),
            "registered_final_mapping_holds": final_mapping_ok,
            "bellman_nodes": int(exact_stats["bellman_nodes"]),
        },
        "map": {
            "latent": mapped["latent_name"],
            "root_action": ACTION_NAMES[int(mapped["policy"]["selected_action"])],
            "exact_environment_value": map_value,
            "normalized_regret": float(map_regret),
            "on_support": bool(mapped["on_support"]),
        },
        "posterior_sampling": {
            "exact_environment_value": sampled_value,
            "normalized_regret": float(sampled_regret),
            "root_action_distribution": dict(
                zip(ACTION_NAMES, sampled["root_action_distribution"], strict=True)
            ),
            "on_support": bool(sampled["on_support"]),
            "sampled_model_persists_for_full_policy": bool(
                sampled["sampled_model_persists_for_full_policy"]
            ),
        },
        "open_loop": {
            "value": open_loop_value,
            "selected_actions": named_actions(open_loop["selected_actions"]),
            "sequence_count": int(open_loop["sequence_count"]),
            "normalized_exact_advantage": float(open_loop_advantage),
        },
        "myopic": {
            "root_action": ACTION_NAMES[int(myopic_policy["selected_action"])],
            "exact_environment_value": float(myopic_value),
            "normalized_regret": float(myopic_regret),
        },
        "integrity": {
            "point_model_on_support_rate": float(
                (int(mapped["on_support"]) + int(sampled["on_support"])) / 2.0
            ),
            "fallback_count": 0,
            "all_reported_values_finite": all(
                value == value and abs(value) != float("inf")
                for value in (
                    exact_value,
                    map_value,
                    sampled_value,
                    open_loop_value,
                    myopic_value,
                )
            ),
        },
    }


def evaluate_gates(row: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["gates"]
    ps_distribution = row["posterior_sampling"]["root_action_distribution"]
    return {
        "exact_value_positive": row["exact"]["value"] > gates["minimumExactValue"],
        "exact_root_action": row["exact"]["root_action"]
        == gates["requiredExactRootAction"],
        "exact_root_optimal_set": row["exact"]["root_optimal_actions"]
        == gates["requiredExactRootOptimalActions"],
        "exact_root_margin": row["exact"]["root_action_margin"]
        >= gates["minimumExactRootActionMargin"],
        "exact_normalized_root_margin": row["exact"]["normalized_root_action_margin"]
        >= gates["minimumNormalizedExactRootActionMargin"],
        "map_root_action": row["map"]["root_action"]
        == gates["requiredMAPRootAction"],
        "posterior_sampling_root_action": ps_distribution[
            gates["requiredPosteriorSamplingRootAction"]
        ]
        >= 1.0 - config["tieTolerance"],
        "every_calibration_branch_listens_to_target": len(
            row["exact"]["calibration_branches"]
        )
        == 2
        and all(
            branch["second_action"]
            == gates["requiredSecondActionAfterEveryCalibrationObservation"]
            for branch in row["exact"]["calibration_branches"]
        ),
        "both_final_controls_reachable": len(
            row["exact"]["distinct_final_control_actions"]
        )
        >= gates["minimumDistinctFinalControlActionsAcrossReachablePairedLabels"]
        and row["exact"]["distinct_final_control_actions"]
        == sorted(gates["requiredFinalControlActions"]),
        "registered_final_mapping": bool(
            row["exact"]["registered_final_mapping_holds"]
        ),
        "map_regret_material": row["map"]["normalized_regret"]
        >= gates["minimumNormalizedMAPRegret"],
        "posterior_sampling_regret_material": row["posterior_sampling"][
            "normalized_regret"
        ]
        >= gates["minimumNormalizedPosteriorSamplingRegret"],
        "exact_over_open_loop_material": row["open_loop"][
            "normalized_exact_advantage"
        ]
        >= gates["minimumNormalizedExactOverOpenLoopAdvantage"],
        "myopic_regret_material": row["myopic"]["normalized_regret"]
        >= gates["minimumNormalizedMyopicRegret"],
        "common_support_no_fallback": row["integrity"][
            "point_model_on_support_rate"
        ]
        >= gates["minimumPointModelOnSupportRate"]
        and row["integrity"]["fallback_count"] <= gates["maximumFallbackCount"],
        "all_values_finite": bool(row["integrity"]["all_reported_values_finite"]),
        "protected_and_EIG_firewalls": bool(
            gates["maximumProtectedConfirmationPolicyValueCount"] == 0
            and gates["maximumPriorProtectedAccessCount"] == 0
            and gates["maximumCandidateEIGValueCount"] == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        default="configs/v74-active-sensing-development-evaluator-lock.json",
    )
    args = parser.parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V74 evaluator lock payload drifted")
    if not lock["authorization"]["run_development_outcomes_once"]:
        raise RuntimeError("V74 evaluator lock does not authorize the one-shot run")
    for path_key, hash_key in (
        ("structural_lock", "structural_lock_sha256"),
        ("evaluation_config", "evaluation_config_sha256"),
        ("exporter", "exporter_sha256"),
        ("planning_core", "planning_core_sha256"),
        ("evaluator", "evaluator_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V74 locked dependency drifted: {path_key}")

    output_dir = PROJECT_ROOT / "outputs/v74-active-sensing/development-evaluation"
    if output_dir.exists():
        raise RuntimeError("V74 development evaluation already exists")
    output_dir.mkdir(parents=True)
    attempt = {
        "schema_version": "74-active-sensing-development-attempt",
        "experiment": "v74_pomdppy_tiger_development_attempt",
        "attempt_number": 1,
        "maximum_attempts": 1,
        "model_count": 1,
        "protected_confirmation_policy_value_count": 0,
        "prior_protected_access_count": 0,
        "candidate_EIG_value_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    (output_dir / "attempt.json").write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n"
    )

    config = json.loads((PROJECT_ROOT / lock["evaluation_config"]).read_text())
    row = evaluate(config)
    gates = evaluate_gates(row, config)
    result = {
        "schema_version": "74-active-sensing-development-evaluation",
        "experiment": "v74_pomdppy_tiger_source_grounded_development",
        "passed": all(gates.values()),
        "decision": (
            "freeze_positive_development_result_and_authorize_confirmation_design_only"
            if all(gates.values())
            else "freeze_negative_result_and_stop_V74_before_protected_discovery"
        ),
        "claim_boundary": "source-grounded project-authored development adapter; not an unchanged external environment or confirmation result",
        "gates": gates,
        "model": row,
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
