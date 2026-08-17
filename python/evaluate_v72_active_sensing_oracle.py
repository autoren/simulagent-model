#!/usr/bin/env python3
"""Run the one-shot V72 engineered active-sensing mechanism oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
from v72_active_sensing_oracles import (
    ACTION_NAMES,
    build_oracle,
    structural_diagnostics,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _named_actions(indices: list[int] | tuple[int, ...]) -> list[str]:
    return [ACTION_NAMES[int(index)] for index in indices]


def evaluate_fixture(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    fixture = build_oracle(kind, reliability=config["sharedParameters"]["sensorReliability"])
    horizon = int(config["sharedParameters"]["horizonActions"])
    tolerance = float(config["sharedParameters"]["tieTolerance"])
    exact = plan_exact(fixture.kernel, fixture.initial_belief, horizon, tie_tolerance=tolerance)
    mapped = map_control(fixture.kernel, fixture.initial_belief, horizon, tie_tolerance=tolerance)
    sampled = posterior_sampling_control(
        fixture.kernel, fixture.initial_belief, horizon, tie_tolerance=tolerance
    )
    open_loop = best_open_loop_sequence(
        fixture.kernel, fixture.initial_belief, horizon, tie_tolerance=tolerance
    )
    myopic_policy = plan_myopic(
        fixture.kernel, fixture.initial_belief, horizon, tie_tolerance=tolerance
    )
    myopic_value = evaluate_policy_exact(
        fixture.kernel, fixture.initial_belief, myopic_policy, horizon
    )
    scale = finite_horizon_return_scale(fixture.kernel, horizon)
    q_values = [float(value) for value in exact["q_values"]]
    ordered = sorted(q_values, reverse=True)

    calibration_branches = []
    terminal_repairs: set[str] = set()
    if ACTION_NAMES[int(exact["selected_action"])] == "calibrate":
        for observation, child in sorted(exact["branches"].items()):
            terminal_actions = []
            for second_observation, grandchild in sorted(child["branches"].items()):
                name = ACTION_NAMES[int(grandchild["selected_action"])]
                terminal_actions.append(
                    {
                        "observation": fixture.kernel.observation_names[second_observation],
                        "action": name,
                    }
                )
                if name.startswith("repair_"):
                    terminal_repairs.add(name)
            calibration_branches.append(
                {
                    "calibration_observation": fixture.kernel.observation_names[observation],
                    "second_action": ACTION_NAMES[int(child["selected_action"])],
                    "terminal_actions": terminal_actions,
                }
            )

    exact_value = float(exact["value"])
    return {
        "name": fixture.name,
        "kind": kind,
        "structural": structural_diagnostics(fixture),
        "return_scale": scale,
        "exact": {
            "value": exact_value,
            "root_action": ACTION_NAMES[int(exact["selected_action"])],
            "root_optimal_actions": _named_actions(exact["optimal_actions"]),
            "root_q_values": dict(zip(ACTION_NAMES, q_values, strict=True)),
            "root_action_margin": float(ordered[0] - ordered[1]),
            "calibration_branches": calibration_branches,
            "distinct_terminal_repair_actions": sorted(terminal_repairs),
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
            "selected_actions": _named_actions(open_loop["selected_actions"]),
            "sequence_count": int(open_loop["sequence_count"]),
        },
        "myopic": {
            "root_action": ACTION_NAMES[int(myopic_policy["selected_action"])],
            "exact_environment_value": float(myopic_value),
        },
    }


def oracle_gates(rows: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, bool]:
    positive = rows["positive"]
    negative = rows["negative_control"]
    gates = config["oracleGates"]
    return {
        "positive_calibration_information": positive["structural"][
            "calibration_mutual_information_nats"
        ]
        >= gates["minimumPositiveCalibrationMutualInformationNats"],
        "positive_inspection_state_information": positive["structural"][
            "inspection_state_mutual_information_given_codebook_nats"
        ]
        >= gates["minimumPositiveInspectionStateMutualInformationGivenCodebookNats"],
        "positive_exact_root_action": positive["exact"]["root_action"]
        == gates["requiredPositiveExactRootAction"],
        "positive_map_root_action": positive["map"]["root_action"]
        == gates["requiredPositiveMAPRootAction"],
        "positive_exact_root_margin": positive["exact"]["root_action_margin"]
        >= gates["minimumPositiveExactRootActionMargin"],
        "positive_every_calibration_branch_inspects_second": all(
            row["second_action"]
            == gates["requiredPositiveSecondActionAfterEveryCalibrationObservation"]
            for row in positive["exact"]["calibration_branches"]
        )
        and len(positive["exact"]["calibration_branches"]) == 2,
        "positive_distinct_terminal_repairs": len(
            positive["exact"]["distinct_terminal_repair_actions"]
        )
        >= gates[
            "minimumDistinctPositiveTerminalRepairActionsAcrossReachableHistories"
        ],
        "positive_map_regret": positive["map"]["normalized_regret"]
        >= gates["minimumPositiveNormalizedMAPRegret"],
        "positive_posterior_sampling_regret": positive["posterior_sampling"][
            "normalized_regret"
        ]
        >= gates["minimumPositiveNormalizedPosteriorSamplingRegret"],
        "negative_map_regret": negative["map"]["normalized_regret"]
        <= gates["maximumNegativeControlNormalizedMAPRegret"],
        "negative_posterior_sampling_regret": negative["posterior_sampling"][
            "normalized_regret"
        ]
        <= gates["maximumNegativeControlNormalizedPosteriorSamplingRegret"],
        "negative_exact_map_root_agreement": negative["exact"]["root_action"]
        == negative["map"]["root_action"],
        "shared_support_and_no_fallback": all(
            row["structural"]["point_model_on_support_rate"]
            >= gates["minimumPointModelOnSupportRate"]
            and row["structural"]["fallback_count"] <= gates["maximumFallbackCount"]
            for row in rows.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        default="configs/v72-active-sensing-oracle-evaluator-lock.json",
    )
    args = parser.parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V72 evaluator lock payload drifted")
    if not lock["authorization"]["run_engineered_oracle_outcomes_once"]:
        raise RuntimeError("V72 evaluator lock does not authorize the oracle run")
    for path_key, hash_key in (
        ("design_lock", "design_lock_sha256"),
        ("fixture_core", "fixture_core_sha256"),
        ("planning_core", "planning_core_sha256"),
        ("evaluator", "evaluator_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V72 locked dependency drifted: {path_key}")

    output_dir = PROJECT_ROOT / "outputs/v72-active-sensing/oracle-evaluation"
    if output_dir.exists():
        raise RuntimeError("V72 oracle evaluation already exists")
    output_dir.mkdir(parents=True)
    attempt_path = output_dir / "attempt.json"
    attempt = {
        "schema_version": "72-active-sensing-oracle",
        "experiment": "v72_active_sensing_oracle_attempt",
        "attempt_number": 1,
        "fixture_count": 2,
        "external_candidate_metadata_records_read": 0,
        "external_candidate_policy_values_computed": 0,
        "V71_protected_access_count": 0,
        "human_record_access_count": 0,
        "model_forward_pass_count": 0,
        "adapter_training_run_count": 0,
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")

    design_lock = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())
    config = design_lock["config_payload"]
    rows = {
        "positive": evaluate_fixture("positive", config),
        "negative_control": evaluate_fixture("negative_control", config),
    }
    gates = oracle_gates(rows, config)
    result = {
        "schema_version": "72-active-sensing-oracle",
        "experiment": "v72_shared_support_active_sensing_oracle",
        "claim_boundary": "engineered mechanism oracle only; not scientific evidence",
        "passed": all(gates.values()),
        "decision": (
            "freeze_mechanism_and_authorize_metadata_only_external_source_discovery"
            if all(gates.values())
            else "repair_engineered_fixture_without_external_candidate_outcomes"
        ),
        "gates": gates,
        "fixtures": rows,
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
