#!/usr/bin/env python3
"""Durable one-shot V79 evaluator with explicit unresolved terminal utility."""
from __future__ import annotations

import argparse
import json

import numpy as np

import evaluate_v78_clarification_benchmark as v78_evaluator
from locked_census_harness import run_locked_census_once
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v78_clarification_benchmark import build_fixture
from v79_terminal_utility_planning import (
    ACTIVE_UNRESOLVED_TERMINAL_UTILITY,
    TERMINAL_STATE_TERMINAL_UTILITY,
    best_open_loop_sequence,
    evaluate_policy_exact,
    map_control,
    oracle_interpretation_value,
    plan_exact,
    posterior_sampling_control,
    terminal_utility,
)


def install_terminal_utility_planner() -> None:
    """Replace only the planning/evaluation functions used by the V78 reporter."""
    v78_evaluator.plan_exact = plan_exact
    v78_evaluator.evaluate_policy_exact = evaluate_policy_exact
    v78_evaluator.map_control = map_control
    v78_evaluator.posterior_sampling_control = posterior_sampling_control
    v78_evaluator.best_open_loop_sequence = best_open_loop_sequence
    v78_evaluator.oracle_interpretation_value = oracle_interpretation_value


def evaluate_fixture(row: dict, config: dict) -> dict:
    install_terminal_utility_planner()
    result = v78_evaluator.evaluate_fixture(row, config)
    fixture = build_fixture(config, row["name"])
    terminal = np.zeros_like(fixture.initial_belief)
    terminal[:, 1] = fixture.initial_belief.sum(axis=1)
    result["claim_boundary"] = (
        "targeted model-free successor with explicit unresolved terminal utility"
    )
    result["terminal_utility"] = {
        "configured_active_unresolved": ACTIVE_UNRESOLVED_TERMINAL_UTILITY,
        "configured_terminal_state": TERMINAL_STATE_TERMINAL_UTILITY,
        "initial_active_belief_at_horizon_zero": terminal_utility(
            fixture.kernel, fixture.initial_belief
        ),
        "matched_terminal_belief_at_horizon_zero": terminal_utility(
            fixture.kernel, terminal
        ),
        "exact_policy_replay_agrees": abs(
            result["exact"]["value"] - result["exact"]["replay_value"]
        )
        <= 1e-10,
    }
    return result


def evaluate_gates(fixtures: dict, config: dict, access: dict) -> dict[str, bool]:
    gates = v78_evaluator.evaluate_gates(fixtures, config, access)
    additional = config["additionalGates"]
    gates.update(
        {
            "active_unresolved_terminal_utility_is_explicit": all(
                row["terminal_utility"]["initial_active_belief_at_horizon_zero"]
                == additional["requiredActiveUnresolvedTerminalUtility"]
                for row in fixtures.values()
            ),
            "terminal_state_terminal_utility_is_zero": all(
                row["terminal_utility"]["matched_terminal_belief_at_horizon_zero"]
                == additional["requiredTerminalStateTerminalUtility"]
                for row in fixtures.values()
            ),
            "terminal_utility_replay_agreement": all(
                row["terminal_utility"]["exact_policy_replay_agrees"]
                for row in fixtures.values()
            ),
        }
    )
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock", default="configs/v79-terminal-utility-implementation-lock.json"
    )
    args = parser.parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if v78_evaluator.payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V79 implementation lock payload drifted")
    if not lock["authorization"]["run_model_free_census_once"]:
        raise RuntimeError("V79 implementation lock does not authorize the census")
    for path_key, hash_key in (
        ("design_lock", "design_lock_sha256"),
        ("terminal_planner", "terminal_planner_sha256"),
        ("evaluator", "evaluator_sha256"),
        ("parent_reporter", "parent_reporter_sha256"),
        ("census_harness", "census_harness_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V79 locked dependency drifted: {path_key}")

    config = lock["resolved_config_payload"]
    access = {
        "attempt_number": 1,
        "model_forward_pass_count": 0,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
        "real_tool_call_count": 0,
        "external_side_effect_count": 0,
        "real_message_or_file_send_count": 0,
    }
    result = run_locked_census_once(
        output_dir=(
            PROJECT_ROOT
            / "outputs/v79-structured-llm-interface/model-free-evaluation"
        ),
        attempt=access,
        fixture_rows=config["fixtures"],
        evaluate_fixture=lambda row: evaluate_fixture(row, config),
        evaluate_gates=lambda fixtures: evaluate_gates(fixtures, config, access),
        result_metadata={
            "schema_version": "79-terminal-utility-benchmark-outcome",
            "experiment": "v79_model_free_terminal_utility_census",
            "claim_boundary": (
                "targeted project-authored model-free successor; not language-model, "
                "human, external-benchmark, real-tool, or safety evidence"
            ),
            "inherited_V78_task_design_changed": False,
            "explicit_terminal_utility_added": True,
        },
        pass_decision=(
            "freeze_terminal_utility_mechanism_and_authorize_local_model_protocol_preregistration"
        ),
        fail_decision=(
            "freeze_terminal_utility_successor_failure_without_parameter_tuning"
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
