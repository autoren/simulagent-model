#!/usr/bin/env python3
"""Durable one-shot evaluator for the fresh V78 clarification benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Iterable

from locked_census_harness import (
    named_structural_resources,
    run_locked_census_once,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v77_clarification_benchmark import (
    act_immediately_policy,
    ask_always_policy,
    best_open_loop_sequence,
    evaluate_policy_exact,
    finite_horizon_return_scale,
    map_control,
    oracle_interpretation_value,
    plan_exact,
    posterior_sampling_control,
)
from v77r1_execution_repair import complete_terminal_branches
from v78_clarification_benchmark import (
    ACTION_NAMES,
    EXECUTION_ACTIONS,
    HYPOTHESIS_NAMES,
    INFORMATION_ACTIONS,
    NONE_HYPOTHESIS,
    OBSERVATION_NAMES,
    build_fixture,
    structural_diagnostics,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def policy_nodes(
    policy: dict[str, Any], history: tuple[int, ...] = ()
) -> Iterable[dict[str, Any]]:
    if policy.get("terminal"):
        return
    yield {
        "history": history,
        "action": int(policy["selected_action"]),
        "hypothesis_masses": policy.get("hypothesis_masses"),
    }
    for observation, child in sorted(policy.get("branches", {}).items()):
        yield from policy_nodes(child, history + (int(observation),))


def named_policy_nodes(policy: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "history": [OBSERVATION_NAMES[index] for index in row["history"]],
            "action": ACTION_NAMES[row["action"]],
            "hypothesis_masses": row["hypothesis_masses"],
        }
        for row in policy_nodes(policy)
    ]


def control_summary(kernel, belief, policy: dict[str, Any], horizon: int) -> dict[str, Any]:
    completed = complete_terminal_branches(kernel, belief, policy, horizon)
    violations: list[dict[str, Any]] = []
    value = evaluate_policy_exact(
        kernel,
        belief,
        completed,
        horizon,
        certificate_violations=violations,
    )
    return {
        "value": float(value),
        "root_action": kernel.action_names[int(policy["selected_action"])],
        "complete_belief_certificate_violation_count": len(violations),
        "shadow_only": True,
    }


def evaluate_fixture(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    fixture = build_fixture(config, row["name"])
    kernel = fixture.kernel
    belief = fixture.initial_belief
    horizon = int(config["sharedParameters"]["horizonActions"])
    tolerance = float(config["sharedParameters"]["tieTolerance"])
    stats: dict[str, int] = {}
    exact = plan_exact(
        kernel,
        belief,
        horizon,
        tie_tolerance=tolerance,
        stats=stats,
    )
    exact_violations: list[dict[str, Any]] = []
    exact_replay_value = evaluate_policy_exact(
        kernel,
        belief,
        exact,
        horizon,
        certificate_violations=exact_violations,
    )
    mapped = map_control(kernel, belief, horizon, tie_tolerance=tolerance)
    sampled = posterior_sampling_control(
        kernel, belief, horizon, tie_tolerance=tolerance
    )
    immediate = control_summary(
        kernel,
        belief,
        act_immediately_policy(kernel, belief, horizon, tie_tolerance=tolerance),
        horizon,
    )
    ask = control_summary(
        kernel,
        belief,
        ask_always_policy(kernel, belief, horizon, tie_tolerance=tolerance),
        horizon,
    )
    open_loop = best_open_loop_sequence(
        kernel, belief, horizon, tie_tolerance=tolerance
    )
    oracle = oracle_interpretation_value(
        kernel, belief, horizon, tie_tolerance=tolerance
    )
    scale = finite_horizon_return_scale(kernel, horizon)
    exact_value = float(exact["value"])
    nodes = named_policy_nodes(exact)
    information_actions = sorted(
        {
            node["action"]
            for node in nodes
            if ACTION_NAMES.index(node["action"]) in INFORMATION_ACTIONS
        }
    )
    unknown_nodes = [
        node
        for node in nodes
        if any(observation.endswith("_other") for observation in node["history"])
        and node["hypothesis_masses"] is not None
        and float(node["hypothesis_masses"][NONE_HYPOTHESIS])
        > float(belief[NONE_HYPOTHESIS].sum())
    ]
    safe_unknown = [
        node for node in unknown_nodes if node["action"] in ("safe_preview", "abstain")
    ]
    unsafe_unknown = [
        node
        for node in unknown_nodes
        if ACTION_NAMES.index(node["action"]) in EXECUTION_ACTIONS
    ]
    map_value = float(mapped["value"])
    sampled_value = float(sampled["value"])
    q_values = {
        ACTION_NAMES[int(action)]: float(value)
        for action, value in exact["q_values"].items()
    }
    return {
        "name": fixture.name,
        "synthetic_instruction": fixture.synthetic_instruction,
        "reward_profile": fixture.reward_profile,
        "claim_boundary": "project-authored fresh model-free development mechanism",
        "structural": structural_diagnostics(fixture),
        "resource": {
            "bellman_nodes": int(stats["bellman_nodes"]),
            "belief_checks": int(stats["belief_checks"]),
            "normalized_belief_checks": int(stats["normalized_belief_checks"]),
            "belief_normalization_rate": float(
                stats["normalized_belief_checks"] / stats["belief_checks"]
            ),
            "open_loop_sequence_count": int(open_loop["sequence_count"]),
        },
        "return_scale": scale,
        "exact": {
            "value": exact_value,
            "replay_value": float(exact_replay_value),
            "root_action": ACTION_NAMES[int(exact["selected_action"])],
            "root_optimal_actions": [
                ACTION_NAMES[int(action)] for action in exact["optimal_actions"]
            ],
            "root_q_values": q_values,
            "policy_nodes": nodes,
            "reachable_information_actions": information_actions,
            "unknown_indicating_node_count": len(unknown_nodes),
            "safe_unknown_continuation_count": len(safe_unknown),
            "unknown_branch_irreversible_execution_count": len(unsafe_unknown),
            "maximum_none_posterior_on_unknown_history": max(
                (
                    float(node["hypothesis_masses"][NONE_HYPOTHESIS])
                    for node in unknown_nodes
                ),
                default=float(belief[NONE_HYPOTHESIS].sum()),
            ),
            "complete_belief_certificate_violation_count": len(exact_violations),
        },
        "map": {
            "value": map_value,
            "normalized_regret": float((exact_value - map_value) / scale),
            "root_action": ACTION_NAMES[int(mapped["policy"]["selected_action"])],
            "point_hypothesis": mapped["hypothesis_name"],
            "point_hypothesis_mass": float(mapped["hypothesis_mass"]),
            "complete_belief_certificate_violation_count": len(
                mapped["complete_belief_certificate_violations"]
            ),
            "off_support_fallback_count": int(mapped["off_support_fallback_count"]),
            "shadow_only": True,
        },
        "posterior_sampling": {
            "value": sampled_value,
            "normalized_regret": float((exact_value - sampled_value) / scale),
            "root_action_distribution": {
                ACTION_NAMES[action]: float(probability)
                for action, probability in enumerate(sampled["root_action_distribution"])
                if probability > 0.0
            },
            "complete_belief_certificate_violation_count": len(
                sampled["complete_belief_certificate_violations"]
            ),
            "off_support_fallback_count": int(sampled["off_support_fallback_count"]),
            "shadow_only": True,
        },
        "act_immediately": {
            **immediate,
            "normalized_regret": float((exact_value - immediate["value"]) / scale),
        },
        "ask_always": {
            **ask,
            "normalized_regret": float((exact_value - ask["value"]) / scale),
        },
        "best_open_loop": {
            "value": float(open_loop["value"]),
            "normalized_regret": float(
                (exact_value - float(open_loop["value"])) / scale
            ),
            "selected_actions": [
                ACTION_NAMES[int(action)] for action in open_loop["selected_actions"]
            ],
            "sequence_count": int(open_loop["sequence_count"]),
        },
        "oracle_interpretation": {
            "value": float(oracle["value"]),
            "advantage_over_exact": float(float(oracle["value"]) - exact_value),
            "rows": [
                {
                    **oracle_row,
                    "root_action": ACTION_NAMES[int(oracle_row["root_action"])],
                }
                for oracle_row in oracle["hypotheses"]
            ],
        },
    }


def evaluate_gates(
    fixtures: dict[str, dict[str, Any]],
    config: dict[str, Any],
    access: dict[str, int],
) -> dict[str, bool]:
    gates = config["benchmarkGates"]
    ambiguous = fixtures["ambiguous_tool_intent"]
    clear = fixtures["clear_tool_intent"]
    unknown = fixtures["unknown_heavy_tool_intent"]
    dominant = fixtures["dominant_safe_preview"]
    named_rows = named_structural_resources(fixtures)
    return {
        "complete_fixture_and_hypothesis_census": bool(
            len(fixtures) == gates["requiredFixtureCount"]
            and all(
                row["structural"]["hypothesis_count"]
                == gates["requiredHypothesisCount"]
                for row in named_rows
            )
        ),
        "ambiguous_begins_with_focused_clarification": ambiguous["exact"]["root_action"]
        in gates["requiredAmbiguousRootActionSet"],
        "ambiguous_uses_both_focused_information_actions": all(
            action in ambiguous["exact"]["reachable_information_actions"]
            for action in gates["requiredAmbiguousReachableInformationActions"]
        ),
        "ambiguous_MAP_regret_is_material": ambiguous["map"]["normalized_regret"]
        >= gates["minimumAmbiguousNormalizedMAPRegret"],
        "ambiguous_act_immediately_regret_is_material": ambiguous["act_immediately"][
            "normalized_regret"
        ]
        >= gates["minimumAmbiguousNormalizedActImmediatelyRegret"],
        "clear_executes_immediately": clear["exact"]["root_action"]
        == gates["requiredClearRootAction"],
        "clear_ask_always_regret_is_material": clear["ask_always"]["normalized_regret"]
        >= gates["minimumClearNormalizedAskAlwaysRegret"],
        "unknown_begins_with_clarification": unknown["exact"]["root_action"]
        in gates["requiredUnknownRootActionSet"],
        "unknown_has_safe_unknown_continuation": bool(
            unknown["exact"]["safe_unknown_continuation_count"] > 0
        ),
        "unknown_never_executes_on_unknown_indicating_branch": unknown["exact"][
            "unknown_branch_irreversible_execution_count"
        ]
        <= gates["maximumUnknownBranchIrreversibleExecutionCount"],
        "dominant_safe_preview_is_immediate": dominant["exact"]["root_action"]
        == gates["requiredDominantRootAction"],
        "dominant_MAP_regret_is_zero": dominant["map"]["normalized_regret"]
        <= gates["maximumDominantNormalizedMAPRegret"],
        "dominant_posterior_sampling_regret_is_zero": dominant[
            "posterior_sampling"
        ]["normalized_regret"]
        <= gates["maximumDominantNormalizedPosteriorSamplingRegret"],
        "all_transition_rows_normalize": all(
            row["structural"]["transition_normalization_rate"]
            >= gates["minimumTransitionNormalizationRate"]
            for row in named_rows
        ),
        "all_observation_rows_normalize": all(
            row["structural"]["observation_normalization_rate"]
            >= gates["minimumObservationNormalizationRate"]
            for row in named_rows
        ),
        "all_hypothesis_supports_are_identical": all(
            row["structural"]["identical_hypothesis_support_rate"]
            >= gates["minimumIdenticalHypothesisSupportRate"]
            for row in named_rows
        ),
        "all_initial_and_reachable_beliefs_normalize": all(
            row["structural"]["belief_normalizes"]
            and row["resource"]["belief_normalization_rate"]
            >= gates["minimumBeliefNormalizationRate"]
            for row in named_rows
        ),
        "exact_policy_has_zero_execution_certificate_violations": sum(
            row["exact"]["complete_belief_certificate_violation_count"]
            for row in fixtures.values()
        )
        <= gates["maximumCompleteBeliefExecutionCertificateViolations"],
        "zero_off_support_fallback": sum(
            row[control]["off_support_fallback_count"]
            for row in fixtures.values()
            for control in ("map", "posterior_sampling")
        )
        <= gates["maximumOffSupportFallbackCount"],
        "zero_model_API_adapter_human_tool_and_external_access": bool(
            access["model_forward_pass_count"] <= gates["maximumModelForwardPassCount"]
            and access["API_call_count"] <= gates["maximumAPICallCount"]
            and access["adapter_training_run_count"]
            <= gates["maximumAdapterTrainingRunCount"]
            and access["human_record_access_count"]
            <= gates["maximumHumanRecordAccessCount"]
            and access["real_tool_call_count"] <= gates["maximumRealToolCallCount"]
            and access["external_side_effect_count"]
            <= gates["maximumExternalSideEffectCount"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock", default="configs/v78-clarification-implementation-lock.json"
    )
    args = parser.parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V78 implementation lock payload drifted")
    if not lock["authorization"]["run_model_free_census_once"]:
        raise RuntimeError("V78 implementation lock does not authorize the census")
    for path_key, hash_key in (
        ("design_lock", "design_lock_sha256"),
        ("resource_budget", "resource_budget_sha256"),
        ("benchmark_core", "benchmark_core_sha256"),
        ("evaluator", "evaluator_sha256"),
        ("census_harness", "census_harness_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V78 locked dependency drifted: {path_key}")

    config = lock["design_payload"]
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
            / "outputs/v78-structured-llm-interface/model-free-evaluation"
        ),
        attempt=access,
        fixture_rows=config["fixtures"],
        evaluate_fixture=lambda row: evaluate_fixture(row, config),
        evaluate_gates=lambda fixtures: evaluate_gates(fixtures, config, access),
        result_metadata={
            "schema_version": "78-clarification-benchmark-outcome",
            "experiment": "v78_model_free_clarification_benchmark_census",
            "claim_boundary": (
                "fresh project-authored, synthetic, model-free development benchmark; "
                "not human-language, external-benchmark, model, or safety evidence"
            ),
        },
        pass_decision=(
            "freeze_model_free_benchmark_and_authorize_local_model_protocol_preregistration"
        ),
        fail_decision=(
            "freeze_model_free_benchmark_design_failure_without_parameter_tuning"
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
