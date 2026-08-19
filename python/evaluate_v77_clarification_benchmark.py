#!/usr/bin/env python3
"""Run the one authorized model-free V77 clarification benchmark census."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v77_clarification_benchmark import (
    ACTION_NAMES,
    INFORMATION_ACTIONS,
    NONE_HYPOTHESIS,
    OBSERVATION_NAMES,
    SEND_ACTIONS,
    act_immediately_policy,
    ask_always_policy,
    best_open_loop_sequence,
    build_fixture,
    evaluate_policy_exact,
    finite_horizon_return_scale,
    map_control,
    oracle_interpretation_value,
    plan_exact,
    posterior_sampling_control,
    structural_diagnostics,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _policy_nodes(
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
        yield from _policy_nodes(child, history + (int(observation),))


def _named_policy_nodes(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _policy_nodes(policy):
        rows.append(
            {
                "history": [OBSERVATION_NAMES[index] for index in row["history"]],
                "action": ACTION_NAMES[row["action"]],
                "hypothesis_masses": row["hypothesis_masses"],
            }
        )
    return rows


def _control_summary(
    kernel,
    belief,
    policy: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    value = evaluate_policy_exact(
        kernel,
        belief,
        policy,
        horizon,
        certificate_violations=violations,
    )
    return {
        "value": float(value),
        "root_action": ACTION_NAMES[int(policy["selected_action"])],
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
    mapped = map_control(kernel, belief, horizon, tie_tolerance=tolerance)
    sampled = posterior_sampling_control(
        kernel, belief, horizon, tie_tolerance=tolerance
    )
    immediate_policy = act_immediately_policy(
        kernel, belief, horizon, tie_tolerance=tolerance
    )
    ask_policy = ask_always_policy(kernel, belief, horizon, tie_tolerance=tolerance)
    immediate = _control_summary(kernel, belief, immediate_policy, horizon)
    ask = _control_summary(kernel, belief, ask_policy, horizon)
    open_loop = best_open_loop_sequence(
        kernel, belief, horizon, tie_tolerance=tolerance
    )
    oracle = oracle_interpretation_value(
        kernel, belief, horizon, tie_tolerance=tolerance
    )
    scale = finite_horizon_return_scale(kernel, horizon)
    exact_value = float(exact["value"])
    policy_nodes = _named_policy_nodes(exact)
    information_actions = sorted(
        {
            node["action"]
            for node in policy_nodes
            if ACTION_NAMES.index(node["action"]) in INFORMATION_ACTIONS
        }
    )
    unknown_nodes = [
        node
        for node in policy_nodes
        if any(observation.endswith("_other") for observation in node["history"])
        and node["hypothesis_masses"] is not None
        and float(node["hypothesis_masses"][NONE_HYPOTHESIS])
        > float(belief[NONE_HYPOTHESIS].sum())
    ]
    safe_unknown_nodes = [
        node for node in unknown_nodes if node["action"] in ("safe_draft", "abstain")
    ]
    unsafe_unknown_nodes = [
        node for node in unknown_nodes if node["action"] in {
            ACTION_NAMES[index] for index in SEND_ACTIONS
        }
    ]
    q_values = {
        ACTION_NAMES[int(action)]: float(value)
        for action, value in exact["q_values"].items()
    }
    return {
        "name": fixture.name,
        "synthetic_instruction": fixture.synthetic_instruction,
        "reward_profile": fixture.reward_profile,
        "claim_boundary": "project-authored model-free development mechanism",
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
            "root_action": ACTION_NAMES[int(exact["selected_action"])],
            "root_optimal_actions": [
                ACTION_NAMES[int(action)] for action in exact["optimal_actions"]
            ],
            "root_q_values": q_values,
            "policy_nodes": policy_nodes,
            "reachable_information_actions": information_actions,
            "unknown_indicating_node_count": len(unknown_nodes),
            "safe_unknown_continuation_count": len(safe_unknown_nodes),
            "unknown_branch_irreversible_send_count": len(unsafe_unknown_nodes),
            "maximum_none_posterior_on_unknown_history": max(
                (
                    float(node["hypothesis_masses"][NONE_HYPOTHESIS])
                    for node in unknown_nodes
                ),
                default=float(belief[NONE_HYPOTHESIS].sum()),
            ),
        },
        "map": {
            "hypothesis": mapped["hypothesis_name"],
            "root_action": ACTION_NAMES[int(mapped["policy"]["selected_action"])],
            "exact_environment_value": float(mapped["value"]),
            "normalized_regret": float((exact_value - float(mapped["value"])) / scale),
            "complete_belief_certificate_violation_count": len(
                mapped["complete_belief_certificate_violations"]
            ),
            "shadow_only": True,
            "off_support_fallback_count": mapped["off_support_fallback_count"],
        },
        "posterior_sampling": {
            "exact_environment_value": float(sampled["value"]),
            "normalized_regret": float((exact_value - float(sampled["value"])) / scale),
            "root_action_distribution": dict(
                zip(ACTION_NAMES, sampled["root_action_distribution"], strict=True)
            ),
            "complete_belief_certificate_violation_count": len(
                sampled["complete_belief_certificate_violations"]
            ),
            "shadow_only": True,
            "off_support_fallback_count": sampled["off_support_fallback_count"],
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
            "normalized_regret": float((exact_value - float(open_loop["value"])) / scale),
            "selected_actions": [ACTION_NAMES[int(action)] for action in open_loop["selected_actions"]],
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
    fixtures: dict[str, dict[str, Any]], config: dict[str, Any], access: dict[str, int]
) -> dict[str, bool]:
    gates = config["benchmarkGates"]
    ambiguous = fixtures["ambiguous_send"]
    clear = fixtures["clear_send"]
    unknown = fixtures["unknown_heavy"]
    dominant = fixtures["dominant_safe_draft"]
    structural_rows = [row["structural"] for row in fixtures.values()]
    return {
        "complete_fixture_and_hypothesis_census": bool(
            len(fixtures) == gates["requiredFixtureCount"]
            and all(
                row["hypothesis_count"] == gates["requiredHypothesisCount"]
                for row in structural_rows
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
        "clear_sends_immediately": clear["exact"]["root_action"]
        == gates["requiredClearRootAction"],
        "clear_ask_always_regret_is_material": clear["ask_always"]["normalized_regret"]
        >= gates["minimumClearNormalizedAskAlwaysRegret"],
        "unknown_begins_with_clarification": unknown["exact"]["root_action"]
        in gates["requiredUnknownRootActionSet"],
        "unknown_has_safe_unknown_continuation": bool(
            unknown["exact"]["safe_unknown_continuation_count"] > 0
        ),
        "unknown_never_sends_on_unknown_indicating_branch": unknown["exact"][
            "unknown_branch_irreversible_send_count"
        ]
        <= gates["maximumUnknownBranchIrreversibleSendCount"],
        "dominant_safe_draft_is_immediate": dominant["exact"]["root_action"]
        == gates["requiredDominantRootAction"],
        "dominant_MAP_regret_is_zero": dominant["map"]["normalized_regret"]
        <= gates["maximumDominantNormalizedMAPRegret"],
        "dominant_posterior_sampling_regret_is_zero": dominant[
            "posterior_sampling"
        ]["normalized_regret"]
        <= gates["maximumDominantNormalizedPosteriorSamplingRegret"],
        "all_observation_rows_normalize": all(
            row["observation_normalization_rate"]
            >= gates["minimumObservationNormalizationRate"]
            for row in structural_rows
        ),
        "all_hypothesis_supports_are_identical": all(
            row["identical_hypothesis_support_rate"]
            >= gates["minimumIdenticalHypothesisSupportRate"]
            for row in structural_rows
        ),
        "all_initial_and_reachable_beliefs_normalize": all(
            row["belief_normalizes"]
            and fixtures[row["name"]]["resource"]["belief_normalization_rate"]
            >= gates["minimumBeliefNormalizationRate"]
            for row in structural_rows
        ),
        "zero_off_support_fallback": sum(
            row[control]["off_support_fallback_count"]
            for row in fixtures.values()
            for control in ("map", "posterior_sampling")
        )
        <= gates["maximumOffSupportFallbackCount"],
        "zero_model_API_adapter_human_and_external_access": bool(
            access["model_forward_pass_count"] <= gates["maximumModelForwardPassCount"]
            and access["API_call_count"] <= gates["maximumAPICallCount"]
            and access["adapter_training_run_count"]
            <= gates["maximumAdapterTrainingRunCount"]
            and access["human_record_access_count"]
            <= gates["maximumHumanRecordAccessCount"]
            and access["external_side_effect_count"]
            <= gates["maximumExternalSideEffectCount"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock", default="configs/v77-clarification-implementation-lock.json"
    )
    args = parser.parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V77 implementation lock payload drifted")
    if not lock["authorization"]["run_model_free_census_once"]:
        raise RuntimeError("V77 implementation lock does not authorize the census")
    for path_key, hash_key in (
        ("design_lock", "design_lock_sha256"),
        ("resource_budget", "resource_budget_sha256"),
        ("benchmark_core", "benchmark_core_sha256"),
        ("evaluator", "evaluator_sha256"),
    ):
        if file_sha256(PROJECT_ROOT / lock[path_key]) != lock[hash_key]:
            raise RuntimeError(f"V77 locked dependency drifted: {path_key}")

    output_dir = PROJECT_ROOT / "outputs/v77-structured-llm-interface/model-free-evaluation"
    if output_dir.exists():
        raise RuntimeError("V77 model-free evaluation already exists")
    output_dir.mkdir(parents=True)
    access = {
        "attempt_number": 1,
        "model_forward_pass_count": 0,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
        "external_side_effect_count": 0,
        "real_tool_call_count": 0,
        "real_message_or_file_send_count": 0,
    }
    (output_dir / "attempt.json").write_text(
        json.dumps(access, indent=2, sort_keys=True) + "\n"
    )
    config = lock["design_payload"]
    fixtures = {
        row["name"]: evaluate_fixture(row, config) for row in config["fixtures"]
    }
    gates = evaluate_gates(fixtures, config, access)
    result = {
        "schema_version": "77-clarification-benchmark-outcome",
        "experiment": "v77_model_free_clarification_benchmark_census",
        "claim_boundary": (
            "project-authored, synthetic, model-free development benchmark; "
            "not human-language, external-benchmark, model, or safety evidence"
        ),
        "passed": all(gates.values()),
        "decision": (
            "freeze_model_free_benchmark_and_authorize_local_model_protocol_preregistration"
            if all(gates.values())
            else "freeze_model_free_benchmark_design_failure_without_parameter_tuning"
        ),
        "gates": gates,
        "fixtures": fixtures,
        "access": access,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
