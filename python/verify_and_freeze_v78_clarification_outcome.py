#!/usr/bin/env python3
"""Independently recompute and freeze the V78 model-free outcome."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v78_clarification_benchmark import (
    ACTION_NAMES,
    EXECUTION_ACTIONS,
    INFORMATION_ACTIONS,
    NONE_HYPOTHESIS,
    OBSERVATION_NAMES,
    TERMINAL_ACTIONS,
    build_fixture,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalized_belief(kernel, belief: np.ndarray) -> np.ndarray:
    value = np.asarray(belief, dtype=np.float64)
    if value.shape != (len(kernel.hypothesis_names), len(kernel.state_names)):
        raise ValueError("independent V78 belief shape mismatch")
    if np.any(value < 0.0) or not np.isclose(value.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("independent V78 belief failed normalization")
    return value


def terminal(kernel, belief: np.ndarray) -> bool:
    return bool(normalized_belief(kernel, belief)[:, 1].sum() >= 1.0 - 1e-12)


def allowed_actions(kernel, belief: np.ndarray) -> tuple[int, ...]:
    masses = normalized_belief(kernel, belief).sum(axis=1)
    allowed = list(kernel.always_certified_actions)
    for action, hypothesis in kernel.send_action_to_hypothesis:
        if (
            masses[hypothesis] + 1e-12 >= kernel.send_minimum_matching_posterior
            and masses[kernel.none_hypothesis]
            <= kernel.send_maximum_none_posterior + 1e-12
        ):
            allowed.append(action)
    return tuple(sorted(allowed))


def independent_step(kernel, belief: np.ndarray, action: int) -> dict[str, Any]:
    value = normalized_belief(kernel, belief)
    predicted = np.einsum("hs,sq->hq", value, kernel.transition[action], optimize=True)
    joint = predicted[:, :, None] * kernel.observation[:, action]
    probabilities = joint.sum(axis=(0, 1))
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("independent V78 predictive distribution failed")
    posteriors = {
        observation: joint[:, :, observation] / float(probability)
        for observation, probability in enumerate(probabilities)
        if probability > 0.0
    }
    reward = float(
        np.einsum(
            "hs,sq,hsq->",
            value,
            kernel.transition[action],
            kernel.reward[:, action],
            optimize=True,
        )
    )
    return {"reward": reward, "probabilities": probabilities, "posteriors": posteriors}


def select(rows: list[tuple[int, float]], tolerance: float) -> tuple[int, float]:
    maximum = max(value for _, value in rows)
    action = next(action for action, value in rows if maximum - value <= tolerance)
    return action, maximum


def independent_plan(
    kernel, belief: np.ndarray, horizon: int, tolerance: float
) -> dict[str, Any]:
    value = normalized_belief(kernel, belief)
    if horizon == 0 or terminal(kernel, value):
        return {"terminal": True, "horizon": horizon, "value": 0.0}
    candidates: list[dict[str, Any]] = []
    for action in allowed_actions(kernel, value):
        step = independent_step(kernel, value, action)
        branches: dict[int, dict[str, Any]] = {}
        continuation = 0.0
        if horizon > 1:
            for observation, posterior in step["posteriors"].items():
                child = independent_plan(kernel, posterior, horizon - 1, tolerance)
                branches[observation] = child
                continuation += float(step["probabilities"][observation]) * float(
                    child["value"]
                )
        candidates.append(
            {
                "action": action,
                "value": float(step["reward"] + kernel.discount * continuation),
                "branches": branches,
            }
        )
    action, maximum = select(
        [(row["action"], row["value"]) for row in candidates], tolerance
    )
    selected = next(row for row in candidates if row["action"] == action)
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(maximum),
        "selected_action": action,
        "q_values": {row["action"]: row["value"] for row in candidates},
        "hypothesis_masses": value.sum(axis=1).tolist(),
        "branches": selected["branches"],
    }


def independent_evaluate(kernel, belief: np.ndarray, policy: dict, horizon: int) -> float:
    value = normalized_belief(kernel, belief)
    if horizon == 0 or terminal(kernel, value):
        return 0.0
    action = int(policy["selected_action"])
    step = independent_step(kernel, value, action)
    continuation = 0.0
    if horizon > 1:
        for observation, posterior in step["posteriors"].items():
            if terminal(kernel, posterior):
                child_value = 0.0
            else:
                child_value = independent_evaluate(
                    kernel, posterior, policy["branches"][observation], horizon - 1
                )
            continuation += float(step["probabilities"][observation]) * child_value
    return float(step["reward"] + kernel.discount * continuation)


def point_policy(kernel, belief: np.ndarray, hypothesis: int, horizon: int, tolerance: float):
    value = normalized_belief(kernel, belief)
    mass = float(value[hypothesis].sum())
    point = np.zeros_like(value)
    point[hypothesis] = value[hypothesis] / mass
    return independent_plan(kernel, point, horizon, tolerance)


def terminal_control_policy(
    kernel, belief: np.ndarray, horizon: int, tolerance: float, certify: bool
) -> dict[str, Any]:
    actions = TERMINAL_ACTIONS
    if certify:
        allowed = set(allowed_actions(kernel, belief))
        actions = tuple(action for action in actions if action in allowed)
    rows = [
        (action, float(independent_step(kernel, belief, action)["reward"]))
        for action in actions
    ]
    action, maximum = select(rows, tolerance)
    return {
        "terminal": False,
        "horizon": horizon,
        "value": maximum,
        "selected_action": action,
        "branches": {},
    }


def ask_always_policy(kernel, belief: np.ndarray, horizon: int, tolerance: float):
    if horizon <= 1:
        return terminal_control_policy(kernel, belief, max(1, horizon), tolerance, True)
    action = ACTION_NAMES.index("ask_full_details")
    step = independent_step(kernel, belief, action)
    return {
        "terminal": False,
        "horizon": horizon,
        "selected_action": action,
        "branches": {
            observation: ask_always_policy(kernel, posterior, horizon - 1, tolerance)
            for observation, posterior in step["posteriors"].items()
        },
    }


def traverse(policy: dict, history: tuple[int, ...] = ()):
    if policy.get("terminal"):
        return
    yield history, int(policy["selected_action"]), policy.get("hypothesis_masses")
    for observation, child in policy.get("branches", {}).items():
        yield from traverse(child, history + (int(observation),))


def independent_fixture(config: dict[str, Any], name: str) -> dict[str, Any]:
    fixture = build_fixture(config, name)
    kernel = fixture.kernel
    belief = fixture.initial_belief
    horizon = int(config["sharedParameters"]["horizonActions"])
    tolerance = float(config["sharedParameters"]["tieTolerance"])
    exact = independent_plan(kernel, belief, horizon, tolerance)
    exact_value = float(exact["value"])
    masses = belief.sum(axis=1)
    map_hypothesis = int(np.argmax(masses))
    map_policy = point_policy(kernel, belief, map_hypothesis, horizon, tolerance)
    map_value = independent_evaluate(kernel, belief, map_policy, horizon)
    ps_value = 0.0
    for hypothesis, mass in enumerate(masses):
        policy = point_policy(kernel, belief, hypothesis, horizon, tolerance)
        ps_value += float(mass) * independent_evaluate(kernel, belief, policy, horizon)
    immediate_policy = terminal_control_policy(
        kernel, belief, horizon, tolerance, False
    )
    immediate_value = independent_evaluate(
        kernel, belief, immediate_policy, horizon
    )
    ask_policy = ask_always_policy(kernel, belief, horizon, tolerance)
    ask_value = independent_evaluate(kernel, belief, ask_policy, horizon)
    scale = max(
        1.0,
        float(kernel.reward.max() - kernel.reward.min())
        * sum(kernel.discount**depth for depth in range(horizon)),
    )
    nodes = list(traverse(exact))
    information = sorted(
        {ACTION_NAMES[action] for _, action, _ in nodes if action in INFORMATION_ACTIONS}
    )
    unknown_nodes = [
        (history, action, posterior)
        for history, action, posterior in nodes
        if posterior is not None
        and any(OBSERVATION_NAMES[index].endswith("_other") for index in history)
        and float(posterior[NONE_HYPOTHESIS]) > float(masses[NONE_HYPOTHESIS])
    ]
    safe_unknown = sum(
        ACTION_NAMES[action] in ("safe_preview", "abstain")
        for _, action, _ in unknown_nodes
    )
    unsafe_unknown = sum(
        action in EXECUTION_ACTIONS for _, action, _ in unknown_nodes
    )
    return {
        "root_action": ACTION_NAMES[int(exact["selected_action"])],
        "value": exact_value,
        "q_values": {
            ACTION_NAMES[int(action)]: float(value)
            for action, value in exact["q_values"].items()
        },
        "information": information,
        "safe_unknown": safe_unknown,
        "unsafe_unknown": unsafe_unknown,
        "map_regret": float((exact_value - map_value) / scale),
        "ps_regret": float((exact_value - ps_value) / scale),
        "immediate_regret": float((exact_value - immediate_value) / scale),
        "ask_regret": float((exact_value - ask_value) / scale),
    }


def independent_gates(
    rows: dict[str, dict[str, Any]], config: dict[str, Any], result: dict[str, Any]
) -> dict[str, bool]:
    gates = config["benchmarkGates"]
    raw = result["fixtures"]
    ambiguous = rows["ambiguous_tool_intent"]
    clear = rows["clear_tool_intent"]
    unknown = rows["unknown_heavy_tool_intent"]
    dominant = rows["dominant_safe_preview"]
    return {
        "complete_fixture_and_hypothesis_census": len(rows) == 4,
        "ambiguous_begins_with_focused_clarification": ambiguous["root_action"]
        in gates["requiredAmbiguousRootActionSet"],
        "ambiguous_uses_both_focused_information_actions": all(
            action in ambiguous["information"]
            for action in gates["requiredAmbiguousReachableInformationActions"]
        ),
        "ambiguous_MAP_regret_is_material": ambiguous["map_regret"]
        >= gates["minimumAmbiguousNormalizedMAPRegret"],
        "ambiguous_act_immediately_regret_is_material": ambiguous["immediate_regret"]
        >= gates["minimumAmbiguousNormalizedActImmediatelyRegret"],
        "clear_executes_immediately": clear["root_action"] == gates["requiredClearRootAction"],
        "clear_ask_always_regret_is_material": clear["ask_regret"]
        >= gates["minimumClearNormalizedAskAlwaysRegret"],
        "unknown_begins_with_clarification": unknown["root_action"]
        in gates["requiredUnknownRootActionSet"],
        "unknown_has_safe_unknown_continuation": unknown["safe_unknown"] > 0,
        "unknown_never_executes_on_unknown_indicating_branch": unknown["unsafe_unknown"]
        <= gates["maximumUnknownBranchIrreversibleExecutionCount"],
        "dominant_safe_preview_is_immediate": dominant["root_action"]
        == gates["requiredDominantRootAction"],
        "dominant_MAP_regret_is_zero": dominant["map_regret"]
        <= gates["maximumDominantNormalizedMAPRegret"],
        "dominant_posterior_sampling_regret_is_zero": dominant["ps_regret"]
        <= gates["maximumDominantNormalizedPosteriorSamplingRegret"],
        "all_transition_rows_normalize": all(
            row["structural"]["transition_normalization_rate"] == 1.0
            for row in raw.values()
        ),
        "all_observation_rows_normalize": all(
            row["structural"]["observation_normalization_rate"] == 1.0
            for row in raw.values()
        ),
        "all_hypothesis_supports_are_identical": all(
            row["structural"]["identical_hypothesis_support_rate"] == 1.0
            for row in raw.values()
        ),
        "all_initial_and_reachable_beliefs_normalize": all(
            row["structural"]["belief_normalizes"]
            and row["resource"]["belief_normalization_rate"] == 1.0
            for row in raw.values()
        ),
        "exact_policy_has_zero_execution_certificate_violations": all(
            row["exact"]["complete_belief_certificate_violation_count"] == 0
            for row in raw.values()
        ),
        "zero_off_support_fallback": all(
            row[control]["off_support_fallback_count"] == 0
            for row in raw.values()
            for control in ("map", "posterior_sampling")
        ),
        "zero_model_API_adapter_human_tool_and_external_access": all(
            result["attempt"][key] == 0
            for key in (
                "model_forward_pass_count",
                "API_call_count",
                "adapter_training_run_count",
                "human_record_access_count",
                "real_tool_call_count",
                "external_side_effect_count",
            )
        ),
    }


def main() -> None:
    implementation_lock_path = (
        PROJECT_ROOT / "configs/v78-clarification-implementation-lock.json"
    )
    outcome_dir = PROJECT_ROOT / "outputs/v78-structured-llm-interface/model-free-evaluation"
    result_path = outcome_dir / "result.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v78_clarification_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v78-structured-llm-interface/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v78-clarification-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V78 outcome is already independently frozen")

    implementation = json.loads(implementation_lock_path.read_text())
    implementation_payload = {
        key: value
        for key, value in implementation.items()
        if key != "lock_payload_sha256"
    }
    result = json.loads(result_path.read_text())
    config = implementation["design_payload"]
    independent = {
        row["name"]: independent_fixture(config, row["name"])
        for row in config["fixtures"]
    }
    exact_matches = {}
    for name, recomputed in independent.items():
        recorded = result["fixtures"][name]
        exact_matches[name] = bool(
            recomputed["root_action"] == recorded["exact"]["root_action"]
            and abs(recomputed["value"] - recorded["exact"]["value"]) <= 1e-10
            and set(recomputed["q_values"]) == set(recorded["exact"]["root_q_values"])
            and all(
                abs(value - recorded["exact"]["root_q_values"][action]) <= 1e-10
                for action, value in recomputed["q_values"].items()
            )
            and abs(recomputed["map_regret"] - recorded["map"]["normalized_regret"])
            <= 1e-10
            and abs(
                recomputed["ps_regret"]
                - recorded["posterior_sampling"]["normalized_regret"]
            )
            <= 1e-10
            and abs(
                recomputed["immediate_regret"]
                - recorded["act_immediately"]["normalized_regret"]
            )
            <= 1e-10
            and abs(recomputed["ask_regret"] - recorded["ask_always"]["normalized_regret"])
            <= 1e-10
        )
    recomputed_gates = independent_gates(independent, config, result)
    raw_files = sorted((outcome_dir / "raw-fixtures").glob("*.json"))
    raw_match = bool(
        len(raw_files) == len(result["fixtures"])
        and all(
            json.loads(path.read_text())
            == result["fixtures"][json.loads(path.read_text())["name"]]
            for path in raw_files
        )
    )
    checks = {
        "implementation_lock_payload_valid": payload_hash(implementation_payload)
        == implementation["lock_payload_sha256"],
        "locked_core_evaluator_and_harness_unchanged": all(
            file_sha256(PROJECT_ROOT / implementation[path_key])
            == implementation[hash_key]
            for path_key, hash_key in (
                ("benchmark_core", "benchmark_core_sha256"),
                ("evaluator", "evaluator_sha256"),
                ("census_harness", "census_harness_sha256"),
            )
        ),
        "all_four_durable_raw_fixture_artifacts_match_result": raw_match,
        "independent_exact_roots_values_Q_values_and_controls_match": all(
            exact_matches.values()
        ),
        "independent_gate_vector_matches": recomputed_gates == result["gates"],
        "exactly_one_preregistered_gate_failed": sum(
            not value for value in recomputed_gates.values()
        )
        == 1,
        "sole_failure_is_unknown_safe_continuation": bool(
            not recomputed_gates["unknown_has_safe_unknown_continuation"]
            and all(
                value
                for name, value in recomputed_gates.items()
                if name != "unknown_has_safe_unknown_continuation"
            )
        ),
        "outcome_is_frozen_design_failure": bool(
            not result["passed"]
            and result["decision"]
            == "freeze_model_free_benchmark_design_failure_without_parameter_tuning"
        ),
        "zero_model_API_adapter_human_tool_and_external_access": recomputed_gates[
            "zero_model_API_adapter_human_tool_and_external_access"
        ],
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "78-clarification-outcome-audit",
        "experiment": "v78_independent_model_free_outcome_audit",
        "passed": passed,
        "scientific_outcome": "negative_benchmark_design_result",
        "decision": (
            "freeze_v78_failure_and_authorize_fresh_terminal-utility_successor_preregistration"
            if passed
            else "defer_clarification_branch"
        ),
        "checks": checks,
        "fixture_matches": exact_matches,
        "recomputed_gates": recomputed_gates,
        "interpretation": (
            "V78 validates clarification value, MAP regret, clear-case immediacy, "
            "dominant-action invariance, normalization, support, and fail-closed "
            "execution. It fails the registered safe-unknown continuation gate "
            "because unresolved information actions carry no terminal-horizon cost."
        ),
        "access": result["attempt"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "78-clarification-outcome-lock",
        "experiment": "v78_model_free_outcome_lock",
        "implementation_lock": str(implementation_lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_lock_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "raw_fixture_artifacts": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(path),
            }
            for path in raw_files
        ],
        "verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "verifier_sha256": file_sha256(verifier_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "decision": audit["decision"],
        "authorization": {
            "modify_or_rerun_V78": False,
            "claim_V78_as_model_or_human_evidence": False,
            "access_local_or_API_model": False,
            "preregister_fresh_successor_with_explicit_terminal_unresolved_utility": True,
            "compute_successor_outcomes_before_successor_lock": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path), "sha256": file_sha256(lock_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
