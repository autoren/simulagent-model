#!/usr/bin/env python3
"""Independently recompute and freeze the passing V79 terminal-utility result."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v78_clarification_benchmark import ACTION_NAMES, INFORMATION_ACTIONS, NONE_HYPOTHESIS, OBSERVATION_NAMES
from verify_and_freeze_v78_clarification_outcome import (
    allowed_actions,
    ask_always_policy,
    independent_gates,
    independent_step,
    normalized_belief,
    select,
    terminal,
    terminal_control_policy,
    traverse,
)
from v78_clarification_benchmark import EXECUTION_ACTIONS, build_fixture


ACTIVE_UNRESOLVED_TERMINAL_UTILITY = -6.0


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def independent_terminal_value(kernel, belief: np.ndarray) -> float:
    value = normalized_belief(kernel, belief)
    return float(value[:, 0].sum() * ACTIVE_UNRESOLVED_TERMINAL_UTILITY)


def independent_plan79(kernel, belief: np.ndarray, horizon: int, tolerance: float):
    value = normalized_belief(kernel, belief)
    if terminal(kernel, value):
        return {"terminal": True, "horizon": horizon, "value": 0.0}
    if horizon == 0:
        return {
            "terminal": True,
            "horizon": 0,
            "value": independent_terminal_value(kernel, value),
        }
    candidates = []
    for action in allowed_actions(kernel, value):
        step = independent_step(kernel, value, action)
        branches = {}
        continuation = 0.0
        for observation, posterior in step["posteriors"].items():
            child = independent_plan79(kernel, posterior, horizon - 1, tolerance)
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


def independent_evaluate79(kernel, belief: np.ndarray, policy: dict, horizon: int) -> float:
    value = normalized_belief(kernel, belief)
    if terminal(kernel, value):
        return 0.0
    if horizon == 0:
        return independent_terminal_value(kernel, value)
    action = int(policy["selected_action"])
    step = independent_step(kernel, value, action)
    continuation = 0.0
    for observation, posterior in step["posteriors"].items():
        if observation in policy.get("branches", {}):
            child = independent_evaluate79(
                kernel, posterior, policy["branches"][observation], horizon - 1
            )
        elif terminal(kernel, posterior):
            child = 0.0
        else:
            raise RuntimeError("independent V79 policy omitted unresolved branch")
        continuation += float(step["probabilities"][observation]) * child
    return float(step["reward"] + kernel.discount * continuation)


def point_policy79(kernel, belief: np.ndarray, hypothesis: int, horizon: int, tolerance: float):
    value = normalized_belief(kernel, belief)
    mass = float(value[hypothesis].sum())
    point = np.zeros_like(value)
    point[hypothesis] = value[hypothesis] / mass
    return independent_plan79(kernel, point, horizon, tolerance)


def independent_fixture79(config: dict[str, Any], name: str) -> dict[str, Any]:
    fixture = build_fixture(config, name)
    kernel = fixture.kernel
    belief = fixture.initial_belief
    horizon = int(config["sharedParameters"]["horizonActions"])
    tolerance = float(config["sharedParameters"]["tieTolerance"])
    exact = independent_plan79(kernel, belief, horizon, tolerance)
    exact_value = float(exact["value"])
    masses = belief.sum(axis=1)
    map_hypothesis = int(np.argmax(masses))
    map_policy = point_policy79(kernel, belief, map_hypothesis, horizon, tolerance)
    map_value = independent_evaluate79(kernel, belief, map_policy, horizon)
    ps_value = 0.0
    for hypothesis, mass in enumerate(masses):
        policy = point_policy79(kernel, belief, hypothesis, horizon, tolerance)
        ps_value += float(mass) * independent_evaluate79(
            kernel, belief, policy, horizon
        )
    immediate_policy = terminal_control_policy(
        kernel, belief, horizon, tolerance, False
    )
    immediate_value = independent_evaluate79(
        kernel, belief, immediate_policy, horizon
    )
    ask_policy = ask_always_policy(kernel, belief, horizon, tolerance)
    ask_value = independent_evaluate79(kernel, belief, ask_policy, horizon)
    scale = max(
        1.0,
        float(kernel.reward.max() - kernel.reward.min())
        * sum(kernel.discount**depth for depth in range(horizon)),
    )
    nodes = list(traverse(exact))
    information = sorted(
        {ACTION_NAMES[action] for _, action, _ in nodes if action in INFORMATION_ACTIONS}
    )
    unknown = [
        (history, action, posterior)
        for history, action, posterior in nodes
        if posterior is not None
        and any(OBSERVATION_NAMES[index].endswith("_other") for index in history)
        and float(posterior[NONE_HYPOTHESIS]) > float(masses[NONE_HYPOTHESIS])
    ]
    return {
        "root_action": ACTION_NAMES[int(exact["selected_action"])],
        "value": exact_value,
        "q_values": {
            ACTION_NAMES[int(action)]: float(value)
            for action, value in exact["q_values"].items()
        },
        "information": information,
        "safe_unknown": sum(
            ACTION_NAMES[action] in ("safe_preview", "abstain")
            for _, action, _ in unknown
        ),
        "unsafe_unknown": sum(action in EXECUTION_ACTIONS for _, action, _ in unknown),
        "map_regret": float((exact_value - map_value) / scale),
        "ps_regret": float((exact_value - ps_value) / scale),
        "immediate_regret": float((exact_value - immediate_value) / scale),
        "ask_regret": float((exact_value - ask_value) / scale),
        "active_terminal_value": independent_terminal_value(kernel, belief),
    }


def main() -> None:
    implementation_path = (
        PROJECT_ROOT / "configs/v79-terminal-utility-implementation-lock.json"
    )
    outcome_dir = PROJECT_ROOT / "outputs/v79-structured-llm-interface/model-free-evaluation"
    result_path = outcome_dir / "result.json"
    verifier_path = (
        PROJECT_ROOT / "python/verify_and_freeze_v79_terminal_utility_outcome.py"
    )
    audit_path = PROJECT_ROOT / "outputs/v79-structured-llm-interface/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v79-terminal-utility-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V79 outcome is already independently frozen")

    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {
        key: value
        for key, value in implementation.items()
        if key != "lock_payload_sha256"
    }
    result = json.loads(result_path.read_text())
    config = implementation["resolved_config_payload"]
    independent = {
        row["name"]: independent_fixture79(config, row["name"])
        for row in config["fixtures"]
    }
    fixture_matches = {}
    for name, recomputed in independent.items():
        recorded = result["fixtures"][name]
        fixture_matches[name] = bool(
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
            and recomputed["safe_unknown"]
            == recorded["exact"]["safe_unknown_continuation_count"]
            and recomputed["unsafe_unknown"]
            == recorded["exact"]["unknown_branch_irreversible_execution_count"]
            and recomputed["active_terminal_value"]
            == recorded["terminal_utility"]["initial_active_belief_at_horizon_zero"]
        )
    base_gates = independent_gates(independent, config, result)
    recomputed_gates = {
        **base_gates,
        "active_unresolved_terminal_utility_is_explicit": all(
            row["active_terminal_value"] == -6.0 for row in independent.values()
        ),
        "terminal_state_terminal_utility_is_zero": all(
            row["terminal_utility"]["matched_terminal_belief_at_horizon_zero"] == 0.0
            for row in result["fixtures"].values()
        ),
        "terminal_utility_replay_agreement": all(
            abs(row["exact"]["value"] - row["exact"]["replay_value"]) <= 1e-10
            for row in result["fixtures"].values()
        ),
    }
    raw_files = sorted((outcome_dir / "raw-fixtures").glob("*.json"))
    raw_match = bool(
        len(raw_files) == 4
        and all(
            json.loads(path.read_text())
            == result["fixtures"][json.loads(path.read_text())["name"]]
            for path in raw_files
        )
    )
    checks = {
        "implementation_lock_payload_valid": payload_hash(implementation_payload)
        == implementation["lock_payload_sha256"],
        "locked_planner_evaluator_reporter_and_harness_unchanged": all(
            file_sha256(PROJECT_ROOT / implementation[path_key])
            == implementation[hash_key]
            for path_key, hash_key in (
                ("terminal_planner", "terminal_planner_sha256"),
                ("evaluator", "evaluator_sha256"),
                ("parent_reporter", "parent_reporter_sha256"),
                ("census_harness", "census_harness_sha256"),
            )
        ),
        "all_four_raw_fixture_artifacts_match_result": raw_match,
        "independent_terminal_aware_roots_Q_values_and_controls_match": all(
            fixture_matches.values()
        ),
        "independent_complete_gate_vector_matches": recomputed_gates == result["gates"],
        "every_inherited_and_terminal_gate_passed": all(recomputed_gates.values()),
        "outcome_authorizes_protocol_preregistration_only": bool(
            result["passed"]
            and result["decision"]
            == "freeze_terminal_utility_mechanism_and_authorize_local_model_protocol_preregistration"
        ),
        "zero_model_API_adapter_human_tool_and_external_access": base_gates[
            "zero_model_API_adapter_human_tool_and_external_access"
        ],
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "79-terminal-utility-outcome-audit",
        "experiment": "v79_independent_terminal_utility_outcome_audit",
        "passed": passed,
        "scientific_outcome": "positive_model_free_development_mechanism",
        "decision": (
            "freeze_V79_and_authorize_frozen_local_model_protocol_preregistration_only"
            if passed
            else "defer_structured_LLM_interface_branch"
        ),
        "checks": checks,
        "fixture_matches": fixture_matches,
        "recomputed_gates": recomputed_gates,
        "claim_boundary": (
            "V79 supports a synthetic exact decision mechanism with explicit terminal "
            "utility; it is not evidence about any LLM, language, human, API, real tool, "
            "or open-world safety property"
        ),
        "access": result["attempt"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "79-terminal-utility-outcome-lock",
        "experiment": "v79_model_free_terminal_utility_outcome_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
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
            "modify_or_rerun_V79": False,
            "claim_V79_as_LLM_language_human_or_safety_evidence": False,
            "access_local_or_API_model": False,
            "preregister_frozen_local_model_candidate_generation_protocol": True,
            "run_model_forward_pass_before_protocol_lock": False,
            "train_adapter_or_learn_likelihoods": False,
            "perform_real_tool_call_or_external_side_effect": False,
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
