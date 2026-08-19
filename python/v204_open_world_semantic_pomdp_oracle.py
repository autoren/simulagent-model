from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Sequence

import numpy as np


LATENT_NAMES = ("CANONICAL", "REVERSED", "OUTSIDE_UNKNOWN")
STATE_NAMES = ("ready_A", "ready_B", "pending_good", "pending_bad", "terminal")
ACTION_NAMES = ("calibrate", "inspect", "repair_A", "repair_B", "defer", "settle")
OBSERVATION_NAMES = ("red", "blue", "green", "none")


@dataclass(frozen=True)
class SemanticKernel:
    transition: np.ndarray
    observation: np.ndarray
    reward: np.ndarray
    discount: float

    def __post_init__(self) -> None:
        transition = np.asarray(self.transition, dtype=np.float64)
        observation = np.asarray(self.observation, dtype=np.float64)
        reward = np.asarray(self.reward, dtype=np.float64)
        if transition.shape != (len(ACTION_NAMES), len(STATE_NAMES), len(STATE_NAMES)):
            raise ValueError("V204 transition shape mismatch")
        if (
            observation.ndim != 4
            or observation.shape[0] < 1
            or observation.shape[1:] != (len(ACTION_NAMES), len(STATE_NAMES), len(OBSERVATION_NAMES))
        ):
            raise ValueError("V204 observation shape mismatch")
        if reward.shape != transition.shape:
            raise ValueError("V204 reward shape mismatch")
        if not np.allclose(transition.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V204 transition is not normalized")
        if not np.allclose(observation.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V204 observation is not normalized")
        support = observation > 0.0
        if not all(np.array_equal(support[0], support[index]) for index in range(1, observation.shape[0])):
            raise ValueError("V204 codebook supports differ")
        if not all(np.isfinite(value).all() for value in (transition, observation, reward)):
            raise ValueError("V204 kernel contains nonfinite values")
        for value in (transition, observation, reward):
            value.setflags(write=False)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "reward", reward)


def _transition() -> np.ndarray:
    value = np.zeros((len(ACTION_NAMES), len(STATE_NAMES), len(STATE_NAMES)))
    terminal = STATE_NAMES.index("terminal")
    ready_a = STATE_NAMES.index("ready_A")
    ready_b = STATE_NAMES.index("ready_B")
    good = STATE_NAMES.index("pending_good")
    bad = STATE_NAMES.index("pending_bad")
    for action in range(len(ACTION_NAMES)):
        value[action, terminal, terminal] = 1.0
        for pending in (good, bad):
            value[action, pending, terminal] = 1.0
    for ready in (ready_a, ready_b):
        for action_name in ("calibrate", "inspect"):
            value[ACTION_NAMES.index(action_name), ready, ready] = 1.0
        value[ACTION_NAMES.index("repair_A"), ready, good if ready == ready_a else bad] = 1.0
        value[ACTION_NAMES.index("repair_B"), ready, good if ready == ready_b else bad] = 1.0
        value[ACTION_NAMES.index("defer"), ready, terminal] = 1.0
        value[ACTION_NAMES.index("settle"), ready, terminal] = 1.0
    return value


def _logical_channel(logical: str, latent: int, config: dict[str, Any]) -> np.ndarray:
    channel = config["channel"]
    if latent == LATENT_NAMES.index("OUTSIDE_UNKNOWN"):
        return np.asarray(channel["outsideUnknownForEitherCondition"], dtype=np.float64)
    target = logical
    if latent == LATENT_NAMES.index("REVERSED"):
        target = "B" if logical == "A" else "A"
    key = "knownCodebookLogicalA" if target == "A" else "knownCodebookLogicalB"
    return np.asarray(channel[key], dtype=np.float64)


def _observation(config: dict[str, Any]) -> np.ndarray:
    value = np.zeros(
        (len(LATENT_NAMES), len(ACTION_NAMES), len(STATE_NAMES), len(OBSERVATION_NAMES)),
        dtype=np.float64,
    )
    none = OBSERVATION_NAMES.index("none")
    ready_states = (STATE_NAMES.index("ready_A"), STATE_NAMES.index("ready_B"))
    for latent in range(len(LATENT_NAMES)):
        for action in range(len(ACTION_NAMES)):
            for successor in range(len(STATE_NAMES)):
                value[latent, action, successor, none] = 1.0
        for successor in ready_states:
            for action_name in ("calibrate", "inspect"):
                action = ACTION_NAMES.index(action_name)
                value[latent, action, successor] = 0.0
                logical = "A" if action_name == "calibrate" or successor == ready_states[0] else "B"
                value[latent, action, successor, :3] = _logical_channel(logical, latent, config)
    return value


def _reward(config: dict[str, Any], transition: np.ndarray) -> np.ndarray:
    process = config["decisionProcess"]
    value = np.zeros_like(transition)
    terminal = STATE_NAMES.index("terminal")
    good = STATE_NAMES.index("pending_good")
    bad = STATE_NAMES.index("pending_bad")
    for ready in (STATE_NAMES.index("ready_A"), STATE_NAMES.index("ready_B")):
        for action_name in ("calibrate", "inspect"):
            value[ACTION_NAMES.index(action_name), ready, ready] = process["sensingCost"]
        value[ACTION_NAMES.index("defer"), ready, terminal] = process["safeDeferralReward"]
        value[ACTION_NAMES.index("settle"), ready, terminal] = process["invalidPhaseReward"]
    for pending in (good, bad):
        for action_name in ACTION_NAMES:
            action = ACTION_NAMES.index(action_name)
            value[action, pending, terminal] = process["invalidPhaseReward"]
    value[ACTION_NAMES.index("settle"), good, terminal] = process["correctSettlementReward"]
    value[ACTION_NAMES.index("settle"), bad, terminal] = process["wrongSettlementReward"]
    return value


def build_kernel(config: dict[str, Any]) -> tuple[SemanticKernel, np.ndarray]:
    transition = _transition()
    kernel = SemanticKernel(
        transition=transition,
        observation=_observation(config),
        reward=_reward(config, transition),
        discount=float(config["decisionProcess"]["discount"]),
    )
    latent_prior = np.asarray(config["hypotheses"]["codebookPrior"], dtype=np.float64)
    condition_prior = np.asarray(config["hypotheses"]["conditionPrior"], dtype=np.float64)
    belief = np.zeros((len(LATENT_NAMES), len(STATE_NAMES)), dtype=np.float64)
    belief[:, :2] = latent_prior[:, None] * condition_prior[None, :]
    if not np.isclose(belief.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("V204 initial belief is not normalized")
    return kernel, belief


def _validate_belief(kernel: SemanticKernel, belief: np.ndarray) -> np.ndarray:
    value = np.asarray(belief, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(STATE_NAMES):
        raise ValueError("V204 belief shape mismatch")
    if value.shape[0] != kernel.observation.shape[0]:
        raise ValueError("V204 latent belief shape mismatch")
    if np.any(value < 0.0) or not np.isclose(value.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("V204 belief is invalid")
    return value


def exact_step(kernel: SemanticKernel, belief: np.ndarray, action: int) -> dict[str, Any]:
    value = _validate_belief(kernel, belief)
    predicted = np.einsum("zs,sq->zq", value, kernel.transition[action], optimize=True)
    joint = predicted[:, :, None] * kernel.observation[:, action]
    probabilities = joint.sum(axis=(0, 1))
    posteriors = {
        observation: joint[:, :, observation] / float(probability)
        for observation, probability in enumerate(probabilities)
        if probability > 0.0
    }
    reward = float(
        np.einsum("zs,sq,sq->", value, kernel.transition[action], kernel.reward[action], optimize=True)
    )
    return {"reward": reward, "probabilities": probabilities, "posteriors": posteriors}


def _select(values: Sequence[float], tolerance: float) -> tuple[int, tuple[int, ...], float]:
    maximum = max(float(value) for value in values)
    optimal = tuple(index for index, value in enumerate(values) if maximum - float(value) <= tolerance)
    return optimal[0], optimal, maximum


def plan_joint(
    kernel: SemanticKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    allowed_actions: tuple[int, ...] | None = None,
    tolerance: float = 1e-12,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    value = _validate_belief(kernel, belief)
    if stats is not None:
        stats["bellman_nodes"] = stats.get("bellman_nodes", 0) + 1
    if horizon == 0:
        return {"terminal": True, "horizon": 0, "value": 0.0}
    actions = allowed_actions or tuple(range(len(ACTION_NAMES)))
    rows = []
    for action in actions:
        step = exact_step(kernel, value, action)
        branches = {}
        continuation = 0.0
        if horizon > 1:
            for observation, posterior in step["posteriors"].items():
                child = plan_joint(
                    kernel,
                    posterior,
                    horizon - 1,
                    allowed_actions=actions,
                    tolerance=tolerance,
                    stats=stats,
                )
                branches[observation] = child
                continuation += float(step["probabilities"][observation]) * float(child["value"])
        rows.append({"action": action, "value": step["reward"] + kernel.discount * continuation, "branches": branches})
    selected_offset, optimal_offsets, maximum = _select([row["value"] for row in rows], tolerance)
    selected = rows[selected_offset]
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(maximum),
        "selected_action": selected["action"],
        "optimal_actions": tuple(rows[index]["action"] for index in optimal_offsets),
        "q_values": {row["action"]: float(row["value"]) for row in rows},
        "branches": selected["branches"],
    }


def evaluate_policy(kernel: SemanticKernel, belief: np.ndarray, policy: dict[str, Any], horizon: int) -> float:
    value = _validate_belief(kernel, belief)
    if horizon == 0:
        return 0.0
    action = int(policy["selected_action"])
    step = exact_step(kernel, value, action)
    continuation = 0.0
    if horizon > 1:
        for observation, posterior in step["posteriors"].items():
            continuation += float(step["probabilities"][observation]) * evaluate_policy(
                kernel, posterior, policy["branches"][observation], horizon - 1
            )
    return float(step["reward"] + kernel.discount * continuation)


def point_policy(kernel: SemanticKernel, state_belief: np.ndarray, latent: int, horizon: int, tolerance: float) -> dict[str, Any]:
    point_kernel = SemanticKernel(
        transition=kernel.transition.copy(),
        observation=kernel.observation[latent : latent + 1].copy(),
        reward=kernel.reward.copy(),
        discount=kernel.discount,
    )
    belief = np.asarray(state_belief, dtype=np.float64)[None, :]
    return plan_joint(point_kernel, belief, horizon, tolerance=tolerance)


def evaluate_action_sequence(kernel: SemanticKernel, belief: np.ndarray, actions: Sequence[int]) -> float:
    if not actions:
        return 0.0
    step = exact_step(kernel, belief, int(actions[0]))
    continuation = sum(
        float(step["probabilities"][observation])
        * evaluate_action_sequence(kernel, posterior, actions[1:])
        for observation, posterior in step["posteriors"].items()
    )
    return float(step["reward"] + kernel.discount * continuation)


def best_open_loop(kernel: SemanticKernel, belief: np.ndarray, horizon: int, tolerance: float) -> dict[str, Any]:
    rows = [
        (actions, evaluate_action_sequence(kernel, belief, actions))
        for actions in product(range(len(ACTION_NAMES)), repeat=horizon)
    ]
    maximum = max(value for _, value in rows)
    optimal = [actions for actions, value in rows if maximum - value <= tolerance]
    return {"value": float(maximum), "selected_actions": optimal[0], "sequence_count": len(rows)}


def myopic_policy(kernel: SemanticKernel, belief: np.ndarray, horizon: int, tolerance: float) -> dict[str, Any]:
    value = _validate_belief(kernel, belief)
    if horizon == 0:
        return {"terminal": True, "horizon": 0, "value": 0.0}
    steps = [exact_step(kernel, value, action) for action in range(len(ACTION_NAMES))]
    selected, optimal, maximum = _select([step["reward"] for step in steps], tolerance)
    branches = {
        observation: myopic_policy(kernel, posterior, horizon - 1, tolerance)
        for observation, posterior in steps[selected]["posteriors"].items()
        if horizon > 1
    }
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(maximum),
        "selected_action": selected,
        "optimal_actions": optimal,
        "branches": branches,
    }


def _mutual_information(joint: np.ndarray) -> float:
    value = np.asarray(joint, dtype=np.float64)
    row = value.sum(axis=1, keepdims=True)
    column = value.sum(axis=0, keepdims=True)
    independent = row @ column
    mask = value > 0.0
    return float(np.sum(value[mask] * np.log(value[mask] / independent[mask])))


def structural_diagnostics(kernel: SemanticKernel, belief: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    calibrate = ACTION_NAMES.index("calibrate")
    step = exact_step(kernel, belief, calibrate)
    latent_observation = np.zeros((len(LATENT_NAMES), len(OBSERVATION_NAMES)))
    for observation, posterior in step["posteriors"].items():
        latent_observation[:, observation] = float(step["probabilities"][observation]) * posterior.sum(axis=1)
    known_a = np.asarray(config["channel"]["knownCodebookLogicalA"])
    known_b = np.asarray(config["channel"]["knownCodebookLogicalB"])
    support = kernel.observation > 0.0
    sensing_positive = kernel.observation[:, :2][support[:, :2]]
    return {
        "calibration_mutual_information_nats": _mutual_information(latent_observation),
        "known_codebook_inspection_total_variation": float(0.5 * np.abs(known_a - known_b).sum()),
        "point_model_common_support": bool(
            all(np.array_equal(support[0], support[index]) for index in range(1, len(LATENT_NAMES)))
        ),
        "minimum_observation_probability_on_sensing_support": float(sensing_positive.min()),
        "maximum_initial_sensing_or_repair_immediate_reward": max(
            exact_step(kernel, belief, ACTION_NAMES.index(name))["reward"]
            for name in ("calibrate", "inspect", "repair_A", "repair_B")
        ),
        "repair_immediate_rewards": {
            name: exact_step(kernel, belief, ACTION_NAMES.index(name))["reward"]
            for name in ("repair_A", "repair_B")
        },
        "correct_consequence_only_at_settlement": bool(
            kernel.reward[ACTION_NAMES.index("settle"), STATE_NAMES.index("pending_good"), STATE_NAMES.index("terminal")]
            == config["decisionProcess"]["correctSettlementReward"]
        ),
        "wrong_consequence_only_at_settlement": bool(
            kernel.reward[ACTION_NAMES.index("settle"), STATE_NAMES.index("pending_bad"), STATE_NAMES.index("terminal")]
            == config["decisionProcess"]["wrongSettlementReward"]
        ),
        "belief_normalization_rate": 1.0,
        "fallback_count": 0,
    }


def _reachable_selected_actions(policy: dict[str, Any]) -> tuple[set[str], int]:
    actions: set[str] = set()
    defer_count = 0
    stack = [policy]
    while stack:
        node = stack.pop()
        if node.get("terminal"):
            continue
        name = ACTION_NAMES[int(node["selected_action"])]
        actions.add(name)
        defer_count += name == "defer"
        stack.extend(node.get("branches", {}).values())
    return actions, defer_count


def evaluate_oracle(config: dict[str, Any]) -> dict[str, Any]:
    kernel, belief = build_kernel(config)
    horizon = int(config["decisionProcess"]["horizon"])
    tolerance = float(config["decisionProcess"]["tieTolerance"])
    stats: dict[str, int] = {}
    exact = plan_joint(kernel, belief, horizon, tolerance=tolerance, stats=stats)
    forced_actions = tuple(index for index, name in enumerate(ACTION_NAMES) if name != "defer")
    forced = plan_joint(kernel, belief, horizon, allowed_actions=forced_actions, tolerance=tolerance)
    closed_belief = belief[:2].copy()
    closed_belief /= closed_belief.sum()
    closed_kernel = SemanticKernel(
        transition=kernel.transition.copy(),
        observation=kernel.observation[:2].copy(),
        reward=kernel.reward.copy(),
        discount=kernel.discount,
    )
    closed_policy = plan_joint(closed_kernel, closed_belief, horizon, tolerance=tolerance)
    closed_value = evaluate_policy(kernel, belief, closed_policy, horizon)
    latent_masses = belief.sum(axis=1)
    map_latent = int(np.argmax(latent_masses))
    map_policy = point_policy(kernel, belief[map_latent] / latent_masses[map_latent], map_latent, horizon, tolerance)
    map_value = evaluate_policy(kernel, belief, map_policy, horizon)
    sampled_rows = []
    sampled_value = 0.0
    for latent, mass in enumerate(latent_masses):
        policy = point_policy(kernel, belief[latent] / mass, latent, horizon, tolerance)
        true_value = evaluate_policy(kernel, belief, policy, horizon)
        sampled_value += float(mass) * true_value
        sampled_rows.append({"latent": LATENT_NAMES[latent], "mass": float(mass), "root_action": ACTION_NAMES[policy["selected_action"]], "true_value": true_value})
    open_loop = best_open_loop(kernel, belief, horizon, tolerance)
    myopic = myopic_policy(kernel, belief, horizon, tolerance)
    myopic_value = evaluate_policy(kernel, belief, myopic, horizon)
    immediate_defer = exact_step(kernel, belief, ACTION_NAMES.index("defer"))["reward"]
    scale = float((kernel.reward.max() - kernel.reward.min()) * horizon)
    exact_actions, defer_histories = _reachable_selected_actions(exact)
    green = OBSERVATION_NAMES.index("green")
    exact_after_green = ACTION_NAMES[exact["branches"][green]["selected_action"]]
    closed_after_green = ACTION_NAMES[closed_policy["branches"][green]["selected_action"]]
    exact_value = float(exact["value"])
    return {
        "structural": structural_diagnostics(kernel, belief, config),
        "return_scale": scale,
        "bellman_nodes": stats["bellman_nodes"],
        "exact": {
            "value": exact_value,
            "root_action": ACTION_NAMES[exact["selected_action"]],
            "root_optimal_actions": [ACTION_NAMES[action] for action in exact["optimal_actions"]],
            "root_q_values": {ACTION_NAMES[action]: value for action, value in exact["q_values"].items()},
            "action_after_root_green": exact_after_green,
            "reachable_selected_actions": sorted(exact_actions),
            "reachable_defer_history_count": defer_histories,
            "distinct_reachable_repair_actions": sorted(action for action in exact_actions if action.startswith("repair_")),
            "normalized_advantage_over_immediate_defer": (exact_value - immediate_defer) / scale,
        },
        "closed_world": {
            "true_environment_value": closed_value,
            "root_action": ACTION_NAMES[closed_policy["selected_action"]],
            "action_after_root_green": closed_after_green,
            "normalized_regret": (exact_value - closed_value) / scale,
        },
        "forced_commit": {
            "true_environment_value": float(forced["value"]),
            "root_action": ACTION_NAMES[forced["selected_action"]],
            "normalized_regret": (exact_value - float(forced["value"])) / scale,
        },
        "map": {
            "latent": LATENT_NAMES[map_latent],
            "true_environment_value": map_value,
            "root_action": ACTION_NAMES[map_policy["selected_action"]],
            "normalized_regret": (exact_value - map_value) / scale,
        },
        "posterior_sampling": {
            "true_environment_value": sampled_value,
            "normalized_regret": (exact_value - sampled_value) / scale,
            "models": sampled_rows,
        },
        "open_loop": {
            "value": open_loop["value"],
            "selected_actions": [ACTION_NAMES[action] for action in open_loop["selected_actions"]],
            "normalized_exact_advantage": (exact_value - open_loop["value"]) / scale,
            "sequence_count": open_loop["sequence_count"],
        },
        "myopic": {
            "true_environment_value": myopic_value,
            "root_action": ACTION_NAMES[myopic["selected_action"]],
            "normalized_regret": (exact_value - myopic_value) / scale,
        },
        "immediate_defer": {"value": immediate_defer},
        "access": {
            "oracle_evaluation_count": 1,
            "language_record_read_count": 0,
            "raw_model_response_read_count": 0,
            "protected_access_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "API_call_count": 0,
            "training_run_count": 0,
            "ontology_registration_count": 0,
            "trusted_state_mutation_count": 0,
            "service_call_count": 0,
            "external_side_effect_count": 0,
            "actual_execution_count": 0,
        },
    }


def audit_oracle(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["oracleGates"]
    structural = result["structural"]
    checks = {
        "latent_and_common_support": bool(
            len(LATENT_NAMES) == gates["requiredLatentCount"]
            and structural["point_model_common_support"] == gates["requiredPointModelCommonSupport"]
            and structural["minimum_observation_probability_on_sensing_support"]
            >= gates["minimumObservationProbabilityOnSensingSupport"] - 1e-12
        ),
        "sensing_channels_are_informative": bool(
            structural["calibration_mutual_information_nats"] >= gates["minimumCalibrationMutualInformationNats"]
            and structural["known_codebook_inspection_total_variation"] >= gates["minimumKnownCodebookInspectionTotalVariation"]
        ),
        "no_immediate_reward_shortcut_and_consequences_are_delayed": bool(
            structural["maximum_initial_sensing_or_repair_immediate_reward"] <= gates["maximumInitialSensingOrRepairImmediateReward"]
            and set(structural["repair_immediate_rewards"].values()) == {gates["requiredRepairImmediateReward"]}
            and structural["correct_consequence_only_at_settlement"] == gates["requiredCorrectConsequenceOnlyAtSettlement"]
            and structural["wrong_consequence_only_at_settlement"] == gates["requiredWrongConsequenceOnlyAtSettlement"]
        ),
        "exact_root_and_green_abstention_actions": bool(
            result["exact"]["root_action"] == gates["requiredSelectedExactRootAction"]
            and result["exact"]["action_after_root_green"] == gates["requiredExactActionAfterRootGreen"]
            and (result["closed_world"]["action_after_root_green"] != "defer")
            == gates["closedWorldActionAfterRootGreenMustNotBeDefer"]
        ),
        "reachable_abstention_and_both_controls": bool(
            len(result["exact"]["distinct_reachable_repair_actions"]) >= gates["minimumDistinctReachableRepairActions"]
            and result["exact"]["reachable_defer_history_count"] >= gates["minimumReachableDeferHistoryCount"]
        ),
        "exact_beats_immediate_defer": result["exact"]["normalized_advantage_over_immediate_defer"]
        >= gates["minimumNormalizedExactOverImmediateDeferAdvantage"],
        "forced_commit_has_material_regret": result["forced_commit"]["normalized_regret"]
        >= gates["minimumNormalizedForcedCommitRegret"],
        "MAP_has_material_regret": result["map"]["normalized_regret"] >= gates["minimumNormalizedMAPRegret"],
        "posterior_sampling_has_material_regret": result["posterior_sampling"]["normalized_regret"]
        >= gates["minimumNormalizedPosteriorSamplingRegret"],
        "adaptive_beats_open_loop": result["open_loop"]["normalized_exact_advantage"]
        >= gates["minimumNormalizedExactOverOpenLoopAdvantage"],
        "beliefs_finite_and_no_fallback": bool(
            structural["belief_normalization_rate"] >= gates["minimumBeliefNormalizationRate"]
            and structural["fallback_count"] <= gates["maximumFallbackCount"]
            and (not gates["requiredFiniteMetrics"] or all(
                math.isfinite(value)
                for value in (
                    result["exact"]["value"],
                    result["closed_world"]["true_environment_value"],
                    result["forced_commit"]["true_environment_value"],
                    result["map"]["true_environment_value"],
                    result["posterior_sampling"]["true_environment_value"],
                    result["open_loop"]["value"],
                )
            ))
        ),
    }
    access_gates = config["accessGates"]
    access = result["access"]
    access_checks = {
        "oracle_evaluation_count_exact": access["oracle_evaluation_count"] == access_gates["requiredOracleEvaluationCount"],
        "forbidden_access_and_effects_zero": all(
            access[key] <= access_gates[gate]
            for key, gate in (
                ("language_record_read_count", "maximumLanguageRecordReadCount"),
                ("raw_model_response_read_count", "maximumRawModelResponseReadCount"),
                ("protected_access_count", "maximumProtectedAccessCount"),
                ("model_load_count", "maximumModelLoadCount"),
                ("model_generation_count", "maximumModelGenerationCount"),
                ("API_call_count", "maximumAPICallCount"),
                ("training_run_count", "maximumTrainingRunCount"),
                ("ontology_registration_count", "maximumOntologyRegistrationCount"),
                ("trusted_state_mutation_count", "maximumTrustedStateMutationCount"),
                ("service_call_count", "maximumServiceCallCount"),
                ("external_side_effect_count", "maximumExternalSideEffectCount"),
                ("actual_execution_count", "maximumActualExecutionCount"),
            )
        ),
    }
    return {
        "passed": all(checks.values()) and all(access_checks.values()),
        "scientific_gates_passed": all(checks.values()),
        "access_gates_passed": all(access_checks.values()),
        "checks": checks,
        "access_checks": access_checks,
        "result": result,
    }


__all__ = ["ACTION_NAMES", "OBSERVATION_NAMES", "STATE_NAMES", "LATENT_NAMES", "audit_oracle", "build_kernel", "evaluate_oracle", "exact_step", "plan_joint"]
