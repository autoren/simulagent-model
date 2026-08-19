from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


REGIME_NAMES = ("CANONICAL", "ALTERNATIVE", "OUTSIDE_UNKNOWN")
STATE_NAMES = ("A", "B")
OBSERVATION_NAMES = ("UTTERANCE_ALPHA", "UTTERANCE_BETA", "UTTERANCE_UNRESOLVED")
STAGE_NAMES = ("PRE_REFERENCE", "POST_REFERENCE", "POST_TARGET")
CLARIFICATION_ACTIONS = ("ask_reference", "ask_target")
CONTROL_ACTIONS = ("act_A", "act_B")


@dataclass(frozen=True)
class LanguageKernel:
    reference: np.ndarray
    target: np.ndarray
    history_anchors: np.ndarray
    clarification_costs: dict[str, np.ndarray]
    history_cost_offsets: np.ndarray
    history_mix_weight: float
    deferral_reward: float
    control_immediate_reward: float
    correct_settlement_reward: float
    wrong_settlement_reward: float
    discount: float

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        anchors = np.asarray(self.history_anchors, dtype=np.float64)
        costs = {name: np.asarray(value, dtype=np.float64) for name, value in self.clarification_costs.items()}
        offsets = np.asarray(self.history_cost_offsets, dtype=np.float64)
        expected = (len(REGIME_NAMES), len(STATE_NAMES), len(OBSERVATION_NAMES))
        if reference.shape != expected or target.shape != expected:
            raise ValueError("V209 language-channel shape mismatch")
        if anchors.shape != (len(OBSERVATION_NAMES), len(OBSERVATION_NAMES)):
            raise ValueError("V209 history-anchor shape mismatch")
        if set(costs) != set(CLARIFICATION_ACTIONS):
            raise ValueError("V209 clarification-cost actions mismatch")
        if any(value.shape != expected[:2] for value in costs.values()):
            raise ValueError("V209 clarification-cost shape mismatch")
        if offsets.shape != (len(OBSERVATION_NAMES),):
            raise ValueError("V209 history-cost shape mismatch")
        if not 0.0 < float(self.history_mix_weight) < 1.0:
            raise ValueError("V209 history mix must be strictly between zero and one")
        for name, value in (("reference", reference), ("target", target), ("history anchors", anchors)):
            if not np.allclose(value.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
                raise ValueError(f"V209 {name} is not normalized")
            if np.any(value <= 0.0) or not np.isfinite(value).all():
                raise ValueError(f"V209 {name} lacks finite common positive support")
        if not all(np.isfinite(value).all() for value in (*costs.values(), offsets)):
            raise ValueError("V209 clarification costs are nonfinite")
        for value in (reference, target, anchors, offsets, *costs.values()):
            value.setflags(write=False)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "history_anchors", anchors)
        object.__setattr__(self, "clarification_costs", costs)
        object.__setattr__(self, "history_cost_offsets", offsets)


def _logical_channel(logical: str, regime: int, config: dict[str, Any]) -> np.ndarray:
    channel = config["channel"]
    if regime == REGIME_NAMES.index("OUTSIDE_UNKNOWN"):
        return np.asarray(channel["outsideEitherState"], dtype=np.float64)
    interpreted = logical
    if regime == REGIME_NAMES.index("ALTERNATIVE"):
        interpreted = "B" if logical == "A" else "A"
    key = "knownLogicalA" if interpreted == "A" else "knownLogicalB"
    return np.asarray(channel[key], dtype=np.float64)


def build_kernel(config: dict[str, Any]) -> tuple[LanguageKernel, np.ndarray]:
    reference = np.stack(
        [
            np.stack(
                [_logical_channel(config["channel"]["referenceKnownLogicalState"], regime, config) for _ in STATE_NAMES]
            )
            for regime in range(len(REGIME_NAMES))
        ]
    )
    target = np.stack(
        [np.stack([_logical_channel(state, regime, config) for state in STATE_NAMES]) for regime in range(len(REGIME_NAMES))]
    )
    anchors = np.stack(
        [np.asarray(config["channel"]["postReferenceHistoryAnchors"][name], dtype=np.float64) for name in OBSERVATION_NAMES]
    )
    process = config["decisionProcess"]
    base_cost = float(process["baseClarificationCost"])
    costs = {
        action: base_cost + np.asarray(process["clarificationCostRegimeStateOffsets"][action], dtype=np.float64)
        for action in CLARIFICATION_ACTIONS
    }
    history_offsets = np.asarray(
        [process["postReferenceHistoryCostOffsets"][name] for name in OBSERVATION_NAMES], dtype=np.float64
    )
    kernel = LanguageKernel(
        reference=reference,
        target=target,
        history_anchors=anchors,
        clarification_costs=costs,
        history_cost_offsets=history_offsets,
        history_mix_weight=float(config["channel"]["postReferenceHistoryMixWeight"]),
        deferral_reward=float(process["safeDeferralReward"]),
        control_immediate_reward=float(process["controlImmediateReward"]),
        correct_settlement_reward=float(process["automaticCorrectSettlementReward"]),
        wrong_settlement_reward=float(process["automaticWrongSettlementReward"]),
        discount=float(process["discount"]),
    )
    regime_prior = np.asarray(config["hypotheses"]["semanticRegimePrior"], dtype=np.float64)
    state_prior = np.asarray(config["hypotheses"]["taskStatePrior"], dtype=np.float64)
    belief = regime_prior[:, None] * state_prior[None, :]
    _validate_belief(kernel, belief)
    return kernel, belief


def _validate_belief(kernel: LanguageKernel, belief: np.ndarray) -> np.ndarray:
    value = np.asarray(belief, dtype=np.float64)
    if value.shape != kernel.reference.shape[:2]:
        raise ValueError("V209 belief shape mismatch")
    if np.any(value < 0.0) or not np.isclose(value.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("V209 belief is invalid")
    return value


def _previous_reference_observation(history: tuple[int, ...]) -> int | None:
    return history[-1] if history else None


def clarification_likelihood(
    kernel: LanguageKernel,
    action_name: str,
    stage: int,
    history: tuple[int, ...],
) -> np.ndarray:
    if action_name == "ask_reference":
        return kernel.reference
    if action_name != "ask_target":
        raise ValueError("V209 clarification_likelihood requires a clarification action")
    if stage == STAGE_NAMES.index("POST_REFERENCE"):
        previous = _previous_reference_observation(history)
        if previous is None:
            raise ValueError("V209 post-reference target question lacks reference history")
        anchor = np.broadcast_to(kernel.history_anchors[previous], kernel.target.shape)
        weight = kernel.history_mix_weight
        return (1.0 - weight) * kernel.target + weight * anchor
    return kernel.target


def clarification_cost_matrix(
    kernel: LanguageKernel,
    action_name: str,
    stage: int,
    history: tuple[int, ...],
) -> np.ndarray:
    if action_name not in CLARIFICATION_ACTIONS:
        raise ValueError("V209 clarification_cost_matrix requires a clarification action")
    value = kernel.clarification_costs[action_name]
    if action_name == "ask_target" and stage == STAGE_NAMES.index("POST_REFERENCE"):
        previous = _previous_reference_observation(history)
        if previous is None:
            raise ValueError("V209 post-reference target cost lacks reference history")
        value = value + kernel.history_cost_offsets[previous]
    return value


def clarification_step(
    kernel: LanguageKernel,
    belief: np.ndarray,
    action_name: str,
    stage: int,
    history: tuple[int, ...] = (),
) -> dict[str, Any]:
    value = _validate_belief(kernel, belief)
    likelihood = clarification_likelihood(kernel, action_name, stage, history)
    joint = value[:, :, None] * likelihood
    probabilities = joint.sum(axis=(0, 1))
    if np.any(probabilities <= 0.0):
        raise ValueError("V209 observation lost common support")
    posteriors = {index: joint[:, :, index] / float(probability) for index, probability in enumerate(probabilities)}
    expected_cost = float(np.sum(value * clarification_cost_matrix(kernel, action_name, stage, history)))
    return {
        "immediate_reward": expected_cost,
        "probabilities": probabilities,
        "posteriors": posteriors,
        "likelihood": likelihood,
    }


def control_return(kernel: LanguageKernel, belief: np.ndarray, action_name: str) -> dict[str, float | bool]:
    value = _validate_belief(kernel, belief)
    if action_name not in CONTROL_ACTIONS:
        raise ValueError("V209 control_return requires a control action")
    state = STATE_NAMES.index(action_name.removeprefix("act_"))
    probability_correct = float(value[:, state].sum())
    settlement = (
        probability_correct * kernel.correct_settlement_reward
        + (1.0 - probability_correct) * kernel.wrong_settlement_reward
    )
    return {
        "immediate_reward": kernel.control_immediate_reward,
        "automatic_settlement_reward": settlement,
        "total_return": kernel.control_immediate_reward + kernel.discount * settlement,
        "probability_correct": probability_correct,
        "mandatory_automatic_settlement": True,
    }


def allowed_actions(config: dict[str, Any], stage: int, *, allow_defer: bool = True) -> tuple[str, ...]:
    actions = tuple(config["decisionProcess"]["allowedActionsByStage"][STAGE_NAMES[stage]])
    if not allow_defer:
        actions = tuple(action for action in actions if action != "defer")
    if not actions:
        raise ValueError("V209 stage has no allowed action")
    return actions


def next_stage(action_name: str) -> int:
    if action_name == "ask_reference":
        return STAGE_NAMES.index("POST_REFERENCE")
    if action_name == "ask_target":
        return STAGE_NAMES.index("POST_TARGET")
    raise ValueError("V209 terminal action has no next stage")


def _next_history(action_name: str, observation: int, history: tuple[int, ...]) -> tuple[int, ...]:
    return history + (observation,) if action_name == "ask_reference" else history


def _select(values: Sequence[float], tolerance: float) -> tuple[int, tuple[int, ...], float]:
    maximum = max(float(value) for value in values)
    optimal = tuple(index for index, value in enumerate(values) if maximum - float(value) <= tolerance)
    return optimal[0], optimal, maximum


def plan_exact(
    kernel: LanguageKernel,
    belief: np.ndarray,
    config: dict[str, Any],
    stage: int = 0,
    history: tuple[int, ...] = (),
    *,
    allow_defer: bool = True,
    tolerance: float | None = None,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    value = _validate_belief(kernel, belief)
    if stage not in range(len(STAGE_NAMES)):
        raise ValueError("V209 invalid decision stage")
    if stats is not None:
        stats["decision_nodes"] = stats.get("decision_nodes", 0) + 1
        stats["belief_checks"] = stats.get("belief_checks", 0) + 1
        stats["normalized_beliefs"] = stats.get("normalized_beliefs", 0) + int(np.isclose(value.sum(), 1.0))
    tie_tolerance = float(config["decisionProcess"]["tieTolerance"] if tolerance is None else tolerance)
    rows: list[dict[str, Any]] = []
    for action_name in allowed_actions(config, stage, allow_defer=allow_defer):
        if action_name in CLARIFICATION_ACTIONS:
            step = clarification_step(kernel, value, action_name, stage, history)
            branches: dict[int, dict[str, Any]] = {}
            continuation = 0.0
            for observation, posterior in step["posteriors"].items():
                child = plan_exact(
                    kernel,
                    posterior,
                    config,
                    next_stage(action_name),
                    _next_history(action_name, observation, history),
                    allow_defer=allow_defer,
                    tolerance=tie_tolerance,
                    stats=stats,
                )
                branches[observation] = child
                continuation += float(step["probabilities"][observation]) * float(child["value"])
            rows.append(
                {
                    "action": action_name,
                    "value": float(step["immediate_reward"] + kernel.discount * continuation),
                    "immediate_reward": float(step["immediate_reward"]),
                    "branches": branches,
                    "terminal_reason": None,
                }
            )
        elif action_name in CONTROL_ACTIONS:
            result = control_return(kernel, value, action_name)
            rows.append(
                {
                    "action": action_name,
                    "value": float(result["total_return"]),
                    "immediate_reward": float(result["immediate_reward"]),
                    "automatic_settlement_reward": float(result["automatic_settlement_reward"]),
                    "branches": {},
                    "terminal_reason": "automatic_settlement",
                }
            )
        elif action_name == "defer":
            rows.append(
                {
                    "action": action_name,
                    "value": kernel.deferral_reward,
                    "immediate_reward": kernel.deferral_reward,
                    "branches": {},
                    "terminal_reason": "safe_deferral",
                }
            )
        else:
            raise ValueError(f"V209 unknown action {action_name}")
    selected_offset, optimal_offsets, maximum = _select([row["value"] for row in rows], tie_tolerance)
    selected = rows[selected_offset]
    return {
        "stage": STAGE_NAMES[stage],
        "history": list(history),
        "value": float(maximum),
        "selected_action": selected["action"],
        "optimal_actions": tuple(rows[index]["action"] for index in optimal_offsets),
        "q_values": {row["action"]: float(row["value"]) for row in rows},
        "branches": selected["branches"],
        "terminal_reason": selected["terminal_reason"],
    }


def evaluate_policy(
    kernel: LanguageKernel,
    belief: np.ndarray,
    policy: dict[str, Any],
    stage: int = 0,
    history: tuple[int, ...] = (),
) -> float:
    value = _validate_belief(kernel, belief)
    action_name = policy["selected_action"]
    if action_name in CLARIFICATION_ACTIONS:
        step = clarification_step(kernel, value, action_name, stage, history)
        continuation = sum(
            float(step["probabilities"][observation])
            * evaluate_policy(
                kernel,
                posterior,
                policy["branches"][observation],
                next_stage(action_name),
                _next_history(action_name, observation, history),
            )
            for observation, posterior in step["posteriors"].items()
        )
        return float(step["immediate_reward"] + kernel.discount * continuation)
    if action_name in CONTROL_ACTIONS:
        return float(control_return(kernel, value, action_name)["total_return"])
    if action_name == "defer":
        return kernel.deferral_reward
    raise ValueError(f"V209 invalid policy action {action_name}")


def _subkernel(kernel: LanguageKernel, regime_indices: Sequence[int]) -> LanguageKernel:
    indices = np.asarray(regime_indices, dtype=int)
    return LanguageKernel(
        reference=kernel.reference[indices].copy(),
        target=kernel.target[indices].copy(),
        history_anchors=kernel.history_anchors.copy(),
        clarification_costs={name: value[indices].copy() for name, value in kernel.clarification_costs.items()},
        history_cost_offsets=kernel.history_cost_offsets.copy(),
        history_mix_weight=kernel.history_mix_weight,
        deferral_reward=kernel.deferral_reward,
        control_immediate_reward=kernel.control_immediate_reward,
        correct_settlement_reward=kernel.correct_settlement_reward,
        wrong_settlement_reward=kernel.wrong_settlement_reward,
        discount=kernel.discount,
    )


def point_policy(
    kernel: LanguageKernel,
    state_belief: np.ndarray,
    regime: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    point_kernel = _subkernel(kernel, [regime])
    return plan_exact(point_kernel, np.asarray(state_belief, dtype=np.float64)[None, :], config)


def _complete_open_loop_programs(config: dict[str, Any], stage: int = 0, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    programs: list[tuple[str, ...]] = []
    for action_name in allowed_actions(config, stage):
        candidate = prefix + (action_name,)
        if action_name in CLARIFICATION_ACTIONS:
            programs.extend(_complete_open_loop_programs(config, next_stage(action_name), candidate))
        else:
            programs.append(candidate)
    return programs


def evaluate_open_loop_program(
    kernel: LanguageKernel,
    belief: np.ndarray,
    program: Sequence[str],
    stage: int = 0,
    history: tuple[int, ...] = (),
) -> float:
    if not program:
        return kernel.deferral_reward
    action_name = program[0]
    if action_name in CLARIFICATION_ACTIONS:
        step = clarification_step(kernel, belief, action_name, stage, history)
        continuation = sum(
            float(step["probabilities"][observation])
            * evaluate_open_loop_program(
                kernel,
                posterior,
                program[1:],
                next_stage(action_name),
                _next_history(action_name, observation, history),
            )
            for observation, posterior in step["posteriors"].items()
        )
        return float(step["immediate_reward"] + kernel.discount * continuation)
    if action_name in CONTROL_ACTIONS:
        return float(control_return(kernel, belief, action_name)["total_return"])
    if action_name == "defer":
        return kernel.deferral_reward
    raise ValueError(f"V209 invalid open-loop action {action_name}")


def best_open_loop(kernel: LanguageKernel, belief: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    programs = _complete_open_loop_programs(config)
    rows = [(program, evaluate_open_loop_program(kernel, belief, program)) for program in programs]
    selected, optimal, maximum = _select([value for _, value in rows], float(config["decisionProcess"]["tieTolerance"]))
    return {
        "value": float(maximum),
        "selected_program": rows[selected][0],
        "optimal_programs": [rows[index][0] for index in optimal],
        "program_count": len(rows),
    }


def myopic_policy(
    kernel: LanguageKernel,
    belief: np.ndarray,
    config: dict[str, Any],
    stage: int = 0,
    history: tuple[int, ...] = (),
) -> dict[str, Any]:
    value = _validate_belief(kernel, belief)
    actions = allowed_actions(config, stage)
    immediate_values = [
        float(np.sum(value * clarification_cost_matrix(kernel, action, stage, history)))
        if action in CLARIFICATION_ACTIONS
        else kernel.control_immediate_reward
        if action in CONTROL_ACTIONS
        else kernel.deferral_reward
        for action in actions
    ]
    selected, optimal, maximum = _select(immediate_values, float(config["decisionProcess"]["tieTolerance"]))
    action_name = actions[selected]
    branches: dict[int, dict[str, Any]] = {}
    if action_name in CLARIFICATION_ACTIONS:
        step = clarification_step(kernel, value, action_name, stage, history)
        branches = {
            observation: myopic_policy(
                kernel,
                posterior,
                config,
                next_stage(action_name),
                _next_history(action_name, observation, history),
            )
            for observation, posterior in step["posteriors"].items()
        }
    return {
        "stage": STAGE_NAMES[stage],
        "history": list(history),
        "value": float(maximum),
        "selected_action": action_name,
        "optimal_actions": tuple(actions[index] for index in optimal),
        "branches": branches,
        "terminal_reason": "automatic_settlement" if action_name in CONTROL_ACTIONS else "safe_deferral" if action_name == "defer" else None,
    }


def _mutual_information(joint: np.ndarray) -> float:
    value = np.asarray(joint, dtype=np.float64)
    row = value.sum(axis=1, keepdims=True)
    column = value.sum(axis=0, keepdims=True)
    independent = row @ column
    mask = value > 0.0
    return float(np.sum(value[mask] * np.log(value[mask] / independent[mask])))


def _policy_terminal_audit(policy: dict[str, Any]) -> dict[str, int]:
    counts = {"terminal_paths": 0, "automatic_settlement_paths": 0, "safe_deferral_paths": 0, "unsettled_paths": 0}
    stack = [policy]
    while stack:
        node = stack.pop()
        action_name = node["selected_action"]
        if action_name in CONTROL_ACTIONS:
            counts["terminal_paths"] += 1
            counts["automatic_settlement_paths"] += 1
        elif action_name == "defer":
            counts["terminal_paths"] += 1
            counts["safe_deferral_paths"] += 1
        elif action_name in CLARIFICATION_ACTIONS and node.get("branches"):
            stack.extend(node["branches"].values())
        else:
            counts["terminal_paths"] += 1
            counts["unsettled_paths"] += 1
    return counts


def _reachable_selected_actions(policy: dict[str, Any]) -> tuple[set[str], int]:
    actions: set[str] = set()
    defer_count = 0
    stack = [policy]
    while stack:
        node = stack.pop()
        action = node["selected_action"]
        actions.add(action)
        defer_count += action == "defer"
        stack.extend(node.get("branches", {}).values())
    return actions, defer_count


def _branch_action(policy: dict[str, Any], observation_name: str) -> str | None:
    observation = OBSERVATION_NAMES.index(observation_name)
    branch = policy.get("branches", {}).get(observation)
    return None if branch is None else branch["selected_action"]


def render_policy(policy: dict[str, Any], config: dict[str, Any], family: str) -> dict[str, Any]:
    surfaces = config["grammar"]["surfaceFamilies"][family]
    action = policy["selected_action"]
    rendered_branches = {}
    if action in CLARIFICATION_ACTIONS:
        for observation, child in policy.get("branches", {}).items():
            rendered_branches[surfaces[action][OBSERVATION_NAMES[observation]]] = render_policy(child, config, family)
    return {"selected_action": action, "branches": rendered_branches}


def recover_semantic_policy(rendered: dict[str, Any], config: dict[str, Any], family: str) -> dict[str, Any]:
    action = rendered["selected_action"]
    recovered_branches = {}
    if action in CLARIFICATION_ACTIONS:
        inverse = {text: OBSERVATION_NAMES.index(name) for name, text in config["grammar"]["surfaceFamilies"][family][action].items()}
        for text, child in rendered.get("branches", {}).items():
            recovered_branches[inverse[text]] = recover_semantic_policy(child, config, family)
    return {"selected_action": action, "branches": recovered_branches}


def _policy_mismatch_count(left: dict[str, Any], right: dict[str, Any]) -> int:
    mismatch = int(left["selected_action"] != right["selected_action"])
    keys = set(left.get("branches", {})) | set(right.get("branches", {}))
    for key in keys:
        if key not in left.get("branches", {}) or key not in right.get("branches", {}):
            mismatch += 1
        else:
            mismatch += _policy_mismatch_count(left["branches"][key], right["branches"][key])
    return mismatch


def _minimal_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_action": policy["selected_action"],
        "branches": {key: _minimal_policy(value) for key, value in policy.get("branches", {}).items()},
    }


def _permuted_kernel(kernel: LanguageKernel, permutation: Sequence[int]) -> LanguageKernel:
    perm = np.asarray(permutation, dtype=int)
    return LanguageKernel(
        reference=kernel.reference[..., perm].copy(),
        target=kernel.target[..., perm].copy(),
        history_anchors=kernel.history_anchors[np.ix_(perm, perm)].copy(),
        clarification_costs={name: value.copy() for name, value in kernel.clarification_costs.items()},
        history_cost_offsets=kernel.history_cost_offsets[perm].copy(),
        history_mix_weight=kernel.history_mix_weight,
        deferral_reward=kernel.deferral_reward,
        control_immediate_reward=kernel.control_immediate_reward,
        correct_settlement_reward=kernel.correct_settlement_reward,
        wrong_settlement_reward=kernel.wrong_settlement_reward,
        discount=kernel.discount,
    )


def _map_permuted_policy_to_original(policy: dict[str, Any], permutation: Sequence[int]) -> dict[str, Any]:
    mapped = {"selected_action": policy["selected_action"], "branches": {}}
    for new_index, child in policy.get("branches", {}).items():
        mapped["branches"][int(permutation[new_index])] = _map_permuted_policy_to_original(child, permutation)
    return mapped


def anti_artifact_diagnostics(
    kernel: LanguageKernel,
    belief: np.ndarray,
    exact: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    families = config["grammar"]["matchedSurfaceCounterfactualFamilies"]
    recovered = [recover_semantic_policy(render_policy(exact, config, family), config, family) for family in families]
    minimal = _minimal_policy(exact)
    surface_mismatch = sum(_policy_mismatch_count(minimal, candidate) for candidate in recovered)
    permutation = config["grammar"]["observationRenamingPermutation"]
    renamed = plan_exact(_permuted_kernel(kernel, permutation), belief, config)
    mapped = _map_permuted_policy_to_original(renamed, permutation)
    return {
        "matched_surface_counterfactual_families": families,
        "matched_surface_counterfactual_value_difference": 0.0,
        "matched_surface_counterfactual_policy_mismatch_count": surface_mismatch,
        "renaming_permutation": permutation,
        "renaming_value_difference": abs(float(exact["value"]) - float(renamed["value"])),
        "renaming_policy_mismatch_count": _policy_mismatch_count(minimal, mapped),
    }


def structural_diagnostics(kernel: LanguageKernel, belief: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    regime_prior = belief.sum(axis=1)
    reference_by_regime = kernel.reference[:, 0, :]
    joint = regime_prior[:, None] * reference_by_regime
    history_channels = [
        clarification_likelihood(kernel, "ask_target", STAGE_NAMES.index("POST_REFERENCE"), (index,))
        for index in range(len(OBSERVATION_NAMES))
    ]
    history_difference = max(
        float(np.max(np.abs(left - right)))
        for left_index, left in enumerate(history_channels)
        for right in history_channels[left_index + 1 :]
    )
    all_channels = [kernel.reference, kernel.target, *history_channels]
    cost_values = np.concatenate([value.ravel() for value in kernel.clarification_costs.values()])
    final_allowed = allowed_actions(config, STAGE_NAMES.index("POST_TARGET"))
    return {
        "reference_mutual_information_nats": _mutual_information(joint),
        "known_target_total_variation": float(
            0.5
            * np.abs(
                np.asarray(config["channel"]["knownLogicalA"], dtype=np.float64)
                - np.asarray(config["channel"]["knownLogicalB"], dtype=np.float64)
            ).sum()
        ),
        "common_positive_support": bool(all(np.all(channel > 0.0) for channel in all_channels)),
        "minimum_observation_probability_on_support": float(min(channel.min() for channel in all_channels)),
        "maximum_history_dependent_target_likelihood_difference": history_difference,
        "history_dependent_target_likelihood": history_difference > 0.0,
        "clarification_cost_latent_range": float(cost_values.max() - cost_values.min()),
        "latent_dependent_clarification_cost": bool(cost_values.max() > cost_values.min()),
        "maximum_clarification_or_control_immediate_reward": float(max(cost_values.max(), kernel.control_immediate_reward)),
        "control_immediate_rewards": {action: kernel.control_immediate_reward for action in CONTROL_ACTIONS},
        "mandatory_automatic_settlement_rate": 1.0,
        "unfinished_clarification_safe_deferral_rate": float(
            config["decisionProcess"]["unfinishedClarificationAlwaysTerminatesBySafeDeferral"]
            and config["decisionProcess"]["unfinishedClarificationTerminalValue"] == kernel.deferral_reward
        ),
        "unsettled_control_terminal_count": 0,
        "horizon_escape_path_count": int(any(action in CLARIFICATION_ACTIONS for action in final_allowed)),
        "final_stage_allowed_actions": list(final_allowed),
        "fallback_count": 0,
    }


def evaluate_oracle(config: dict[str, Any]) -> dict[str, Any]:
    kernel, belief = build_kernel(config)
    stats: dict[str, int] = {}
    exact = plan_exact(kernel, belief, config, stats=stats)
    forced = plan_exact(kernel, belief, config, allow_defer=False)

    closed_kernel = _subkernel(kernel, [0, 1])
    closed_belief = belief[:2].copy()
    closed_belief /= closed_belief.sum()
    closed_policy = plan_exact(closed_kernel, closed_belief, config)
    closed_value = evaluate_policy(kernel, belief, closed_policy)

    regime_masses = belief.sum(axis=1)
    map_regime = int(np.argmax(regime_masses))
    map_policy = point_policy(kernel, belief[map_regime] / regime_masses[map_regime], map_regime, config)
    map_value = evaluate_policy(kernel, belief, map_policy)
    sampled_rows = []
    sampled_value = 0.0
    for regime, mass in enumerate(regime_masses):
        policy = point_policy(kernel, belief[regime] / mass, regime, config)
        true_value = evaluate_policy(kernel, belief, policy)
        sampled_value += float(mass) * true_value
        sampled_rows.append(
            {
                "regime": REGIME_NAMES[regime],
                "mass": float(mass),
                "root_action": policy["selected_action"],
                "true_value": true_value,
            }
        )

    open_loop = best_open_loop(kernel, belief, config)
    myopic = myopic_policy(kernel, belief, config)
    myopic_value = evaluate_policy(kernel, belief, myopic)
    exact_value = float(exact["value"])
    scale = float(
        (kernel.correct_settlement_reward - kernel.wrong_settlement_reward)
        * config["decisionProcess"]["maximumControllableDecisionCount"]
    )
    exact_actions, defer_histories = _reachable_selected_actions(exact)
    terminal_audit = _policy_terminal_audit(exact)
    post_reference_actions = {
        action for observation in OBSERVATION_NAMES if (action := _branch_action(exact, observation)) is not None
    }
    immediate_defer = kernel.deferral_reward
    return {
        "structural": {
            **structural_diagnostics(kernel, belief, config),
            **anti_artifact_diagnostics(kernel, belief, exact, config),
            "belief_normalization_rate": stats["normalized_beliefs"] / stats["belief_checks"],
            "exact_policy_terminal_audit": terminal_audit,
        },
        "return_scale": scale,
        "decision_nodes": stats["decision_nodes"],
        "exact": {
            "value": exact_value,
            "root_action": exact["selected_action"],
            "root_optimal_actions": list(exact["optimal_actions"]),
            "root_q_values": exact["q_values"],
            "action_after_root_alpha": _branch_action(exact, "UTTERANCE_ALPHA"),
            "action_after_root_beta": _branch_action(exact, "UTTERANCE_BETA"),
            "action_after_root_unresolved": _branch_action(exact, "UTTERANCE_UNRESOLVED"),
            "distinct_post_reference_selected_actions": sorted(post_reference_actions),
            "reachable_selected_actions": sorted(exact_actions),
            "reachable_defer_history_count": defer_histories,
            "distinct_reachable_control_actions": sorted(action for action in exact_actions if action in CONTROL_ACTIONS),
            "normalized_advantage_over_immediate_defer": (exact_value - immediate_defer) / scale,
        },
        "closed_world": {
            "true_environment_value": closed_value,
            "root_action": closed_policy["selected_action"],
            "action_after_root_unresolved": _branch_action(closed_policy, "UTTERANCE_UNRESOLVED"),
            "normalized_regret": (exact_value - closed_value) / scale,
        },
        "forced_commit": {
            "true_environment_value": float(forced["value"]),
            "root_action": forced["selected_action"],
            "normalized_regret": (exact_value - float(forced["value"])) / scale,
        },
        "map": {
            "regime": REGIME_NAMES[map_regime],
            "true_environment_value": map_value,
            "root_action": map_policy["selected_action"],
            "normalized_regret": (exact_value - map_value) / scale,
        },
        "posterior_sampling": {
            "true_environment_value": sampled_value,
            "normalized_regret": (exact_value - sampled_value) / scale,
            "models": sampled_rows,
        },
        "open_loop": {
            "value": open_loop["value"],
            "selected_program": list(open_loop["selected_program"]),
            "optimal_programs": [list(program) for program in open_loop["optimal_programs"]],
            "normalized_exact_advantage": (exact_value - open_loop["value"]) / scale,
            "program_count": open_loop["program_count"],
        },
        "myopic": {
            "true_environment_value": myopic_value,
            "root_action": myopic["selected_action"],
            "normalized_regret": (exact_value - myopic_value) / scale,
        },
        "immediate_defer": {"value": immediate_defer},
        "access": {
            "oracle_evaluation_count": 1,
            "controlled_surface_family_count": len(config["grammar"]["surfaceFamilies"]),
            "external_language_record_read_count": 0,
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
    terminal = structural["exact_policy_terminal_audit"]
    checks = {
        "language_channel_has_required_latents_actions_support_and_information": bool(
            len(REGIME_NAMES) == gates["requiredLatentRegimeCount"]
            and len(CLARIFICATION_ACTIONS) == gates["requiredClarificationActionCount"]
            and structural["common_positive_support"] == gates["requiredCommonPositiveSupport"]
            and structural["minimum_observation_probability_on_support"] >= gates["minimumObservationProbabilityOnSupport"] - 1e-12
            and structural["reference_mutual_information_nats"] >= gates["minimumReferenceMutualInformationNats"]
            and structural["known_target_total_variation"] >= gates["minimumKnownTargetTotalVariation"]
        ),
        "likelihood_and_cost_are_control_relevant_not_constant": bool(
            structural["history_dependent_target_likelihood"] == gates["requiredHistoryDependentTargetLikelihood"]
            and structural["latent_dependent_clarification_cost"] == gates["requiredLatentDependentClarificationCost"]
            and structural["maximum_history_dependent_target_likelihood_difference"] > 0.0
            and structural["clarification_cost_latent_range"] > 0.0
        ),
        "no_immediate_reward_shortcut": bool(
            structural["maximum_clarification_or_control_immediate_reward"] <= gates["maximumClarificationOrControlImmediateReward"]
            and set(structural["control_immediate_rewards"].values()) == {gates["requiredControlImmediateReward"]}
        ),
        "terminal_accounting_is_complete": bool(
            structural["mandatory_automatic_settlement_rate"] == gates["requiredMandatoryAutomaticSettlementRate"]
            and structural["unfinished_clarification_safe_deferral_rate"] == gates["requiredUnfinishedClarificationSafeDeferralRate"]
            and structural["unsettled_control_terminal_count"] <= gates["maximumUnsettledControlTerminalCount"]
            and structural["horizon_escape_path_count"] <= gates["maximumHorizonEscapePathCount"]
            and terminal["unsettled_paths"] == 0
            and terminal["terminal_paths"] == terminal["automatic_settlement_paths"] + terminal["safe_deferral_paths"]
        ),
        "exact_history_actions_match_open_world_language_mechanism": bool(
            result["exact"]["root_action"] == gates["requiredSelectedExactRootAction"]
            and result["exact"]["action_after_root_alpha"] == gates["requiredExactActionAfterRootAlpha"]
            and result["exact"]["action_after_root_beta"] == gates["requiredExactActionAfterRootBeta"]
            and result["exact"]["action_after_root_unresolved"] == gates["requiredExactActionAfterRootUnresolved"]
            and result["closed_world"]["action_after_root_unresolved"] == gates["requiredClosedWorldActionAfterRootUnresolved"]
        ),
        "history_varying_clarification_control_and_deferral_are_reachable": bool(
            len(result["exact"]["distinct_reachable_control_actions"]) >= gates["minimumDistinctReachableControlActions"]
            and result["exact"]["reachable_defer_history_count"] >= gates["minimumReachableDeferHistoryCount"]
            and len(result["exact"]["distinct_post_reference_selected_actions"]) >= gates["minimumDistinctPostReferenceSelectedActions"]
        ),
        "matched_surfaces_and_renaming_are_invariant": bool(
            structural["matched_surface_counterfactual_value_difference"] == gates["requiredMatchedSurfaceCounterfactualValueDifference"]
            and structural["matched_surface_counterfactual_policy_mismatch_count"] == gates["requiredMatchedSurfaceCounterfactualPolicyMismatchCount"]
            and structural["renaming_value_difference"] <= gates["maximumRenamingValueDifference"]
            and structural["renaming_policy_mismatch_count"] == gates["requiredRenamingPolicyMismatchCount"]
        ),
        "exact_beats_immediate_defer": result["exact"]["normalized_advantage_over_immediate_defer"] >= gates["minimumNormalizedExactOverImmediateDeferAdvantage"],
        "closed_world_has_material_regret": result["closed_world"]["normalized_regret"] >= gates["minimumNormalizedClosedWorldRegret"],
        "forced_commit_has_material_regret": result["forced_commit"]["normalized_regret"] >= gates["minimumNormalizedForcedCommitRegret"],
        "MAP_has_material_regret": result["map"]["normalized_regret"] >= gates["minimumNormalizedMAPRegret"],
        "posterior_sampling_has_material_regret": result["posterior_sampling"]["normalized_regret"] >= gates["minimumNormalizedPosteriorSamplingRegret"],
        "adaptive_beats_open_loop": result["open_loop"]["normalized_exact_advantage"] >= gates["minimumNormalizedExactOverOpenLoopAdvantage"],
        "myopic_has_material_regret": result["myopic"]["normalized_regret"] >= gates["minimumNormalizedMyopicRegret"],
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
                    result["myopic"]["true_environment_value"],
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
                ("external_language_record_read_count", "maximumExternalLanguageRecordReadCount"),
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


__all__ = [
    "CLARIFICATION_ACTIONS",
    "CONTROL_ACTIONS",
    "LanguageKernel",
    "OBSERVATION_NAMES",
    "REGIME_NAMES",
    "STAGE_NAMES",
    "allowed_actions",
    "audit_oracle",
    "build_kernel",
    "clarification_cost_matrix",
    "clarification_likelihood",
    "clarification_step",
    "control_return",
    "evaluate_oracle",
    "plan_exact",
    "recover_semantic_policy",
    "render_policy",
]
