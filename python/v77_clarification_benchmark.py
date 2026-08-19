#!/usr/bin/env python3
"""Exact finite clarification-and-control benchmark for V77.

The hidden interpretation is static.  The observable state records only whether
the synthetic episode remains active.  This module contains no model inference,
API access, external tools, or human-language collection.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import numpy as np


HYPOTHESIS_NAMES = (
    "q2_report__jordan_lee",
    "annual_report__jordan_lee",
    "q2_report__jordan_patel",
    "annual_report__jordan_patel",
    "none_of_the_above",
)
ACTION_NAMES = (
    "ask_report",
    "ask_recipient",
    "ask_full_details",
    "draft_q2_lee",
    "draft_annual_lee",
    "draft_q2_patel",
    "draft_annual_patel",
    "send_q2_lee",
    "send_annual_lee",
    "send_q2_patel",
    "send_annual_patel",
    "safe_draft",
    "abstain",
)
OBSERVATION_NAMES = (
    "report_q2",
    "report_annual",
    "report_other",
    "recipient_lee",
    "recipient_patel",
    "recipient_other",
    "full_q2_lee",
    "full_annual_lee",
    "full_q2_patel",
    "full_annual_patel",
    "full_other",
    "draft_approved",
    "draft_rejected",
    "done",
)
STATE_NAMES = ("active", "terminal")

INFORMATION_ACTIONS = (0, 1, 2)
DRAFT_ACTIONS = (3, 4, 5, 6)
SEND_ACTIONS = (7, 8, 9, 10)
SAFE_DRAFT_ACTION = 11
ABSTAIN_ACTION = 12
TERMINAL_ACTIONS = SEND_ACTIONS + (SAFE_DRAFT_ACTION, ABSTAIN_ACTION)
NONE_HYPOTHESIS = 4


@dataclass(frozen=True)
class ClarificationKernel:
    hypothesis_names: tuple[str, ...]
    action_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    state_names: tuple[str, ...]
    transition: np.ndarray  # action, state, successor
    observation: np.ndarray  # hypothesis, action, successor, observation
    reward: np.ndarray  # hypothesis, action, state, successor
    discount: float
    send_minimum_matching_posterior: float
    send_maximum_none_posterior: float
    send_action_to_hypothesis: tuple[tuple[int, int], ...]
    none_hypothesis: int
    always_certified_actions: tuple[int, ...]

    def __post_init__(self) -> None:
        hypotheses = len(self.hypothesis_names)
        actions = len(self.action_names)
        states = len(self.state_names)
        observations = len(self.observation_names)
        transition = np.asarray(self.transition, dtype=np.float64)
        observation = np.asarray(self.observation, dtype=np.float64)
        reward = np.asarray(self.reward, dtype=np.float64)
        if transition.shape != (actions, states, states):
            raise ValueError("V77 transition shape mismatch")
        if observation.shape != (hypotheses, actions, states, observations):
            raise ValueError("V77 observation shape mismatch")
        if reward.shape != (hypotheses, actions, states, states):
            raise ValueError("V77 reward shape mismatch")
        if not all(np.isfinite(value).all() for value in (transition, observation, reward)):
            raise ValueError("V77 kernel contains a non-finite value")
        if np.any(transition < 0.0) or np.any(observation < 0.0):
            raise ValueError("V77 kernel probabilities must be nonnegative")
        if not np.allclose(transition.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V77 transitions are not normalized")
        if not np.allclose(observation.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V77 observations are not normalized")
        supports = observation > 0.0
        if not all(np.array_equal(supports[0], supports[index]) for index in range(1, hypotheses)):
            raise ValueError("V77 hypothesis observation supports differ")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("V77 discount must lie in (0,1]")
        if not 0.5 < self.send_minimum_matching_posterior <= 1.0:
            raise ValueError("V77 send matching-posterior threshold is invalid")
        if not 0.0 <= self.send_maximum_none_posterior < 0.5:
            raise ValueError("V77 send none-posterior threshold is invalid")
        if self.none_hypothesis not in range(hypotheses):
            raise ValueError("V77 none hypothesis index is invalid")
        send_actions = tuple(action for action, _ in self.send_action_to_hypothesis)
        send_hypotheses = tuple(hypothesis for _, hypothesis in self.send_action_to_hypothesis)
        if (
            len(set(send_actions)) != len(send_actions)
            or any(action not in range(actions) for action in send_actions)
            or any(hypothesis not in range(hypotheses) for hypothesis in send_hypotheses)
            or any(action not in range(actions) for action in self.always_certified_actions)
            or set(send_actions) & set(self.always_certified_actions)
        ):
            raise ValueError("V77 action certification map is invalid")
        for value in (transition, observation, reward):
            value.setflags(write=False)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "reward", reward)


@dataclass(frozen=True)
class ClarificationFixture:
    name: str
    synthetic_instruction: str
    reward_profile: str
    kernel: ClarificationKernel
    initial_belief: np.ndarray

    def __post_init__(self) -> None:
        belief = validate_belief(self.kernel, self.initial_belief)
        belief.setflags(write=False)
        object.__setattr__(self, "initial_belief", belief)


def validate_belief(kernel: ClarificationKernel, belief: np.ndarray) -> np.ndarray:
    value = np.asarray(belief, dtype=np.float64)
    expected = (len(kernel.hypothesis_names), len(kernel.state_names))
    if value.shape != expected:
        raise ValueError(f"V77 belief shape {value.shape} != {expected}")
    if np.any(value < 0.0) or not np.isfinite(value).all():
        raise ValueError("V77 belief contains an invalid value")
    if not np.isclose(value.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("V77 belief is not normalized")
    return value.copy()


def hypothesis_masses(kernel: ClarificationKernel, belief: np.ndarray) -> np.ndarray:
    return validate_belief(kernel, belief).sum(axis=1)


def is_terminal_belief(kernel: ClarificationKernel, belief: np.ndarray) -> bool:
    value = validate_belief(kernel, belief)
    return bool(value[:, STATE_NAMES.index("terminal")].sum() >= 1.0 - 1e-12)


def certified_actions(kernel: ClarificationKernel, belief: np.ndarray) -> tuple[int, ...]:
    masses = hypothesis_masses(kernel, belief)
    allowed = list(kernel.always_certified_actions)
    for action, hypothesis in kernel.send_action_to_hypothesis:
        if (
            masses[hypothesis] + 1e-12 >= kernel.send_minimum_matching_posterior
            and masses[kernel.none_hypothesis]
            <= kernel.send_maximum_none_posterior + 1e-12
        ):
            allowed.append(action)
    return tuple(sorted(allowed))


def _transition() -> np.ndarray:
    actions = len(ACTION_NAMES)
    transition = np.zeros((actions, len(STATE_NAMES), len(STATE_NAMES)), dtype=np.float64)
    active = STATE_NAMES.index("active")
    terminal = STATE_NAMES.index("terminal")
    for action in range(actions):
        transition[action, terminal, terminal] = 1.0
        successor = terminal if action in TERMINAL_ACTIONS else active
        transition[action, active, successor] = 1.0
    return transition


def _symmetric_channel(correct: int, indices: Sequence[int], reliability: float) -> np.ndarray:
    if correct not in indices or len(indices) < 2:
        raise ValueError("V77 symmetric channel specification is invalid")
    probabilities = np.zeros(len(OBSERVATION_NAMES), dtype=np.float64)
    remainder = (1.0 - reliability) / (len(indices) - 1)
    for index in indices:
        probabilities[index] = reliability if index == correct else remainder
    return probabilities


def _observation(config: dict[str, Any]) -> np.ndarray:
    hypotheses = len(HYPOTHESIS_NAMES)
    actions = len(ACTION_NAMES)
    states = len(STATE_NAMES)
    observations = len(OBSERVATION_NAMES)
    value = np.zeros((hypotheses, actions, states, observations), dtype=np.float64)
    active = STATE_NAMES.index("active")
    terminal = STATE_NAMES.index("terminal")
    done = OBSERVATION_NAMES.index("done")
    focused = float(config["sharedParameters"]["focusedQuestionReliability"])
    full = float(config["sharedParameters"]["fullQuestionReliability"])
    approve_match = float(
        config["sharedParameters"]["matchingDraftApprovalProbability"]
    )
    approve_other = float(
        config["sharedParameters"]["nonmatchingDraftApprovalProbability"]
    )

    report_indices = tuple(OBSERVATION_NAMES.index(name) for name in (
        "report_q2", "report_annual", "report_other"
    ))
    recipient_indices = tuple(OBSERVATION_NAMES.index(name) for name in (
        "recipient_lee", "recipient_patel", "recipient_other"
    ))
    full_indices = tuple(range(
        OBSERVATION_NAMES.index("full_q2_lee"),
        OBSERVATION_NAMES.index("full_other") + 1,
    ))
    approved = OBSERVATION_NAMES.index("draft_approved")
    rejected = OBSERVATION_NAMES.index("draft_rejected")

    # Define a normalized observation distribution even for successor rows
    # that are unreachable under the transition kernel. Reachable active rows
    # for information and draft actions are replaced below.
    value[:, :, :, done] = 1.0

    for hypothesis in range(hypotheses):
        report_correct = report_indices[2]
        recipient_correct = recipient_indices[2]
        if hypothesis < NONE_HYPOTHESIS:
            report_correct = report_indices[0] if hypothesis in (0, 2) else report_indices[1]
            recipient_correct = recipient_indices[0] if hypothesis in (0, 1) else recipient_indices[1]
        value[hypothesis, 0, active] = _symmetric_channel(
            report_correct, report_indices, focused
        )
        value[hypothesis, 1, active] = _symmetric_channel(
            recipient_correct, recipient_indices, focused
        )
        value[hypothesis, 2, active] = _symmetric_channel(
            full_indices[hypothesis], full_indices, full
        )
        for action, candidate in zip(DRAFT_ACTIONS, range(4), strict=True):
            value[hypothesis, action, active] = 0.0
            probability = approve_match if hypothesis == candidate else approve_other
            value[hypothesis, action, active, approved] = probability
            value[hypothesis, action, active, rejected] = 1.0 - probability
    return value


def _reward(config: dict[str, Any], profile: str) -> np.ndarray:
    value = np.zeros(
        (len(HYPOTHESIS_NAMES), len(ACTION_NAMES), len(STATE_NAMES), len(STATE_NAMES)),
        dtype=np.float64,
    )
    active = STATE_NAMES.index("active")
    terminal = STATE_NAMES.index("terminal")
    if profile == "positive":
        rewards = config["positiveRewardProfile"]
    elif profile == "dominant_control":
        rewards = config["dominantControlRewardProfile"]
    else:
        raise ValueError("V77 reward profile is invalid")

    action_costs = (
        float(rewards["askReport"]),
        float(rewards["askRecipient"]),
        float(rewards["askFullDetails"]),
    )
    for hypothesis in range(len(HYPOTHESIS_NAMES)):
        for action, cost in zip(INFORMATION_ACTIONS, action_costs, strict=True):
            value[hypothesis, action, active, active] = cost
        for action in DRAFT_ACTIONS:
            value[hypothesis, action, active, active] = float(rewards["candidateDraft"])
        for action, candidate in zip(SEND_ACTIONS, range(4), strict=True):
            if profile == "positive":
                send_reward = (
                    rewards["correctSend"]
                    if hypothesis == candidate
                    else rewards["wrongOrUnsupportedSend"]
                )
            else:
                send_reward = rewards["everySend"]
            value[hypothesis, action, active, terminal] = float(send_reward)
        value[hypothesis, SAFE_DRAFT_ACTION, active, terminal] = float(rewards["safeDraft"])
        value[hypothesis, ABSTAIN_ACTION, active, terminal] = float(rewards["abstain"])
    return value


def build_fixture(config: dict[str, Any], name: str) -> ClarificationFixture:
    fixture_rows = {row["name"]: row for row in config["fixtures"]}
    if name not in fixture_rows:
        raise KeyError(f"unknown V77 fixture: {name}")
    row = fixture_rows[name]
    shared = config["sharedParameters"]
    kernel = ClarificationKernel(
        hypothesis_names=HYPOTHESIS_NAMES,
        action_names=ACTION_NAMES,
        observation_names=OBSERVATION_NAMES,
        state_names=STATE_NAMES,
        transition=_transition(),
        observation=_observation(config),
        reward=_reward(config, row["rewardProfile"]),
        discount=float(shared["discount"]),
        send_minimum_matching_posterior=float(
            shared["irreversibleSendMinimumMatchingPosterior"]
        ),
        send_maximum_none_posterior=float(
            shared["irreversibleSendMaximumNonePosterior"]
        ),
        send_action_to_hypothesis=tuple(zip(SEND_ACTIONS, range(4), strict=True)),
        none_hypothesis=NONE_HYPOTHESIS,
        always_certified_actions=(
            INFORMATION_ACTIONS + DRAFT_ACTIONS + (SAFE_DRAFT_ACTION, ABSTAIN_ACTION)
        ),
    )
    belief = np.zeros((len(HYPOTHESIS_NAMES), len(STATE_NAMES)), dtype=np.float64)
    belief[:, STATE_NAMES.index("active")] = np.asarray(row["prior"], dtype=np.float64)
    return ClarificationFixture(
        name=row["name"],
        synthetic_instruction=row["syntheticInstruction"],
        reward_profile=row["rewardProfile"],
        kernel=kernel,
        initial_belief=belief,
    )


def exact_step(
    kernel: ClarificationKernel, belief: np.ndarray, action: int
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    if action not in range(len(kernel.action_names)):
        raise ValueError("V77 action index is invalid")
    predicted = np.einsum("hs,sq->hq", value, kernel.transition[action], optimize=True)
    joint = predicted[:, :, None] * kernel.observation[:, action]
    probabilities = joint.sum(axis=(0, 1))
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("V77 predictive observations do not normalize")
    posteriors = {
        observation: joint[:, :, observation] / float(probability)
        for observation, probability in enumerate(probabilities)
        if probability > 0.0
    }
    immediate = float(
        np.einsum(
            "hs,sq,hsq->",
            value,
            kernel.transition[action],
            kernel.reward[:, action],
            optimize=True,
        )
    )
    return {
        "reward": immediate,
        "probabilities": probabilities,
        "posteriors": posteriors,
    }


def _select(rows: Sequence[tuple[int, float]], tolerance: float) -> tuple[int, tuple[int, ...], float]:
    maximum = max(float(value) for _, value in rows)
    optimal = tuple(action for action, value in rows if maximum - float(value) <= tolerance)
    return optimal[0], optimal, maximum


def plan_exact(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    if horizon < 0:
        raise ValueError("V77 horizon cannot be negative")
    if stats is not None:
        stats["belief_checks"] = stats.get("belief_checks", 0) + 1
        stats["normalized_belief_checks"] = (
            stats.get("normalized_belief_checks", 0) + 1
        )
        stats["bellman_nodes"] = stats.get("bellman_nodes", 0) + 1
    if horizon == 0 or is_terminal_belief(kernel, value):
        return {"terminal": True, "horizon": horizon, "value": 0.0}
    allowed = certified_actions(kernel, value)
    candidates: list[dict[str, Any]] = []
    for action in allowed:
        step = exact_step(kernel, value, action)
        branches: dict[int, dict[str, Any]] = {}
        continuation = 0.0
        if horizon > 1:
            for observation, posterior in step["posteriors"].items():
                child = plan_exact(
                    kernel,
                    posterior,
                    horizon - 1,
                    tie_tolerance=tie_tolerance,
                    stats=stats,
                )
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
    rows = [(int(row["action"]), float(row["value"])) for row in candidates]
    selected, optimal, maximum = _select(rows, tie_tolerance)
    selected_row = next(row for row in candidates if row["action"] == selected)
    masses = value.sum(axis=1)
    return {
        "terminal": False,
        "horizon": horizon,
        "value": float(maximum),
        "selected_action": selected,
        "optimal_actions": optimal,
        "q_values": {int(row["action"]): float(row["value"]) for row in candidates},
        "certified_actions": allowed,
        "hypothesis_masses": masses.tolist(),
        "branches": selected_row["branches"],
    }


def evaluate_policy_exact(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    policy: dict[str, Any],
    horizon: int,
    *,
    certificate_violations: list[dict[str, Any]] | None = None,
) -> float:
    value = validate_belief(kernel, belief)
    if horizon == 0 or is_terminal_belief(kernel, value):
        return 0.0
    if policy.get("terminal") or int(policy.get("horizon", -1)) != horizon:
        raise ValueError("V77 policy horizon or terminal marker is invalid")
    action = int(policy["selected_action"])
    if certificate_violations is not None and action not in certified_actions(kernel, value):
        certificate_violations.append(
            {
                "horizon": horizon,
                "action": action,
                "hypothesis_masses": value.sum(axis=1).tolist(),
            }
        )
    step = exact_step(kernel, value, action)
    continuation = 0.0
    if horizon > 1:
        for observation, posterior in step["posteriors"].items():
            if observation not in policy.get("branches", {}):
                raise RuntimeError("V77 policy omits a reachable observation branch")
            continuation += float(step["probabilities"][observation]) * evaluate_policy_exact(
                kernel,
                posterior,
                policy["branches"][observation],
                horizon - 1,
                certificate_violations=certificate_violations,
            )
    return float(step["reward"] + kernel.discount * continuation)


def point_policy(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    hypothesis: int,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    masses = value.sum(axis=1)
    if hypothesis not in range(len(kernel.hypothesis_names)) or masses[hypothesis] <= 0.0:
        raise ValueError("V77 point hypothesis has zero or invalid mass")
    point = np.zeros_like(value)
    point[hypothesis] = value[hypothesis] / float(masses[hypothesis])
    return plan_exact(kernel, point, horizon, tie_tolerance=tie_tolerance)


def map_control(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    masses = value.sum(axis=1)
    hypothesis = int(np.argmax(masses))
    policy = point_policy(
        kernel, value, hypothesis, horizon, tie_tolerance=tie_tolerance
    )
    violations: list[dict[str, Any]] = []
    exact_value = evaluate_policy_exact(
        kernel, value, policy, horizon, certificate_violations=violations
    )
    return {
        "hypothesis": hypothesis,
        "hypothesis_name": kernel.hypothesis_names[hypothesis],
        "hypothesis_mass": float(masses[hypothesis]),
        "policy": policy,
        "value": exact_value,
        "complete_belief_certificate_violations": violations,
        "shadow_only": True,
        "off_support_fallback_count": 0,
    }


def posterior_sampling_control(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    masses = value.sum(axis=1)
    expected = 0.0
    root_distribution = np.zeros(len(kernel.action_names), dtype=np.float64)
    rows = []
    violations: list[dict[str, Any]] = []
    for hypothesis, mass in enumerate(masses):
        if mass <= 0.0:
            continue
        policy = point_policy(
            kernel, value, hypothesis, horizon, tie_tolerance=tie_tolerance
        )
        local_violations: list[dict[str, Any]] = []
        exact_value = evaluate_policy_exact(
            kernel,
            value,
            policy,
            horizon,
            certificate_violations=local_violations,
        )
        expected += float(mass) * exact_value
        root_distribution[int(policy["selected_action"])] += float(mass)
        violations.extend(local_violations)
        rows.append(
            {
                "hypothesis": hypothesis,
                "hypothesis_name": kernel.hypothesis_names[hypothesis],
                "mass": float(mass),
                "root_action": int(policy["selected_action"]),
                "exact_environment_value": exact_value,
                "complete_belief_certificate_violation_count": len(local_violations),
            }
        )
    return {
        "value": float(expected),
        "root_action_distribution": root_distribution.tolist(),
        "hypotheses": rows,
        "complete_belief_certificate_violations": violations,
        "sampled_hypothesis_persists_for_full_policy": True,
        "shadow_only": True,
        "off_support_fallback_count": 0,
    }


def _terminal_control_policy(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    enforce_complete_belief_certificate: bool,
    tie_tolerance: float,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    actions = TERMINAL_ACTIONS
    if enforce_complete_belief_certificate:
        certified = set(certified_actions(kernel, value))
        actions = tuple(action for action in actions if action in certified)
    rows = [(action, float(exact_step(kernel, value, action)["reward"])) for action in actions]
    selected, optimal, maximum = _select(rows, tie_tolerance)
    return {
        "terminal": False,
        "horizon": horizon,
        "value": maximum,
        "selected_action": selected,
        "optimal_actions": optimal,
        "q_values": dict(rows),
        "branches": {},
    }


def act_immediately_policy(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    return _terminal_control_policy(
        kernel,
        belief,
        horizon,
        enforce_complete_belief_certificate=False,
        tie_tolerance=tie_tolerance,
    )


def ask_always_policy(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    if horizon <= 1:
        return _terminal_control_policy(
            kernel,
            value,
            max(1, horizon),
            enforce_complete_belief_certificate=True,
            tie_tolerance=tie_tolerance,
        )
    action = ACTION_NAMES.index("ask_full_details")
    step = exact_step(kernel, value, action)
    branches = {
        observation: ask_always_policy(
            kernel,
            posterior,
            horizon - 1,
            tie_tolerance=tie_tolerance,
        )
        for observation, posterior in step["posteriors"].items()
    }
    return {
        "terminal": False,
        "horizon": horizon,
        "value": 0.0,
        "selected_action": action,
        "optimal_actions": (action,),
        "q_values": {action: float(step["reward"])},
        "branches": branches,
    }


def evaluate_action_sequence(
    kernel: ClarificationKernel, belief: np.ndarray, actions: Sequence[int]
) -> float:
    value = validate_belief(kernel, belief)
    if not actions or is_terminal_belief(kernel, value):
        return 0.0
    step = exact_step(kernel, value, int(actions[0]))
    continuation = 0.0
    if len(actions) > 1:
        continuation = sum(
            float(step["probabilities"][observation])
            * evaluate_action_sequence(kernel, posterior, actions[1:])
            for observation, posterior in step["posteriors"].items()
        )
    return float(step["reward"] + kernel.discount * continuation)


def best_open_loop_sequence(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    rows = [
        (actions, evaluate_action_sequence(kernel, belief, actions))
        for actions in product(range(len(kernel.action_names)), repeat=horizon)
    ]
    maximum = max(value for _, value in rows)
    optimal = tuple(actions for actions, value in rows if maximum - value <= tie_tolerance)
    return {
        "value": float(maximum),
        "selected_actions": optimal[0],
        "optimal_sequences": optimal,
        "sequence_count": len(rows),
    }


def oracle_interpretation_value(
    kernel: ClarificationKernel,
    belief: np.ndarray,
    horizon: int,
    *,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    value = validate_belief(kernel, belief)
    masses = value.sum(axis=1)
    expected = 0.0
    rows = []
    for hypothesis, mass in enumerate(masses):
        if mass <= 0.0:
            continue
        point = np.zeros_like(value)
        point[hypothesis] = value[hypothesis] / float(mass)
        policy = plan_exact(kernel, point, horizon, tie_tolerance=tie_tolerance)
        conditional_value = evaluate_policy_exact(kernel, point, policy, horizon)
        expected += float(mass) * conditional_value
        rows.append(
            {
                "hypothesis": hypothesis,
                "hypothesis_name": kernel.hypothesis_names[hypothesis],
                "mass": float(mass),
                "root_action": int(policy["selected_action"]),
                "conditional_value": conditional_value,
            }
        )
    return {"value": float(expected), "hypotheses": rows}


def finite_horizon_return_scale(kernel: ClarificationKernel, horizon: int) -> float:
    reward_span = float(kernel.reward.max() - kernel.reward.min())
    discount_sum = sum(kernel.discount**depth for depth in range(horizon))
    return float(max(1.0, reward_span * discount_sum))


def structural_diagnostics(fixture: ClarificationFixture) -> dict[str, Any]:
    kernel = fixture.kernel
    supports = kernel.observation > 0.0
    identical = all(
        np.array_equal(supports[0], supports[hypothesis])
        for hypothesis in range(1, len(kernel.hypothesis_names))
    )
    observation_rows = kernel.observation.reshape(-1, len(kernel.observation_names))
    belief = validate_belief(kernel, fixture.initial_belief)
    return {
        "hypothesis_count": len(kernel.hypothesis_names),
        "action_count": len(kernel.action_names),
        "observation_count": len(kernel.observation_names),
        "state_count": len(kernel.state_names),
        "dense_kernel_bytes": int(
            kernel.transition.nbytes + kernel.observation.nbytes + kernel.reward.nbytes
        ),
        "observation_normalization_rate": float(
            np.mean(np.isclose(observation_rows.sum(axis=1), 1.0, atol=1e-12, rtol=0.0))
        ),
        "identical_hypothesis_support_rate": 1.0 if identical else 0.0,
        "belief_normalizes": bool(np.isclose(belief.sum(), 1.0, atol=1e-12, rtol=0.0)),
        "none_hypothesis_prior": float(belief[NONE_HYPOTHESIS].sum()),
        "initial_certified_actions": [
            kernel.action_names[action] for action in certified_actions(kernel, belief)
        ],
        "off_support_fallback_count": 0,
        "model_forward_pass_count": 0,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
        "external_side_effect_count": 0,
    }
