#!/usr/bin/env python3
"""Fresh V78 scheduling/sending clarification benchmark construction."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from v77_clarification_benchmark import (
    ClarificationFixture,
    ClarificationKernel,
    structural_diagnostics as base_structural_diagnostics,
)


HYPOTHESIS_NAMES = (
    "schedule_review__alex_chen",
    "send_summary__alex_chen",
    "schedule_review__alex_kim",
    "send_summary__alex_kim",
    "none_of_the_above",
)
ACTION_NAMES = (
    "ask_operation",
    "ask_recipient",
    "ask_full_details",
    "preview_schedule_chen",
    "preview_send_chen",
    "preview_schedule_kim",
    "preview_send_kim",
    "execute_schedule_chen",
    "execute_send_chen",
    "execute_schedule_kim",
    "execute_send_kim",
    "safe_preview",
    "abstain",
)
OBSERVATION_NAMES = (
    "operation_schedule",
    "operation_send",
    "operation_other",
    "recipient_chen",
    "recipient_kim",
    "recipient_other",
    "full_schedule_chen",
    "full_send_chen",
    "full_schedule_kim",
    "full_send_kim",
    "full_other",
    "preview_approved",
    "preview_rejected",
    "done",
)
STATE_NAMES = ("active", "terminal")

INFORMATION_ACTIONS = (0, 1, 2)
PREVIEW_ACTIONS = (3, 4, 5, 6)
EXECUTION_ACTIONS = (7, 8, 9, 10)
SAFE_PREVIEW_ACTION = 11
ABSTAIN_ACTION = 12
TERMINAL_ACTIONS = EXECUTION_ACTIONS + (SAFE_PREVIEW_ACTION, ABSTAIN_ACTION)
NONE_HYPOTHESIS = 4


def _transition() -> np.ndarray:
    value = np.zeros(
        (len(ACTION_NAMES), len(STATE_NAMES), len(STATE_NAMES)), dtype=np.float64
    )
    active = STATE_NAMES.index("active")
    terminal = STATE_NAMES.index("terminal")
    for action in range(len(ACTION_NAMES)):
        value[action, terminal, terminal] = 1.0
        value[action, active, terminal if action in TERMINAL_ACTIONS else active] = 1.0
    return value


def _symmetric_channel(
    correct: int, indices: Sequence[int], reliability: float
) -> np.ndarray:
    if correct not in indices or len(indices) < 2:
        raise ValueError("V78 symmetric channel specification is invalid")
    value = np.zeros(len(OBSERVATION_NAMES), dtype=np.float64)
    remainder = (1.0 - reliability) / (len(indices) - 1)
    for index in indices:
        value[index] = reliability if index == correct else remainder
    return value


def _observation(config: dict[str, Any]) -> np.ndarray:
    value = np.zeros(
        (
            len(HYPOTHESIS_NAMES),
            len(ACTION_NAMES),
            len(STATE_NAMES),
            len(OBSERVATION_NAMES),
        ),
        dtype=np.float64,
    )
    active = STATE_NAMES.index("active")
    done = OBSERVATION_NAMES.index("done")
    shared = config["sharedParameters"]
    focused = float(shared["focusedQuestionReliability"])
    full = float(shared["fullQuestionReliability"])
    approve_match = float(shared["matchingPreviewApprovalProbability"])
    approve_other = float(shared["nonmatchingPreviewApprovalProbability"])
    operation_indices = tuple(
        OBSERVATION_NAMES.index(name)
        for name in ("operation_schedule", "operation_send", "operation_other")
    )
    recipient_indices = tuple(
        OBSERVATION_NAMES.index(name)
        for name in ("recipient_chen", "recipient_kim", "recipient_other")
    )
    full_indices = tuple(
        range(
            OBSERVATION_NAMES.index("full_schedule_chen"),
            OBSERVATION_NAMES.index("full_other") + 1,
        )
    )
    approved = OBSERVATION_NAMES.index("preview_approved")
    rejected = OBSERVATION_NAMES.index("preview_rejected")

    # Normalize every row, including transition-unreachable successor rows.
    value[:, :, :, done] = 1.0
    for hypothesis in range(len(HYPOTHESIS_NAMES)):
        operation_correct = operation_indices[2]
        recipient_correct = recipient_indices[2]
        if hypothesis < NONE_HYPOTHESIS:
            operation_correct = (
                operation_indices[0] if hypothesis in (0, 2) else operation_indices[1]
            )
            recipient_correct = (
                recipient_indices[0] if hypothesis in (0, 1) else recipient_indices[1]
            )
        value[hypothesis, 0, active] = _symmetric_channel(
            operation_correct, operation_indices, focused
        )
        value[hypothesis, 1, active] = _symmetric_channel(
            recipient_correct, recipient_indices, focused
        )
        value[hypothesis, 2, active] = _symmetric_channel(
            full_indices[hypothesis], full_indices, full
        )
        for action, candidate in zip(PREVIEW_ACTIONS, range(4), strict=True):
            value[hypothesis, action, active] = 0.0
            probability = approve_match if hypothesis == candidate else approve_other
            value[hypothesis, action, active, approved] = probability
            value[hypothesis, action, active, rejected] = 1.0 - probability
    return value


def _reward(config: dict[str, Any], profile: str) -> np.ndarray:
    value = np.zeros(
        (
            len(HYPOTHESIS_NAMES),
            len(ACTION_NAMES),
            len(STATE_NAMES),
            len(STATE_NAMES),
        ),
        dtype=np.float64,
    )
    active = STATE_NAMES.index("active")
    terminal = STATE_NAMES.index("terminal")
    if profile == "positive":
        rewards = config["positiveRewardProfile"]
    elif profile == "dominant_control":
        rewards = config["dominantControlRewardProfile"]
    else:
        raise ValueError("V78 reward profile is invalid")
    costs = (
        float(rewards["askOperation"]),
        float(rewards["askRecipient"]),
        float(rewards["askFullDetails"]),
    )
    for hypothesis in range(len(HYPOTHESIS_NAMES)):
        for action, cost in zip(INFORMATION_ACTIONS, costs, strict=True):
            value[hypothesis, action, active, active] = cost
        for action in PREVIEW_ACTIONS:
            value[hypothesis, action, active, active] = float(
                rewards["candidatePreview"]
            )
        for action, candidate in zip(EXECUTION_ACTIONS, range(4), strict=True):
            if profile == "positive":
                reward = (
                    rewards["correctExecution"]
                    if hypothesis == candidate
                    else rewards["wrongOrUnsupportedExecution"]
                )
            else:
                reward = rewards["everyExecution"]
            value[hypothesis, action, active, terminal] = float(reward)
        value[hypothesis, SAFE_PREVIEW_ACTION, active, terminal] = float(
            rewards["safePreview"]
        )
        value[hypothesis, ABSTAIN_ACTION, active, terminal] = float(
            rewards["abstain"]
        )
    return value


def build_fixture(config: dict[str, Any], name: str) -> ClarificationFixture:
    rows = {row["name"]: row for row in config["fixtures"]}
    if name not in rows:
        raise KeyError(f"unknown V78 fixture: {name}")
    row = rows[name]
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
            shared["irreversibleExecutionMinimumMatchingPosterior"]
        ),
        send_maximum_none_posterior=float(
            shared["irreversibleExecutionMaximumNonePosterior"]
        ),
        send_action_to_hypothesis=tuple(
            zip(EXECUTION_ACTIONS, range(4), strict=True)
        ),
        none_hypothesis=NONE_HYPOTHESIS,
        always_certified_actions=(
            INFORMATION_ACTIONS
            + PREVIEW_ACTIONS
            + (SAFE_PREVIEW_ACTION, ABSTAIN_ACTION)
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


def structural_diagnostics(fixture: ClarificationFixture) -> dict[str, Any]:
    row = base_structural_diagnostics(fixture)
    transition_rows = fixture.kernel.transition.reshape(-1, len(STATE_NAMES))
    row["transition_normalization_rate"] = float(
        np.mean(
            np.isclose(
                transition_rows.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
            )
        )
    )
    row["real_tool_call_count"] = 0
    return row
