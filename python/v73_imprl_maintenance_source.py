#!/usr/bin/env python3
"""Source-grounded V73 maintenance adapter with a non-harvestable beacon.

The target transition, inspection, replacement, costs, initial belief, and
discount are frozen from component 4 (index 3) of IMPRL's
``hard-4-of-4_infinite.yaml`` at commit
3c9cde75b48a2cba54f62330ead1e1dbc054d0cf.  The calibration beacon and latent
label-codebook layer are project-authored and development-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from v71_exact_planning import SensorCodebookKernel


SOURCE_COMMIT = "3c9cde75b48a2cba54f62330ead1e1dbc054d0cf"
STATE_NAMES = ("healthy", "degraded", "failed")
ACTION_NAMES = (
    "do_nothing",
    "replace_target",
    "inspect_target",
    "calibrate_beacon",
)
OBSERVATION_NAMES = ("label_0", "label_1", "label_2", "none")
LATENT_NAMES = ("canonical", "swap_labels_0_and_1")

SOURCE_DETERIORATION = np.array(
    (
        (0.72, 0.28, 0.0),
        (0.0, 0.78, 0.22),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
SOURCE_INITIAL_BELIEF = np.array((0.6, 0.4, 0.0), dtype=np.float64)
SOURCE_INSPECTION_ACCURACY = 0.9
SOURCE_FAILURE_OBSERVATION_ACCURACY = 1.0
SOURCE_REPLACEMENT_ACCURACY = 1.0
SOURCE_REPLACEMENT_REWARD = -90.0
SOURCE_INSPECTION_REWARD = -4.0
SOURCE_MOBILISATION_REWARD = -4.0
SOURCE_FAILURE_PENALTY_FACTOR = 3.0
PROJECTED_FAILURE_REWARD = (
    SOURCE_REPLACEMENT_REWARD * SOURCE_FAILURE_PENALTY_FACTOR
)
SOURCE_DISCOUNT = 0.8


@dataclass(frozen=True)
class V73MaintenanceFamily:
    kernel: SensorCodebookKernel
    initial_belief: np.ndarray
    source_commit: str = SOURCE_COMMIT

    def __post_init__(self) -> None:
        belief = np.asarray(self.initial_belief, dtype=np.float64)
        if belief.shape != (2, len(STATE_NAMES)):
            raise ValueError("V73 initial joint belief shape mismatch")
        if np.any(belief < 0.0) or not np.isclose(
            belief.sum(), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ValueError("V73 initial joint belief is invalid")
        belief.setflags(write=False)
        object.__setattr__(self, "initial_belief", belief)


def source_inspection_model() -> np.ndarray:
    """Reproduce IMPRL's three-state inspection construction for component 4."""
    p = SOURCE_INSPECTION_ACCURACY
    failure_p = SOURCE_FAILURE_OBSERVATION_ACCURACY
    return np.array(
        (
            (p, 1.0 - p, 0.0),
            ((1.0 - p) / 2.0, p, (1.0 - p) / 2.0),
            (0.0, 1.0 - failure_p, failure_p),
        ),
        dtype=np.float64,
    )


def source_replacement_transition() -> np.ndarray:
    """Reproduce IMPRL's replacement @ deterioration transition."""
    replacement = np.zeros_like(SOURCE_DETERIORATION)
    replacement[0, 0] = 1.0
    for state in range(1, len(STATE_NAMES)):
        replacement[state, 0] = SOURCE_REPLACEMENT_ACCURACY
        replacement[state, state] = 1.0 - SOURCE_REPLACEMENT_ACCURACY
    return replacement @ SOURCE_DETERIORATION


def _transition() -> np.ndarray:
    transition = np.zeros((len(ACTION_NAMES), len(STATE_NAMES), len(STATE_NAMES)))
    transition[ACTION_NAMES.index("do_nothing")] = SOURCE_DETERIORATION
    transition[ACTION_NAMES.index("replace_target")] = (
        source_replacement_transition()
    )
    transition[ACTION_NAMES.index("inspect_target")] = SOURCE_DETERIORATION
    transition[ACTION_NAMES.index("calibrate_beacon")] = SOURCE_DETERIORATION
    return transition


def _swap_labels_0_and_1(row: np.ndarray) -> np.ndarray:
    value = np.asarray(row, dtype=np.float64)
    return value[np.array((1, 0, 2))]


def _observation() -> np.ndarray:
    observations = np.zeros(
        (2, len(ACTION_NAMES), len(STATE_NAMES), len(OBSERVATION_NAMES)),
        dtype=np.float64,
    )
    none = OBSERVATION_NAMES.index("none")
    for latent in range(2):
        for action_name in ("do_nothing", "replace_target"):
            observations[latent, ACTION_NAMES.index(action_name), :, none] = 1.0

    inspection = source_inspection_model()
    inspect = ACTION_NAMES.index("inspect_target")
    calibrate = ACTION_NAMES.index("calibrate_beacon")
    for latent in range(2):
        for successor in range(len(STATE_NAMES)):
            row = inspection[successor]
            if latent == 1:
                row = _swap_labels_0_and_1(row)
            observations[latent, inspect, successor, :3] = row

        # The beacon is known healthy and is not part of target state.  Its
        # distribution is therefore the source healthy inspection row for
        # every target successor.
        beacon_row = inspection[STATE_NAMES.index("healthy")]
        if latent == 1:
            beacon_row = _swap_labels_0_and_1(beacon_row)
        observations[latent, calibrate, :, :3] = beacon_row
    return observations


def _reward() -> np.ndarray:
    reward = np.zeros((len(ACTION_NAMES), len(STATE_NAMES), len(STATE_NAMES)))
    action_cost = {
        "do_nothing": 0.0,
        "replace_target": SOURCE_REPLACEMENT_REWARD + SOURCE_MOBILISATION_REWARD,
        "inspect_target": SOURCE_INSPECTION_REWARD + SOURCE_MOBILISATION_REWARD,
        "calibrate_beacon": SOURCE_INSPECTION_REWARD + SOURCE_MOBILISATION_REWARD,
    }
    failed = STATE_NAMES.index("failed")
    for action_name, cost in action_cost.items():
        action = ACTION_NAMES.index(action_name)
        reward[action, :, :] = cost
        reward[action, failed, :] += PROJECTED_FAILURE_REWARD
    return reward


def build_family() -> V73MaintenanceFamily:
    kernel = SensorCodebookKernel(
        action_names=ACTION_NAMES,
        observation_names=OBSERVATION_NAMES,
        state_names=STATE_NAMES,
        transition=_transition(),
        observation=_observation(),
        reward=_reward(),
        discount=SOURCE_DISCOUNT,
    )
    initial = np.stack(
        (0.5 * SOURCE_INITIAL_BELIEF, 0.5 * SOURCE_INITIAL_BELIEF), axis=0
    )
    return V73MaintenanceFamily(kernel=kernel, initial_belief=initial)


def calibration_mutual_information_nats() -> float:
    p = SOURCE_INSPECTION_ACCURACY
    entropy = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
    return float(math.log(2.0) - entropy)


def healthy_degraded_inspection_total_variation() -> float:
    rows = source_inspection_model()
    return float(0.5 * np.abs(rows[0] - rows[1]).sum())


def paired_label_threshold_diagnostics() -> dict[str, float | bool]:
    """Binary proxy fixed in the design lock; no planner is called here."""
    prior_degraded = float(SOURCE_INITIAL_BELIEF[1])
    prior_healthy = float(SOURCE_INITIAL_BELIEF[0])
    p = SOURCE_INSPECTION_ACCURACY
    same_given_same_condition = p * p + (1.0 - p) * (1.0 - p)
    same_given_different_condition = 2.0 * p * (1.0 - p)
    same_probability = (
        prior_healthy * same_given_same_condition
        + prior_degraded * same_given_different_condition
    )
    degraded_given_same = (
        prior_degraded * same_given_different_condition / same_probability
    )
    different_probability = 1.0 - same_probability
    degraded_given_different = (
        prior_degraded * same_given_same_condition / different_probability
    )
    replacement_cost = -(
        SOURCE_REPLACEMENT_REWARD + SOURCE_MOBILISATION_REWARD
    )
    failure_cost = -PROJECTED_FAILURE_REWARD
    threshold = replacement_cost / failure_cost
    return {
        "replacement_threshold": float(threshold),
        "degraded_posterior_same_label": float(degraded_given_same),
        "degraded_posterior_different_label": float(degraded_given_different),
        "threshold_straddled": bool(
            degraded_given_same < threshold < degraded_given_different
        ),
    }


def fixed_structural_policy(horizon: int = 5) -> dict[str, Any]:
    """Build the preregistered non-optimized adaptive policy tree."""
    if horizon != 5:
        raise ValueError("V73 fixed structural policy is registered only at horizon five")
    none = OBSERVATION_NAMES.index("none")
    do_nothing = ACTION_NAMES.index("do_nothing")
    replace = ACTION_NAMES.index("replace_target")
    inspect = ACTION_NAMES.index("inspect_target")
    calibrate = ACTION_NAMES.index("calibrate_beacon")

    terminal = {"terminal": True, "horizon": 0, "value": 0.0}
    fourth = {
        "terminal": False,
        "horizon": 1,
        "selected_action": do_nothing,
        "branches": {},
    }
    third = {
        "terminal": False,
        "horizon": 2,
        "selected_action": do_nothing,
        "branches": {none: fourth},
    }

    calibration_branches: dict[int, dict[str, Any]] = {}
    for calibration_label in (0, 1):
        target_branches: dict[int, dict[str, Any]] = {}
        for target_label in (0, 1, 2):
            control = (
                replace
                if target_label == 2 or target_label != calibration_label
                else do_nothing
            )
            target_branches[target_label] = {
                "terminal": False,
                "horizon": 3,
                "selected_action": control,
                "branches": {none: third},
            }
        calibration_branches[calibration_label] = {
            "terminal": False,
            "horizon": 4,
            "selected_action": inspect,
            "branches": target_branches,
        }
    policy = {
        "terminal": False,
        "horizon": 5,
        "selected_action": calibrate,
        "branches": calibration_branches,
    }
    # Keep the terminal marker reachable for schema tests without placing it
    # in the branches traversed by evaluate_policy_exact.
    policy["registered_terminal_template"] = terminal
    return policy


def structural_resource_metrics(horizon: int = 5) -> dict[str, int]:
    family = build_family()
    kernel = family.kernel
    # do_nothing and replace have one observation, inspect has three, and
    # calibration has two: total observation-action branching sum seven.
    branching_sum = 7
    return {
        "states": len(kernel.state_names),
        "actions": len(kernel.action_names),
        "observations": len(kernel.observation_names),
        "dense_kernel_bytes": int(
            kernel.transition.nbytes + kernel.observation.nbytes + kernel.reward.nbytes
        ),
        "exact_bellman_node_upper_bound": sum(
            branching_sum**depth for depth in range(horizon)
        ),
    }


def known_reward_diagnostics() -> dict[str, float | int | bool]:
    reward = build_family().kernel.reward
    positive = int(np.count_nonzero(reward > 0.0))
    calibrate = ACTION_NAMES.index("calibrate_beacon")
    return {
        "strictly_positive_immediate_reward_count": positive,
        "maximum_immediate_reward": float(reward.max()),
        "calibration_maximum_immediate_reward": float(reward[calibrate].max()),
        "calibration_beacon_harvestable": False,
        "calibration_transition_matches_source_do_nothing": bool(
            np.array_equal(
                build_family().kernel.transition[calibrate], SOURCE_DETERIORATION
            )
        ),
    }
