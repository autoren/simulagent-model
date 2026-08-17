#!/usr/bin/env python3
"""Source-grounded V74 Tiger adapter with a non-harvestable codebook beacon."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from v71_exact_planning import SensorCodebookKernel


SOURCE_COMMIT = "bd0e4392247aebfe9a95b449275237dcc25e7737"
STATE_NAMES = ("tiger_left", "tiger_right")
ACTION_NAMES = ("calibrate_beacon", "listen_target", "open_left", "open_right")
OBSERVATION_NAMES = ("label_left", "label_right", "none")
LATENT_NAMES = ("canonical", "reverse_left_right_labels")

SOURCE_OBSERVATION_NOISE = 0.01
SOURCE_OBSERVATION_ACCURACY = 1.0 - SOURCE_OBSERVATION_NOISE
SOURCE_SAFE_OPEN_REWARD = 10.0
SOURCE_TIGER_OPEN_REWARD = -100.0
SOURCE_TARGET_LISTEN_REWARD = -1.0
SOURCE_DISCOUNT = 0.95
SOURCE_LISTEN_FLIP_PROBABILITY = 1e-9
PROJECT_BEACON_REWARD = -0.5


@dataclass(frozen=True)
class V74TigerFamily:
    kernel: SensorCodebookKernel
    initial_belief: np.ndarray
    source_commit: str = SOURCE_COMMIT

    def __post_init__(self) -> None:
        belief = np.asarray(self.initial_belief, dtype=np.float64)
        if belief.shape != (2, 2):
            raise ValueError("V74 initial joint belief shape mismatch")
        if np.any(belief < 0.0) or not np.isclose(
            belief.sum(), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ValueError("V74 initial joint belief is invalid")
        belief.setflags(write=False)
        object.__setattr__(self, "initial_belief", belief)


def source_listen_transition() -> np.ndarray:
    flip = SOURCE_LISTEN_FLIP_PROBABILITY
    return np.array(((1.0 - flip, flip), (flip, 1.0 - flip)), dtype=np.float64)


def source_open_transition() -> np.ndarray:
    return np.full((2, 2), 0.5, dtype=np.float64)


def _transition() -> np.ndarray:
    value = np.zeros((len(ACTION_NAMES), 2, 2), dtype=np.float64)
    value[ACTION_NAMES.index("calibrate_beacon")] = source_listen_transition()
    value[ACTION_NAMES.index("listen_target")] = source_listen_transition()
    value[ACTION_NAMES.index("open_left")] = source_open_transition()
    value[ACTION_NAMES.index("open_right")] = source_open_transition()
    return value


def _observation() -> np.ndarray:
    value = np.zeros((2, len(ACTION_NAMES), 2, len(OBSERVATION_NAMES)))
    p = SOURCE_OBSERVATION_ACCURACY
    calibrate = ACTION_NAMES.index("calibrate_beacon")
    listen = ACTION_NAMES.index("listen_target")
    none = OBSERVATION_NAMES.index("none")

    # The beacon's physical condition is known tiger-left and independent of
    # the target successor. The latent codebook changes only emitted labels.
    value[0, calibrate, :, :2] = (p, 1.0 - p)
    value[1, calibrate, :, :2] = (1.0 - p, p)

    canonical = np.array(((p, 1.0 - p), (1.0 - p, p)), dtype=np.float64)
    value[0, listen, :, :2] = canonical
    value[1, listen, :, :2] = canonical[:, ::-1]

    # pomdp-py assigns probability 0.5 to either label after opening,
    # independently of successor state. One none symbol is belief-equivalent.
    for action_name in ("open_left", "open_right"):
        value[:, ACTION_NAMES.index(action_name), :, none] = 1.0
    return value


def _reward() -> np.ndarray:
    value = np.zeros((len(ACTION_NAMES), 2, 2), dtype=np.float64)
    value[ACTION_NAMES.index("calibrate_beacon"), :, :] = PROJECT_BEACON_REWARD
    value[ACTION_NAMES.index("listen_target"), :, :] = SOURCE_TARGET_LISTEN_REWARD
    open_left = ACTION_NAMES.index("open_left")
    open_right = ACTION_NAMES.index("open_right")
    tiger_left = STATE_NAMES.index("tiger_left")
    tiger_right = STATE_NAMES.index("tiger_right")
    value[open_left, tiger_left, :] = SOURCE_TIGER_OPEN_REWARD
    value[open_left, tiger_right, :] = SOURCE_SAFE_OPEN_REWARD
    value[open_right, tiger_left, :] = SOURCE_SAFE_OPEN_REWARD
    value[open_right, tiger_right, :] = SOURCE_TIGER_OPEN_REWARD
    return value


def build_family() -> V74TigerFamily:
    kernel = SensorCodebookKernel(
        action_names=ACTION_NAMES,
        observation_names=OBSERVATION_NAMES,
        state_names=STATE_NAMES,
        transition=_transition(),
        observation=_observation(),
        reward=_reward(),
        discount=SOURCE_DISCOUNT,
    )
    return V74TigerFamily(kernel=kernel, initial_belief=np.full((2, 2), 0.25))


def calibration_mutual_information_nats() -> float:
    p = SOURCE_OBSERVATION_ACCURACY
    entropy = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
    return float(math.log(2.0) - entropy)


def target_listen_total_variation() -> float:
    p = SOURCE_OBSERVATION_ACCURACY
    return float(abs(2.0 * p - 1.0))


def paired_decision_correct_probability() -> float:
    p = SOURCE_OBSERVATION_ACCURACY
    return float(p * p + (1.0 - p) * (1.0 - p))


def fixed_structural_policy(horizon: int = 3) -> dict[str, Any]:
    if horizon != 3:
        raise ValueError("V74 fixed structural policy is registered only at horizon three")
    calibrate = ACTION_NAMES.index("calibrate_beacon")
    listen = ACTION_NAMES.index("listen_target")
    open_left = ACTION_NAMES.index("open_left")
    open_right = ACTION_NAMES.index("open_right")
    root_branches: dict[int, dict[str, Any]] = {}
    for beacon_label in (0, 1):
        target_branches: dict[int, dict[str, Any]] = {}
        for target_label in (0, 1):
            action = open_right if beacon_label == target_label else open_left
            target_branches[target_label] = {
                "terminal": False,
                "horizon": 1,
                "selected_action": action,
                "branches": {},
            }
        root_branches[beacon_label] = {
            "terminal": False,
            "horizon": 2,
            "selected_action": listen,
            "branches": target_branches,
        }
    return {
        "terminal": False,
        "horizon": 3,
        "selected_action": calibrate,
        "branches": root_branches,
    }


def initial_unconditioned_action_rewards() -> dict[str, float]:
    family = build_family()
    state_belief = family.initial_belief.sum(axis=0)
    return {
        name: float(
            np.einsum(
                "s,sq,sq->",
                state_belief,
                family.kernel.transition[action],
                family.kernel.reward[action],
                optimize=True,
            )
        )
        for action, name in enumerate(ACTION_NAMES)
    }


def structural_resource_metrics(horizon: int = 3) -> dict[str, int]:
    family = build_family()
    kernel = family.kernel
    branching_sum = 6  # two labels for each sensing action; one for each open
    return {
        "states": len(STATE_NAMES),
        "actions": len(ACTION_NAMES),
        "observations": len(OBSERVATION_NAMES),
        "dense_kernel_bytes": int(
            kernel.transition.nbytes + kernel.observation.nbytes + kernel.reward.nbytes
        ),
        "exact_bellman_node_upper_bound": sum(
            branching_sum**depth for depth in range(horizon)
        ),
    }


def structural_diagnostics() -> dict[str, float | bool | str]:
    family = build_family()
    rewards = initial_unconditioned_action_rewards()
    best = max(rewards, key=rewards.get)
    calibrate = ACTION_NAMES.index("calibrate_beacon")
    return {
        "calibration_mutual_information_nats": calibration_mutual_information_nats(),
        "target_listen_total_variation": target_listen_total_variation(),
        "paired_decision_correct_probability": paired_decision_correct_probability(),
        "point_model_supports_identical": bool(
            np.array_equal(
                family.kernel.observation[0] > 0.0,
                family.kernel.observation[1] > 0.0,
            )
        ),
        "calibration_beacon_harvestable": False,
        "calibration_transition_matches_source_listen": bool(
            np.array_equal(
                family.kernel.transition[calibrate], source_listen_transition()
            )
        ),
        "initial_best_unconditioned_action": best,
        "initial_best_unconditioned_expected_reward": rewards[best],
    }
