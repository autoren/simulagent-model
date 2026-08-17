#!/usr/bin/env python3
"""Source-grounded V75 NOVA paint adapter with a codebook reference beacon."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v68_cassandra_pomdp import parse_cassandra_pomdp_file
from v71_exact_planning import SensorCodebookKernel


SOURCE_REPOSITORY = "https://github.com/kylewray/nova"
SOURCE_COMMIT = "fa6f0bf038509cb7bb94fb79e38e691c6e6d83e9"
SOURCE_RELATIVE_PATH = "tests/benchmarks/algorithms/domains/paint_95.pomdp"
SOURCE_SHA256 = "2293f505297e6265c002c448c009f3a548d12590bcdb7234c188b7b1a68cd516"
SOURCE_PATH = (
    PROJECT_ROOT
    / "data/v75-active-sensing-confirmation/source-checkouts/nova"
    / SOURCE_RELATIVE_PATH
)

STATE_NAMES = ("NFL-NBL-NPA", "NFL-NBL-PA", "FL-NBL-PA", "FL-BL-NPA")
SOURCE_ACTION_NAMES = ("paint", "inspect", "ship", "reject")
ACTION_NAMES = ("calibrate_beacon", "paint", "inspect_target", "ship", "reject")
OBSERVATION_NAMES = ("label_NBL", "label_BL", "none")
LATENT_NAMES = ("canonical", "reverse_NBL_BL_labels")
SOURCE_DISCOUNT = 0.95
SOURCE_INSPECTION_ACCURACY = 0.75


@dataclass(frozen=True)
class V75PaintFamily:
    kernel: SensorCodebookKernel
    initial_belief: np.ndarray

    def __post_init__(self) -> None:
        belief = np.asarray(self.initial_belief, dtype=np.float64)
        if belief.shape != (2, 4):
            raise ValueError("V75 initial joint belief shape mismatch")
        if np.any(belief < 0.0) or not np.isclose(
            belief.sum(), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ValueError("V75 initial joint belief is invalid")
        belief.setflags(write=False)
        object.__setattr__(self, "initial_belief", belief)


def load_source_model():
    if file_sha256(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("V75 pinned NOVA paint source hash drifted")
    model = parse_cassandra_pomdp_file(SOURCE_PATH)
    if (
        model.states != STATE_NAMES
        or model.actions != SOURCE_ACTION_NAMES
        or model.observations != ("NBL", "BL")
        or model.discount != SOURCE_DISCOUNT
    ):
        raise RuntimeError("V75 pinned NOVA paint metadata drifted")
    return model


def _transition(source) -> np.ndarray:
    value = np.zeros((len(ACTION_NAMES), 4, 4), dtype=np.float64)
    value[ACTION_NAMES.index("calibrate_beacon")] = np.eye(4)
    for target_name, source_name in (
        ("paint", "paint"),
        ("inspect_target", "inspect"),
        ("ship", "ship"),
        ("reject", "reject"),
    ):
        value[ACTION_NAMES.index(target_name)] = source.transition[
            source.actions.index(source_name)
        ]
    return value


def _observation(source) -> np.ndarray:
    value = np.zeros((2, len(ACTION_NAMES), 4, len(OBSERVATION_NAMES)))
    calibrate = ACTION_NAMES.index("calibrate_beacon")
    inspect = ACTION_NAMES.index("inspect_target")
    none = OBSERVATION_NAMES.index("none")
    p = SOURCE_INSPECTION_ACCURACY

    value[0, calibrate, :, :2] = (p, 1.0 - p)
    value[1, calibrate, :, :2] = (1.0 - p, p)
    source_inspect = source.observation[source.actions.index("inspect")]
    value[0, inspect, :, :2] = source_inspect
    value[1, inspect, :, :2] = source_inspect[:, ::-1]
    for action_name in ("paint", "ship", "reject"):
        value[:, ACTION_NAMES.index(action_name), :, none] = 1.0
    return value


def _reward(source) -> np.ndarray:
    value = np.zeros((len(ACTION_NAMES), 4, 4), dtype=np.float64)
    for target_name, source_name in (
        ("paint", "paint"),
        ("inspect_target", "inspect"),
        ("ship", "ship"),
        ("reject", "reject"),
    ):
        value[ACTION_NAMES.index(target_name)] = source.reward[
            source.actions.index(source_name)
        ]
    return value


def build_family() -> V75PaintFamily:
    source = load_source_model()
    kernel = SensorCodebookKernel(
        action_names=ACTION_NAMES,
        observation_names=OBSERVATION_NAMES,
        state_names=STATE_NAMES,
        transition=_transition(source),
        observation=_observation(source),
        reward=_reward(source),
        discount=SOURCE_DISCOUNT,
    )
    initial = np.stack((0.5 * source.initial, 0.5 * source.initial))
    return V75PaintFamily(kernel=kernel, initial_belief=initial)


def calibration_mutual_information_nats() -> float:
    p = SOURCE_INSPECTION_ACCURACY
    entropy = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
    return float(math.log(2.0) - entropy)


def target_inspection_total_variation() -> float:
    return float(abs(2.0 * SOURCE_INSPECTION_ACCURACY - 1.0))


def paired_decision_correct_probability() -> float:
    p = SOURCE_INSPECTION_ACCURACY
    return float(p * p + (1.0 - p) * (1.0 - p))


def fixed_structural_policy(horizon: int = 4) -> dict[str, Any]:
    if horizon != 4:
        raise ValueError("V75 fixed structural policy is registered only at horizon four")
    calibrate = ACTION_NAMES.index("calibrate_beacon")
    paint = ACTION_NAMES.index("paint")
    inspect = ACTION_NAMES.index("inspect_target")
    ship = ACTION_NAMES.index("ship")
    reject = ACTION_NAMES.index("reject")
    root_branches: dict[int, dict[str, Any]] = {}
    for beacon_label in (0, 1):
        target_branches: dict[int, dict[str, Any]] = {}
        for target_label in (0, 1):
            labels_match = beacon_label == target_label
            third_action = paint if labels_match else reject
            final_action = ship if labels_match else calibrate
            target_branches[target_label] = {
                "terminal": False,
                "horizon": 2,
                "selected_action": third_action,
                "branches": {
                    2: {
                        "terminal": False,
                        "horizon": 1,
                        "selected_action": final_action,
                        "branches": {},
                    }
                },
            }
        root_branches[beacon_label] = {
            "terminal": False,
            "horizon": 3,
            "selected_action": inspect,
            "branches": target_branches,
        }
    return {
        "terminal": False,
        "horizon": 4,
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


def structural_resource_metrics(horizon: int = 4) -> dict[str, int]:
    family = build_family()
    kernel = family.kernel
    branching_sum = 7
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
    source = load_source_model()
    rewards = initial_unconditioned_action_rewards()
    best = max(rewards, key=rewards.get)
    source_array_parity = all(
        np.array_equal(
            family.kernel.transition[ACTION_NAMES.index(target_name)],
            source.transition[source.actions.index(source_name)],
        )
        and np.array_equal(
            family.kernel.reward[ACTION_NAMES.index(target_name)],
            source.reward[source.actions.index(source_name)],
        )
        for target_name, source_name in (
            ("paint", "paint"),
            ("inspect_target", "inspect"),
            ("ship", "ship"),
            ("reject", "reject"),
        )
    )
    return {
        "calibration_mutual_information_nats": calibration_mutual_information_nats(),
        "target_inspection_total_variation": target_inspection_total_variation(),
        "paired_decision_correct_probability": paired_decision_correct_probability(),
        "point_model_supports_identical": bool(
            np.array_equal(
                family.kernel.observation[0] > 0.0,
                family.kernel.observation[1] > 0.0,
            )
        ),
        "source_array_parity": bool(source_array_parity),
        "calibration_beacon_harvestable": False,
        "calibration_transition_is_identity": bool(
            np.array_equal(
                family.kernel.transition[ACTION_NAMES.index("calibrate_beacon")],
                np.eye(4),
            )
        ),
        "initial_best_unconditioned_action": best,
        "initial_best_unconditioned_expected_reward": rewards[best],
    }
