#!/usr/bin/env python3
"""Engineered shared-support active-sensing fixtures for the V72 oracle.

These fixtures are implementation tests, not external benchmark evidence.  They
reuse the frozen V71 exact-planning kernel without modifying its semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from v71_exact_planning import SensorCodebookKernel, exact_step


ACTION_NAMES = ("calibrate", "inspect", "repair_A", "repair_B")
OBSERVATION_NAMES = ("red", "blue")
STATE_NAMES = (
    "ready_A",
    "ready_B",
    "calibrated_A",
    "calibrated_B",
    "inspected_A",
    "inspected_B",
    "terminal",
)
LATENT_NAMES = ("canonical", "reversed")


@dataclass(frozen=True)
class ActiveSensingOracle:
    name: str
    kind: str
    kernel: SensorCodebookKernel
    initial_belief: np.ndarray

    def __post_init__(self) -> None:
        belief = np.asarray(self.initial_belief, dtype=np.float64)
        if belief.shape != (2, len(STATE_NAMES)):
            raise ValueError("V72 initial joint belief shape mismatch")
        if np.any(belief < 0.0) or not np.isclose(
            belief.sum(), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ValueError("V72 initial joint belief is invalid")
        belief.setflags(write=False)
        object.__setattr__(self, "initial_belief", belief)


def _deterministic_transition() -> np.ndarray:
    transition = np.zeros((len(ACTION_NAMES), len(STATE_NAMES), len(STATE_NAMES)))
    terminal = STATE_NAMES.index("terminal")
    for action in range(len(ACTION_NAMES)):
        transition[action, terminal, terminal] = 1.0

    calibrate = ACTION_NAMES.index("calibrate")
    inspect = ACTION_NAMES.index("inspect")
    transition[calibrate, 0, 2] = 1.0
    transition[calibrate, 1, 3] = 1.0
    for state in (2, 3, 4, 5):
        transition[calibrate, state, state] = 1.0

    transition[inspect, 0, 4] = 1.0
    transition[inspect, 1, 5] = 1.0
    transition[inspect, 2, 4] = 1.0
    transition[inspect, 3, 5] = 1.0
    transition[inspect, 4, 4] = 1.0
    transition[inspect, 5, 5] = 1.0

    for action in (ACTION_NAMES.index("repair_A"), ACTION_NAMES.index("repair_B")):
        for state in range(terminal):
            transition[action, state, terminal] = 1.0
    return transition


def _label_probabilities(logical_label: str, latent: int, reliability: float) -> np.ndarray:
    if logical_label not in ("A", "B") or latent not in (0, 1):
        raise ValueError("V72 sensor label or codebook is invalid")
    canonical_red = logical_label == "A"
    red_is_correct = canonical_red if latent == 0 else not canonical_red
    red = reliability if red_is_correct else 1.0 - reliability
    return np.array((red, 1.0 - red), dtype=np.float64)


def _observation(reliability: float) -> np.ndarray:
    observation = np.full(
        (2, len(ACTION_NAMES), len(STATE_NAMES), len(OBSERVATION_NAMES)),
        0.5,
        dtype=np.float64,
    )
    calibrate = ACTION_NAMES.index("calibrate")
    inspect = ACTION_NAMES.index("inspect")
    for latent in range(2):
        # A calibrated state carries the known reference target. Because the
        # V71 kernel indexes observations by (action, successor), repeating
        # calibration in this phase repeats the reference reading. At the
        # preregistered horizon this is legal but dominated by inspection.
        for state in (STATE_NAMES.index("calibrated_A"), STATE_NAMES.index("calibrated_B")):
            observation[latent, calibrate, state] = _label_probabilities(
                "A", latent, reliability
            )
        observation[latent, inspect, STATE_NAMES.index("inspected_A")] = (
            _label_probabilities("A", latent, reliability)
        )
        observation[latent, inspect, STATE_NAMES.index("inspected_B")] = (
            _label_probabilities("B", latent, reliability)
        )
    return observation


def _condition(state: int) -> str | None:
    name = STATE_NAMES[state]
    if name.endswith("_A"):
        return "A"
    if name.endswith("_B"):
        return "B"
    return None


def _reward(kind: str, transition: np.ndarray) -> np.ndarray:
    reward = np.zeros_like(transition)
    terminal = STATE_NAMES.index("terminal")
    for action_name in ("calibrate", "inspect"):
        action = ACTION_NAMES.index(action_name)
        for state in range(terminal):
            reward[action, state] = -1.0

    for state in range(terminal):
        condition = _condition(state)
        if kind == "positive":
            reward[ACTION_NAMES.index("repair_A"), state, terminal] = (
                10.0 if condition == "A" else -20.0
            )
            reward[ACTION_NAMES.index("repair_B"), state, terminal] = (
                10.0 if condition == "B" else -20.0
            )
        elif kind == "negative_control":
            reward[ACTION_NAMES.index("repair_A"), state, terminal] = 5.0
            reward[ACTION_NAMES.index("repair_B"), state, terminal] = 4.0
        else:
            raise ValueError("V72 oracle kind is invalid")
    return reward


def build_oracle(kind: str, *, reliability: float = 0.9) -> ActiveSensingOracle:
    if not 0.5 < reliability < 1.0:
        raise ValueError("V72 reliability must lie in (0.5,1)")
    transition = _deterministic_transition()
    kernel = SensorCodebookKernel(
        action_names=ACTION_NAMES,
        observation_names=OBSERVATION_NAMES,
        state_names=STATE_NAMES,
        transition=transition,
        observation=_observation(reliability),
        reward=_reward(kind, transition),
        discount=1.0,
    )
    belief = np.zeros((2, len(STATE_NAMES)), dtype=np.float64)
    belief[:, STATE_NAMES.index("ready_A")] = 0.25
    belief[:, STATE_NAMES.index("ready_B")] = 0.25
    names = {
        "positive": "calibrate_inspect_repair",
        "negative_control": "dominant_safe_repair",
    }
    return ActiveSensingOracle(names[kind], kind, kernel, belief)


def mutual_information(joint: np.ndarray) -> float:
    value = np.asarray(joint, dtype=np.float64)
    if value.ndim != 2 or np.any(value < 0.0):
        raise ValueError("V72 mutual-information joint table is invalid")
    total = float(value.sum())
    if not np.isclose(total, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("V72 mutual-information joint table is not normalized")
    row = value.sum(axis=1, keepdims=True)
    column = value.sum(axis=0, keepdims=True)
    independent = row @ column
    mask = value > 0.0
    return float(np.sum(value[mask] * np.log(value[mask] / independent[mask])))


def structural_diagnostics(fixture: ActiveSensingOracle) -> dict[str, Any]:
    kernel = fixture.kernel
    calibrate = ACTION_NAMES.index("calibrate")
    step = exact_step(kernel, fixture.initial_belief, calibrate)
    latent_observation = np.zeros((2, 2), dtype=np.float64)
    for observation, posterior in step["posteriors"].items():
        latent_observation[:, observation] = (
            float(step["probabilities"][observation]) * posterior.sum(axis=1)
        )

    # With a known codebook, inspection is a binary symmetric channel from
    # condition to observation. Compute this separately from any planner.
    state_observation = np.zeros((2, 2), dtype=np.float64)
    for condition, successor in enumerate((4, 5)):
        state_observation[condition] = 0.5 * kernel.observation[0, 1, successor]

    support = kernel.observation > 0.0
    return {
        "calibration_mutual_information_nats": mutual_information(latent_observation),
        "inspection_state_mutual_information_given_codebook_nats": mutual_information(
            state_observation
        ),
        "point_model_supports_identical": bool(np.array_equal(support[0], support[1])),
        "minimum_observation_probability": float(kernel.observation.min()),
        "point_model_on_support_rate": 1.0 if np.all(support) else 0.0,
        "fallback_count": 0,
        "repeated_calibration_in_calibrated_phase_is_informative": True,
        "calibration_after_inspection_is_uninformative": bool(
            np.allclose(kernel.observation[:, calibrate, 4:6], 0.5)
        ),
    }
