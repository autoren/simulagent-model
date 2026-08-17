#!/usr/bin/env python3
"""Deterministic finite export of the frozen V72 RockSample.jl blueprint."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from v71_exact_planning import SensorCodebookKernel


ACTION_NAMES = (
    "sample",
    "north",
    "east",
    "south",
    "west",
    "check_reference",
    "check_target",
)
OBSERVATION_NAMES = ("good", "bad", "none")
ROCK_POSITIONS = ((1, 1), (2, 2))
INITIAL_POSITION = (2, 1)
MAP_SIZE = (2, 2)
TERMINAL_NAME = "terminal"


@dataclass(frozen=True)
class RockSampleState:
    x: int
    y: int
    reference_good: bool
    target_good: bool

    @property
    def name(self) -> str:
        return (
            f"x{self.x}_y{self.y}_r{int(self.reference_good)}"
            f"_t{int(self.target_good)}"
        )


@dataclass(frozen=True)
class V72RockSampleFamily:
    kernel: SensorCodebookKernel
    states: tuple[RockSampleState | None, ...]
    initial_belief: np.ndarray
    source_commit: str
    observation_noise_floor_weight: float

    def __post_init__(self) -> None:
        belief = np.asarray(self.initial_belief, dtype=np.float64)
        if belief.shape != (2, len(self.states)):
            raise ValueError("V72 RockSample initial belief shape mismatch")
        if np.any(belief < 0.0) or not np.isclose(
            belief.sum(), 1.0, atol=1e-12, rtol=0.0
        ):
            raise ValueError("V72 RockSample initial belief is invalid")
        belief.setflags(write=False)
        object.__setattr__(self, "initial_belief", belief)


def enumerate_states() -> tuple[RockSampleState | None, ...]:
    """Match RockSample.jl's one-based x-fast stateindex, then terminal."""
    states: list[RockSampleState | None] = [None] * 17
    for target_good in (False, True):
        for reference_good in (False, True):
            for y in range(1, MAP_SIZE[1] + 1):
                for x in range(1, MAP_SIZE[0] + 1):
                    index = (
                        x
                        + MAP_SIZE[0] * (y - 1)
                        + 4 * int(reference_good)
                        + 8 * int(target_good)
                        - 1
                    )
                    states[index] = RockSampleState(
                        x, y, reference_good, target_good
                    )
    states[-1] = None
    return tuple(states)


def state_index(state: RockSampleState | None) -> int:
    if state is None:
        return 16
    return (
        state.x
        + MAP_SIZE[0] * (state.y - 1)
        + 4 * int(state.reference_good)
        + 8 * int(state.target_good)
        - 1
    )


def _next_position(state: RockSampleState, action: int) -> tuple[int, int]:
    if action in (0, 5, 6):
        return state.x, state.y
    directions = {
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
        4: (-1, 0),
    }
    dx, dy = directions[action]
    return state.x + dx, state.y + dy


def successor(state: RockSampleState | None, action: int) -> RockSampleState | None:
    if state is None:
        return None
    x, y = _next_position(state, action)
    if x > MAP_SIZE[0]:
        return None
    x = min(max(x, 1), MAP_SIZE[0])
    y = min(max(y, 1), MAP_SIZE[1])
    reference_good = state.reference_good
    target_good = state.target_good
    if action == 0 and (state.x, state.y) in ROCK_POSITIONS:
        rock = ROCK_POSITIONS.index((state.x, state.y))
        if rock == 0:
            reference_good = False
        else:
            target_good = False
    return RockSampleState(x, y, reference_good, target_good)


def source_reward(state: RockSampleState | None, action: int) -> float:
    if state is None:
        return 0.0
    x, _ = _next_position(state, action)
    if x > MAP_SIZE[0]:
        return 5.0
    if action == 0 and (state.x, state.y) in ROCK_POSITIONS:
        rock = ROCK_POSITIONS.index((state.x, state.y))
        good = state.reference_good if rock == 0 else state.target_good
        return 10.0 if good else -10.0
    if action > 4:
        return -0.5
    return 0.0


def source_check_distribution(
    state: RockSampleState, action: int, *, sensor_efficiency: float = 1.0
) -> np.ndarray:
    if action not in (5, 6):
        return np.array((0.0, 0.0, 1.0))
    rock = action - 5
    rx, ry = ROCK_POSITIONS[rock]
    distance = math.hypot(rx - state.x, ry - state.y)
    accuracy = 0.5 * (1.0 + math.exp(-distance * math.log(2.0) / sensor_efficiency))
    good = state.reference_good if rock == 0 else state.target_good
    if good:
        return np.array((accuracy, 1.0 - accuracy, 0.0))
    return np.array((1.0 - accuracy, accuracy, 0.0))


def wrapped_observation_distribution(
    state: RockSampleState | None,
    action: int,
    latent: int,
    *,
    noise_floor_weight: float = 0.2,
) -> np.ndarray:
    if state is None or action <= 4:
        return np.array((0.0, 0.0, 1.0))
    source = source_check_distribution(state, action)
    labels = source[:2] if latent == 0 else source[1::-1]
    labels = (1.0 - noise_floor_weight) * labels + noise_floor_weight * 0.5
    return np.array((labels[0], labels[1], 0.0))


def build_family() -> V72RockSampleFamily:
    states = enumerate_states()
    state_count = len(states)
    transition = np.zeros((len(ACTION_NAMES), state_count, state_count))
    reward = np.zeros_like(transition)
    observation = np.zeros((2, len(ACTION_NAMES), state_count, len(OBSERVATION_NAMES)))
    for action in range(len(ACTION_NAMES)):
        for index, state in enumerate(states):
            next_state = successor(state, action)
            next_index = state_index(next_state)
            transition[action, index, next_index] = 1.0
            reward[action, index, next_index] = source_reward(state, action)
    for latent in range(2):
        for action in range(len(ACTION_NAMES)):
            for next_index, next_state in enumerate(states):
                observation[latent, action, next_index] = (
                    wrapped_observation_distribution(next_state, action, latent)
                )
    kernel = SensorCodebookKernel(
        action_names=ACTION_NAMES,
        observation_names=OBSERVATION_NAMES,
        state_names=tuple(
            TERMINAL_NAME if state is None else state.name for state in states
        ),
        transition=transition,
        observation=observation,
        reward=reward,
        discount=0.95,
    )
    belief = np.zeros((2, state_count))
    for latent in range(2):
        for target_good in (False, True):
            state = RockSampleState(2, 1, True, target_good)
            belief[latent, state_index(state)] = 0.25
    return V72RockSampleFamily(
        kernel=kernel,
        states=states,
        initial_belief=belief,
        source_commit="c8b3566d30c5dd7be6c7790b4b9a54ebfcdeecde",
        observation_noise_floor_weight=0.2,
    )


def structural_resource_metrics(family: V72RockSampleFamily, horizon: int = 4) -> dict[str, int]:
    kernel = family.kernel
    # Five basic actions have one supported observation and two check actions
    # have two, so a full exact Bellman expansion has branching sum 9.
    branching_sum = 5 + 2 * 2
    bellman_nodes = sum(branching_sum**depth for depth in range(horizon))
    return {
        "states": len(kernel.state_names),
        "actions": len(kernel.action_names),
        "observations": len(kernel.observation_names),
        "dense_kernel_bytes": int(
            kernel.transition.nbytes + kernel.observation.nbytes + kernel.reward.nbytes
        ),
        "exact_bellman_node_upper_bound": bellman_nodes,
    }
