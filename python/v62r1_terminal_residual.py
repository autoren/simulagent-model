#!/usr/bin/env python3
"""Independent terminal-aware Bellman residual checker for V62r1."""
from __future__ import annotations

import numpy as np

from v62_external_pomdp import ExactPlanner, POMDPModel


def support_is_all_action_absorbing(
    model: POMDPModel, belief: np.ndarray, *, tolerance: float = 1e-12
) -> bool:
    """Return whether every positive-support state self-loops under every action."""
    values = np.asarray(belief, dtype=np.float64)
    support = [index for index, probability in enumerate(values) if probability > 1e-14]
    if not support:
        raise ValueError("belief has empty positive support")
    for state in support:
        for action in range(len(model.actions)):
            for successor in range(len(model.states)):
                expected = 1.0 if successor == state else 0.0
                if abs(float(model.transition[action, state, successor]) - expected) > tolerance:
                    return False
    return True


def terminal_aware_bellman_residual(
    model: POMDPModel,
    planner: ExactPlanner,
    belief: np.ndarray,
    horizon: int,
) -> float:
    """Independently recompose action values with the runtime's terminal semantics."""
    values = np.asarray(belief, dtype=np.float64)
    if values.shape != (len(model.states),):
        raise ValueError("belief shape does not match model state count")
    if not np.isfinite(values).all() or np.any(values < -1e-14):
        raise ValueError("belief must be finite and nonnegative")
    mass = float(values.sum())
    if abs(mass - 1.0) > 1e-12:
        raise ValueError("belief must be normalized")
    decision = planner.decision(values, horizon)
    if horizon <= 0 or support_is_all_action_absorbing(model, values):
        return float(
            max(
                abs(float(decision.value)),
                *(abs(float(action_value)) for action_value in decision.q_values),
            )
        )

    recomposed: list[float] = []
    for action in range(len(model.actions)):
        action_value = 0.0
        for state in range(len(model.states)):
            for successor in range(len(model.states)):
                action_value += (
                    float(values[state])
                    * float(model.transition[action, state, successor])
                    * float(model.reward[action, state, successor])
                )
        if horizon > 1:
            continuation = 0.0
            for observation in range(len(model.observations)):
                joint = np.zeros(len(model.states), dtype=np.float64)
                for successor in range(len(model.states)):
                    predicted = 0.0
                    for state in range(len(model.states)):
                        predicted += float(values[state]) * float(
                            model.transition[action, state, successor]
                        )
                    joint[successor] = predicted * float(
                        model.observation[action, successor, observation]
                    )
                observation_probability = float(joint.sum())
                if observation_probability <= 1e-15:
                    continue
                posterior = joint / observation_probability
                child = planner.decision(posterior, horizon - 1)
                continuation += observation_probability * float(child.value)
            action_value += float(model.discount) * continuation
        recomposed.append(action_value)
    return float(
        max(
            abs(recomposed[action] - float(decision.q_values[action]))
            for action in range(len(model.actions))
        )
    )
