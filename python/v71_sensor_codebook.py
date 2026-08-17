#!/usr/bin/env python3
"""Exact V71 sensor-codebook beliefs and public-prefix enumeration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v71_cassandra_pomdp import ParsedPOMDPSource


LATENT_NAMES = ("canonical_dominant", "reversed_dominant")


@dataclass(frozen=True)
class PublicPrefix:
    depth: int
    action_index: int | None
    observation_index: int | None
    probability: float
    joint_belief: np.ndarray

    def __post_init__(self) -> None:
        belief = np.asarray(self.joint_belief, dtype=np.float64)
        if belief.ndim != 2 or belief.shape[0] != len(LATENT_NAMES):
            raise ValueError("V71 joint belief must have shape (2, states)")
        if not np.isfinite(belief).all() or np.any(belief < 0.0):
            raise ValueError("V71 joint belief must be finite and nonnegative")
        if not np.isclose(belief.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("V71 joint belief must sum to one")
        belief.setflags(write=False)
        object.__setattr__(self, "joint_belief", belief)


def sensor_observation_models(
    parsed: ParsedPOMDPSource, *, reliability: float = 0.85
) -> np.ndarray:
    if not 0.5 < reliability < 1.0:
        raise ValueError("V71 reliability must be strictly between 0.5 and 1")
    source = parsed.model.observation
    reversed_source = source[..., ::-1]
    result = np.stack(
        (
            reliability * source + (1.0 - reliability) * reversed_source,
            reliability * reversed_source + (1.0 - reliability) * source,
        )
    )
    if not np.allclose(result.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("V71 sensor observation rows are not normalized")
    if not np.array_equal(result[0] > 0.0, result[1] > 0.0):
        raise ValueError("V71 sensor point-model supports differ")
    result.setflags(write=False)
    return result


def initial_joint_belief(parsed: ParsedPOMDPSource) -> np.ndarray:
    belief = np.stack((0.5 * parsed.model.initial, 0.5 * parsed.model.initial))
    belief.setflags(write=False)
    return belief


def update_joint_belief(
    parsed: ParsedPOMDPSource,
    sensor_observation: np.ndarray,
    joint_belief: np.ndarray,
    action_index: int,
    observation_index: int,
) -> tuple[float, np.ndarray]:
    model = parsed.model
    belief = np.asarray(joint_belief, dtype=np.float64)
    if belief.shape != (len(LATENT_NAMES), len(model.states)):
        raise ValueError("V71 joint belief shape does not match source model")
    predicted = np.einsum(
        "zs,sq->zq", belief, model.transition[action_index], optimize=True
    )
    unnormalized = predicted * sensor_observation[
        :, action_index, :, observation_index
    ]
    probability = float(unnormalized.sum())
    if probability <= 0.0:
        raise ValueError("cannot update V71 belief on a zero-probability observation")
    posterior = unnormalized / probability
    posterior.setflags(write=False)
    return probability, posterior


def enumerate_public_prefixes(
    parsed: ParsedPOMDPSource, *, reliability: float = 0.85
) -> tuple[PublicPrefix, ...]:
    sensor = sensor_observation_models(parsed, reliability=reliability)
    initial = initial_joint_belief(parsed)
    records = [
        PublicPrefix(
            depth=0,
            action_index=None,
            observation_index=None,
            probability=1.0,
            joint_belief=initial,
        )
    ]
    for action in range(len(parsed.model.actions)):
        for observation in range(len(parsed.model.observations)):
            try:
                probability, posterior = update_joint_belief(
                    parsed, sensor, initial, action, observation
                )
            except ValueError as exc:
                if "zero-probability" in str(exc):
                    continue
                raise
            records.append(
                PublicPrefix(
                    depth=1,
                    action_index=action,
                    observation_index=observation,
                    probability=probability,
                    joint_belief=posterior,
                )
            )
    return tuple(records)
