#!/usr/bin/env python3
"""Exact dominant latent action-remapping family for V69."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from v62_external_pomdp import POMDPModel
from v64_external_eig import scaled_beta_2_2_quadrature
from v66_bayes_adaptive_reward import StaticKernel
from v68_multi_environment_exact import CommandChannelFamily, cycle_permutations


def build_dominant_remapping_family(
    model: POMDPModel,
    canonical_action_cycle: Sequence[str],
    *,
    quadrature_nodes: int,
    theta_support: tuple[float, float] = (0.6, 0.95),
) -> CommandChannelFamily:
    """Put θ mass on an identity-specific remapped source action."""
    low, high = map(float, theta_support)
    theta, theta_weights = scaled_beta_2_2_quadrature(quadrature_nodes, low, high)
    permutations, canonical = cycle_permutations(model, canonical_action_cycle)
    transitions = np.asarray(
        [
            [
                [
                    value * model.transition[permutation[action]]
                    + (1.0 - value) * model.transition[action]
                    for action in range(len(model.actions))
                ]
                for value in theta
            ]
            for permutation in permutations
        ],
        dtype=np.float64,
    )
    static_weights = np.concatenate([0.5 * theta_weights, 0.5 * theta_weights])
    identities = np.repeat(np.arange(2, dtype=np.int16), quadrature_nodes)
    thetas = np.tile(theta, 2)
    kernel = StaticKernel(
        action_names=model.actions,
        observation_names=model.observations,
        state_names=model.states,
        canonical_actions=canonical,
        transitions=transitions.reshape(
            2 * quadrature_nodes,
            len(model.actions),
            len(model.states),
            len(model.states),
        ),
        observations=model.observation,
        rewards=model.reward,
        discount=model.discount,
        identities=identities,
        thetas=thetas,
    )
    initial_belief = static_weights[:, None] * model.initial[None, :]
    return CommandChannelFamily(
        model=model,
        kernel=kernel,
        initial_belief=initial_belief,
        theta=theta,
        theta_weights=theta_weights,
        permutations=permutations,
        canonical_action_labels=tuple(str(label) for label in canonical_action_cycle),
        identity_names=(
            "dominant_forward_cycle_remapping",
            "dominant_backward_cycle_remapping",
        ),
        theta_support=(low, high),
    )
