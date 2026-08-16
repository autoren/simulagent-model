#!/usr/bin/env python3
"""Cross-check V63 family arrays and samples in the unchanged pinned POBAX runtime."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def load_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("v63_pinned_pobax_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned POBAX runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def family_transition(anchor: dict, identity: int, theta: float) -> np.ndarray:
    transition = np.asarray(anchor["transition"], dtype=np.float64).copy()
    for source, same_side in ((0, 2), (1, 3), (2, 2), (3, 3)):
        other = 3 if same_side == 2 else 2
        same_probability = theta if identity == 0 else 1.0 - theta
        transition[0, source] = 0.0
        transition[0, source, same_side] = same_probability
        transition[0, source, other] = 1.0 - same_probability
    return transition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    runtime = load_runtime(Path(args.runtime).resolve())
    anchor = json.loads(Path(args.anchor).read_text())
    request = json.loads(Path(args.request).read_text())
    observation = np.asarray(anchor["observation"], dtype=np.float64)
    if not np.allclose(observation, observation[0], atol=0.0, rtol=0.0):
        raise RuntimeError("pinned runtime requires action-independent observations")
    records = []
    environments = {}
    compiled = {}
    for cell in request["cells"]:
        identity, theta = int(cell["identity"]), float(cell["theta"])
        key = (identity, theta)
        expected_transition = family_transition(anchor, identity, theta)
        if key not in environments:
            env = runtime.POMDP(
                jnp.asarray(expected_transition),
                jnp.asarray(anchor["reward"], dtype=jnp.float64),
                jnp.asarray(anchor["initial"], dtype=jnp.float64),
                float(anchor["discount"]),
                jnp.asarray(observation[0]),
                fully_observable=False,
            )
            environments[key] = env
            params = env.default_params
            compiled[key] = jax.jit(jax.vmap(
                lambda sample_key, state, frozen_env=env, frozen_params=params:
                frozen_env.step_env(sample_key, state, 0, frozen_params)
            ))
        env = environments[key]
        episodes = int(cell["episodes"])
        source_state = int(cell["source_state"])
        keys = jax.random.split(jax.random.PRNGKey(int(cell["seed"])), episodes)
        states = jnp.full((episodes,), source_state, dtype=jnp.int32)
        observed, successors, _, _, _ = compiled[key](keys, states)
        observed_index = np.asarray(observed).argmax(axis=1)
        successor_index = np.asarray(successors, dtype=int)
        empirical = np.zeros((5, 4), dtype=np.float64)
        for successor, report in zip(successor_index, observed_index, strict=True):
            empirical[successor, report] += 1.0 / episodes
        expected_joint = expected_transition[0, source_state, :, None] * observation[0]
        records.append({
            "id": cell["id"],
            "identity": identity,
            "theta": theta,
            "source_state": source_state,
            "episodes": episodes,
            "maximum_transition_array_error": float(
                np.abs(np.asarray(env.T) - expected_transition).max()
            ),
            "maximum_observation_array_error": float(
                np.abs(np.asarray(env.phi) - observation[0]).max()
            ),
            "maximum_empirical_probability_error": float(
                np.abs(empirical - expected_joint).max()
            ),
            "finite_rate": float(np.isfinite(empirical).mean()),
        })
    result = {
        "schema_version": 63,
        "experiment": "v63_pinned_pobax_runtime_crosscheck",
        "cells": records,
        "completed_fraction": len(records) / len(request["cells"]),
        "maximum_transition_array_error": max(
            row["maximum_transition_array_error"] for row in records
        ),
        "maximum_observation_array_error": max(
            row["maximum_observation_array_error"] for row in records
        ),
        "maximum_empirical_probability_error": max(
            row["maximum_empirical_probability_error"] for row in records
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
