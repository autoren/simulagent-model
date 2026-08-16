#!/usr/bin/env python3
"""Execute frozen policies in the unchanged pinned POBAX POMDP runtime."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def project_python() -> Path:
    return Path(__file__).resolve().parent


sys.path.insert(0, str(project_python()))
from v62_external_pomdp import (  # noqa: E402
    ExactPlanner,
    POMDPModel,
    belief_key,
    condition_initial,
    observation_only_belief,
    public_policy_action_distribution,
    terminal_mask,
    update_belief,
)


def load_model(path: Path) -> POMDPModel:
    payload = json.loads(path.read_text())
    return POMDPModel(
        payload["name"], tuple(payload["states"]), tuple(payload["actions"]),
        tuple(payload["observations"]), float(payload["discount"]),
        np.asarray(payload["initial"]), np.asarray(payload["transition"]),
        np.asarray(payload["observation"]), np.asarray(payload["reward"]),
    )


def load_official_runtime(path: Path):
    spec = importlib.util.spec_from_file_location("v62_pinned_pobax_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned POBAX runtime source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def oracle_actions(model: POMDPModel, states: np.ndarray, horizon: int) -> np.ndarray:
    terminals = terminal_mask(model)

    @lru_cache(maxsize=None)
    def decision(state: int, remaining: int) -> tuple[float, int]:
        if remaining <= 0 or terminals[state]:
            return 0.0, 0
        values = []
        for action in range(len(model.actions)):
            value = 0.0
            for successor, probability in enumerate(model.transition[action, state]):
                if probability > 0.0:
                    value += probability * (
                        model.reward[action, state, successor]
                        + model.discount * decision(successor, remaining - 1)[0]
                    )
            values.append(value)
        maximum = max(values)
        action = next(index for index, value in enumerate(values) if maximum - value <= 1e-12)
        return float(maximum), action

    return np.asarray([decision(int(state), horizon)[1] for state in states], dtype=np.int32)


def public_actions(
    model: POMDPModel,
    planner: ExactPlanner,
    beliefs: np.ndarray,
    observations: np.ndarray,
    horizon: int,
    policy: str,
) -> np.ndarray:
    cache: dict[tuple[tuple[float, ...], int, int, str], int] = {}
    actions = np.zeros(len(beliefs), dtype=np.int32)
    for index, (belief, observation) in enumerate(zip(beliefs, observations, strict=True)):
        key = (belief_key(belief), int(observation), horizon, policy)
        if key not in cache:
            distribution = public_policy_action_distribution(
                model, planner, belief, int(observation), horizon, policy
            )
            cache[key] = int(np.argmax(distribution))
        actions[index] = cache[key]
    return actions


def update_public_beliefs(
    model: POMDPModel,
    beliefs: np.ndarray,
    actions: np.ndarray,
    observations: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    updated = np.array(beliefs, copy=True)
    cache: dict[tuple[tuple[float, ...], int, int], np.ndarray] = {}
    for index in np.flatnonzero(active):
        key = (belief_key(beliefs[index]), int(actions[index]), int(observations[index]))
        if key not in cache:
            cache[key] = update_belief(model, beliefs[index], key[1], key[2])[0]
        updated[index] = cache[key]
    return updated


def run_cell(runtime, model: POMDPModel, request: dict[str, object]) -> dict[str, object]:
    if not np.allclose(model.observation, model.observation[0], atol=1e-12, rtol=0.0):
        raise RuntimeError("pinned POBAX runtime requires action-independent observations")
    env = runtime.POMDP(
        jnp.asarray(model.transition), jnp.asarray(model.reward),
        jnp.asarray(model.initial), float(model.discount),
        jnp.asarray(model.observation[0]), fully_observable=False,
    )
    params = env.default_params
    episodes = int(request["episodes"])
    horizon = int(request["horizon"])
    seed = int(request["seed"])
    policy = str(request["policy"])
    reset_keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    reset_batch = jax.jit(jax.vmap(lambda key: env.reset_env(key, params)))
    observations_jax, states_jax = reset_batch(reset_keys)
    observations = np.asarray(observations_jax).argmax(axis=1).astype(np.int32)
    states = np.asarray(states_jax).astype(np.int32)
    beliefs = np.stack([condition_initial(model, int(obs))[0] for obs in observations])
    returns = np.zeros(episodes, dtype=np.float64)
    active = np.ones(episodes, dtype=bool)
    planner = ExactPlanner(model)
    # POBAX marks the action argument static. Execute one vectorized group per
    # action so the unchanged upstream method receives an ordinary integer.
    step_batches = {
        action: jax.jit(
            jax.vmap(
                lambda key, state, frozen_action=action: env.step_env(
                    key, state, frozen_action, params
                )
            )
        )
        for action in range(len(model.actions))
    }
    for step in range(horizon):
        remaining = horizon - step
        if policy == "fully_observed_oracle":
            actions = oracle_actions(model, states, remaining)
        else:
            actions = public_actions(model, planner, beliefs, observations, remaining, policy)
        step_root = jax.random.fold_in(jax.random.PRNGKey(seed + 104729), step)
        step_keys = jax.random.split(step_root, episodes)
        rewards = np.zeros(episodes, dtype=np.float64)
        done = np.zeros(episodes, dtype=bool)
        next_observations = np.array(observations, copy=True)
        next_states = np.array(states, copy=True)
        for action in range(len(model.actions)):
            indices = np.flatnonzero(active & (actions == action))
            if len(indices) == 0:
                continue
            next_obs_jax, next_states_jax, rewards_jax, done_jax, _ = step_batches[action](
                step_keys[indices], jnp.asarray(states[indices])
            )
            rewards[indices] = np.asarray(rewards_jax, dtype=np.float64)
            done[indices] = np.asarray(done_jax, dtype=bool)
            next_observations[indices] = np.asarray(next_obs_jax).argmax(axis=1).astype(np.int32)
            next_states[indices] = np.asarray(next_states_jax).astype(np.int32)
        returns += active * (model.discount**step) * rewards
        continue_active = active & ~done
        if policy != "fully_observed_oracle" and step + 1 < horizon:
            beliefs = update_public_beliefs(
                model, beliefs, actions, next_observations, continue_active
            )
        observations = next_observations
        states = next_states
        active = continue_active
    return {
        "model_id": request["model_id"],
        "horizon": horizon,
        "policy": policy,
        "episodes": episodes,
        "seed": seed,
        "mean_return": float(returns.mean()),
        "minimum_return": float(returns.min()),
        "maximum_return": float(returns.max()),
        "finite_return_rate": float(np.isfinite(returns).mean()),
        "terminated_fraction": float((~active).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    request = json.loads(Path(args.request).read_text())
    runtime_path = bundle / "source/pobax/envs/classic/pomdp.py"
    runtime = load_official_runtime(runtime_path)
    records = []
    models: dict[str, POMDPModel] = {}
    for cell in request["cells"]:
        model_id = cell["model_id"]
        if model_id not in models:
            models[model_id] = load_model(bundle / f"models/{model_id}/model.json")
        records.append(run_cell(runtime, models[model_id], cell))
    result = {
        "schema_version": 62,
        "experiment": "v62_pinned_pobax_runtime_rollout",
        "runtime_source_sha256": __import__("hashlib").sha256(runtime_path.read_bytes()).hexdigest(),
        "runtime_versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "chex": chex.__version__,
            "gymnax": importlib.metadata.version("gymnax"),
            "numpy": np.__version__,
        },
        "records": records,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
