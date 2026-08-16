#!/usr/bin/env python3
"""Strict external POMDP parser, exact filter, and finite-horizon planner for V62."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class POMDPModel:
    name: str
    states: tuple[str, ...]
    actions: tuple[str, ...]
    observations: tuple[str, ...]
    discount: float
    initial: np.ndarray
    transition: np.ndarray
    observation: np.ndarray
    reward: np.ndarray

    def __post_init__(self) -> None:
        state_count = len(self.states)
        action_count = len(self.actions)
        observation_count = len(self.observations)
        expected = {
            "initial": (state_count,),
            "transition": (action_count, state_count, state_count),
            "observation": (action_count, state_count, observation_count),
            "reward": (action_count, state_count, state_count),
        }
        for field, shape in expected.items():
            value = np.asarray(getattr(self, field), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{field} shape {value.shape} != {shape}")
            value.setflags(write=False)
            object.__setattr__(self, field, value)
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")


@dataclass(frozen=True)
class Decision:
    action: int
    value: float
    q_values: tuple[float, ...]
    optimal_actions: tuple[int, ...]


def _names(tokens: list[str]) -> tuple[str, ...]:
    if len(tokens) == 1 and tokens[0].isdigit():
        return tuple(str(index) for index in range(int(tokens[0])))
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("state/action/observation names must be nonempty and unique")
    return tuple(tokens)


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _indices(names: tuple[str, ...], token: str) -> tuple[int, ...]:
    if token == "*":
        return tuple(range(len(names)))
    try:
        return (names.index(token),)
    except ValueError as exc:
        raise ValueError(f"unknown symbol {token!r}") from exc


def parse_pomdp_text(text: str, *, name: str = "model") -> POMDPModel:
    """Parse the prospectively declared matrix/wildcard subset of Cassandra POMDP."""
    lines = _clean_lines(text)
    headers: dict[str, object] = {}
    for line in lines:
        for key in ("discount", "values", "states", "actions", "observations"):
            prefix = f"{key}:"
            if line.startswith(prefix):
                body = line[len(prefix):].strip().split()
                headers[key] = body
                break
    required = {"discount", "values", "states", "actions", "observations"}
    if set(headers) != required:
        raise ValueError(f"missing or duplicate headers: {required - set(headers)}")
    if headers["values"] != ["reward"]:
        raise ValueError("only reward-valued POMDP files are supported")
    states = _names(headers["states"])  # type: ignore[arg-type]
    actions = _names(headers["actions"])  # type: ignore[arg-type]
    observations = _names(headers["observations"])  # type: ignore[arg-type]
    discount = float(headers["discount"][0])  # type: ignore[index]
    a_count, s_count, o_count = len(actions), len(states), len(observations)
    transition = np.zeros((a_count, s_count, s_count), dtype=np.float64)
    observation = np.zeros((a_count, s_count, o_count), dtype=np.float64)
    reward = np.zeros((a_count, s_count, s_count), dtype=np.float64)
    initial: np.ndarray | None = None

    index = 0
    while index < len(lines):
        line = lines[index]
        if any(
            line.startswith(f"{key}:")
            for key in ("discount", "values", "states", "actions", "observations")
        ):
            index += 1
            continue
        if line.startswith("start:"):
            inline = line[len("start:"):].strip()
            if inline:
                values = inline.split()
                index += 1
            else:
                index += 1
                values = lines[index].split()
                index += 1
            initial = np.asarray([float(value) for value in values], dtype=np.float64)
            if initial.shape != (s_count,):
                raise ValueError("start distribution has the wrong length")
            continue
        if line.startswith("T:"):
            parts = [part.strip() for part in line.split(":")]
            if len(parts) != 2:
                raise ValueError("V62 requires full transition matrices")
            action_indices = _indices(actions, parts[1])
            index += 1
            rows = []
            for _ in range(s_count):
                row = [float(value) for value in lines[index].split()]
                if len(row) != s_count:
                    raise ValueError("transition matrix row has wrong length")
                rows.append(row)
                index += 1
            matrix = np.asarray(rows, dtype=np.float64)
            for action in action_indices:
                transition[action] = matrix
            continue
        if line.startswith("O:"):
            parts = [part.strip() for part in line.split(":")]
            if len(parts) != 2:
                raise ValueError("V62 requires full observation matrices")
            action_indices = _indices(actions, parts[1])
            index += 1
            rows = []
            for _ in range(s_count):
                row = [float(value) for value in lines[index].split()]
                if len(row) != o_count:
                    raise ValueError("observation matrix row has wrong length")
                rows.append(row)
                index += 1
            matrix = np.asarray(rows, dtype=np.float64)
            for action in action_indices:
                observation[action] = matrix
            continue
        if line.startswith("R:"):
            parts = [part.strip() for part in line.split(":")]
            if len(parts) != 5:
                raise ValueError("reward entry must have action/state/successor/observation")
            tail = parts[4].split()
            if len(tail) != 2:
                raise ValueError("reward observation and scalar must share the final field")
            observation_token, value_text = tail
            if observation_token != "*":
                raise ValueError("V62 supports only observation-independent rewards")
            value = float(value_text)
            for action in _indices(actions, parts[1]):
                for state in _indices(states, parts[2]):
                    for successor in _indices(states, parts[3]):
                        reward[action, state, successor] = value
            index += 1
            continue
        raise ValueError(f"unsupported line: {line}")
    if initial is None:
        initial = np.full(s_count, 1.0 / s_count, dtype=np.float64)
    return POMDPModel(
        name=name,
        states=states,
        actions=actions,
        observations=observations,
        discount=discount,
        initial=initial,
        transition=transition,
        observation=observation,
        reward=reward,
    )


def parse_pomdp_file(path: str | Path) -> POMDPModel:
    source = Path(path)
    return parse_pomdp_text(source.read_text(), name=source.stem)


def terminal_mask(model: POMDPModel, *, atol: float = 1e-12) -> np.ndarray:
    mask = np.ones(len(model.states), dtype=bool)
    for state in range(len(model.states)):
        target = np.zeros(len(model.states), dtype=np.float64)
        target[state] = 1.0
        for action in range(len(model.actions)):
            mask[state] &= bool(np.allclose(model.transition[action, state], target, atol=atol, rtol=0.0))
    return mask


def validate_model(model: POMDPModel, *, atol: float = 1e-12) -> dict[str, bool]:
    return {
        "transition_normalized": bool(
            np.all(model.transition >= -atol)
            and np.allclose(model.transition.sum(axis=2), 1.0, atol=atol, rtol=0.0)
        ),
        "observation_normalized": bool(
            np.all(model.observation >= -atol)
            and np.allclose(model.observation.sum(axis=2), 1.0, atol=atol, rtol=0.0)
        ),
        "initial_normalized": bool(
            np.all(model.initial >= -atol)
            and np.isclose(model.initial.sum(), 1.0, atol=atol, rtol=0.0)
        ),
        "finite_reward_and_discount": bool(
            np.isfinite(model.reward).all() and np.isfinite(model.discount)
        ),
    }


def normalize(weights: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(weights, dtype=np.float64)
    mass = float(values.sum())
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("cannot normalize nonpositive or nonfinite mass")
    posterior = values / mass
    posterior[np.abs(posterior) < 1e-16] = 0.0
    return posterior, mass


def initial_observation_distribution(model: POMDPModel) -> np.ndarray:
    if not np.allclose(model.observation, model.observation[0], atol=1e-12, rtol=0.0):
        raise ValueError("V62 initial observation requires action-independent observation kernels")
    return model.initial @ model.observation[0]


def condition_initial(model: POMDPModel, observation: int) -> tuple[np.ndarray, float]:
    return normalize(model.initial * model.observation[0, :, observation])


def predict(model: POMDPModel, belief: np.ndarray, action: int) -> np.ndarray:
    return np.asarray(belief, dtype=np.float64) @ model.transition[action]


def observation_distribution(
    model: POMDPModel, belief: np.ndarray, action: int
) -> np.ndarray:
    return predict(model, belief, action) @ model.observation[action]


def update_belief(
    model: POMDPModel, belief: np.ndarray, action: int, observation: int
) -> tuple[np.ndarray, float]:
    predicted = predict(model, belief, action)
    return normalize(predicted * model.observation[action, :, observation])


def expected_reward(model: POMDPModel, belief: np.ndarray, action: int) -> float:
    weighted = model.transition[action] * model.reward[action]
    return float(np.asarray(belief, dtype=np.float64) @ weighted.sum(axis=1))


def belief_key(belief: np.ndarray) -> tuple[float, ...]:
    normalized, _ = normalize(np.asarray(belief, dtype=np.float64))
    return tuple(float(value) for value in np.round(normalized, 15))


class ExactPlanner:
    def __init__(self, model: POMDPModel, *, tie_tolerance: float = 1e-12):
        self.model = model
        self.tie_tolerance = tie_tolerance
        self.terminals = terminal_mask(model)
        self._cache: dict[tuple[tuple[float, ...], int], Decision] = {}

    def decision(self, belief: np.ndarray, horizon: int) -> Decision:
        key = (belief_key(belief), int(horizon))
        if key in self._cache:
            return self._cache[key]
        canonical = np.asarray(key[0], dtype=np.float64)
        if horizon <= 0 or bool(np.all(self.terminals[np.flatnonzero(canonical > 1e-14)])):
            result = Decision(0, 0.0, tuple(0.0 for _ in self.model.actions), tuple(range(len(self.model.actions))))
            self._cache[key] = result
            return result
        q_values = []
        for action in range(len(self.model.actions)):
            value = expected_reward(self.model, canonical, action)
            if horizon > 1:
                obs_probabilities = observation_distribution(self.model, canonical, action)
                continuation = 0.0
                for observation, probability in enumerate(obs_probabilities):
                    if probability <= 1e-15:
                        continue
                    posterior, observed_probability = update_belief(
                        self.model, canonical, action, observation
                    )
                    if abs(observed_probability - probability) > 1e-10:
                        raise RuntimeError("observation probability implementations disagree")
                    continuation += probability * self.decision(posterior, horizon - 1).value
                value += self.model.discount * continuation
            q_values.append(float(value))
        maximum = max(q_values)
        optimal = tuple(
            action
            for action, value in enumerate(q_values)
            if maximum - value <= self.tie_tolerance
        )
        result = Decision(optimal[0], maximum, tuple(q_values), optimal)
        self._cache[key] = result
        return result

    def initial_value(self, horizon: int) -> float:
        value = 0.0
        for observation, probability in enumerate(initial_observation_distribution(self.model)):
            if probability <= 1e-15:
                continue
            belief, observed_probability = condition_initial(self.model, observation)
            if abs(observed_probability - probability) > 1e-10:
                raise RuntimeError("initial observation probability implementations disagree")
            value += probability * self.decision(belief, horizon).value
        return float(value)

    def reachable_decisions(self, horizon: int) -> dict[tuple[tuple[float, ...], int], Decision]:
        frontier: list[tuple[np.ndarray, int]] = []
        for observation, probability in enumerate(initial_observation_distribution(self.model)):
            if probability > 1e-15:
                frontier.append((condition_initial(self.model, observation)[0], horizon))
        seen: set[tuple[tuple[float, ...], int]] = set()
        while frontier:
            belief, remaining = frontier.pop()
            key = (belief_key(belief), remaining)
            if key in seen:
                continue
            seen.add(key)
            decision = self.decision(belief, remaining)
            if remaining <= 1:
                continue
            for action in decision.optimal_actions:
                probabilities = observation_distribution(self.model, belief, action)
                for observation, probability in enumerate(probabilities):
                    if probability > 1e-15:
                        frontier.append(
                            (update_belief(self.model, belief, action, observation)[0], remaining - 1)
                        )
        return {key: self._cache[key] for key in seen}


def observation_only_belief(model: POMDPModel, observation: int) -> np.ndarray:
    prior = (~terminal_mask(model)).astype(np.float64)
    if prior.sum() == 0.0:
        prior = np.ones(len(model.states), dtype=np.float64)
    prior /= prior.sum()
    weights = prior * model.observation[0, :, observation]
    if weights.sum() <= 1e-15:
        return prior
    return normalize(weights)[0]


def public_policy_action_distribution(
    model: POMDPModel,
    planner: ExactPlanner,
    belief: np.ndarray,
    observation: int,
    horizon: int,
    policy: str,
) -> np.ndarray:
    action_count = len(model.actions)
    probabilities = np.zeros(action_count, dtype=np.float64)
    if policy == "uniform_random":
        probabilities[:] = 1.0 / action_count
        return probabilities
    decision_belief = np.asarray(belief, dtype=np.float64)
    if policy == "observation_only":
        decision_belief = observation_only_belief(model, observation)
    elif policy == "map_collapse":
        point = np.zeros(len(model.states), dtype=np.float64)
        point[int(np.argmax(decision_belief))] = 1.0
        decision_belief = point
    elif policy != "exact_history":
        raise ValueError(f"unsupported public policy {policy}")
    probabilities[planner.decision(decision_belief, horizon).action] = 1.0
    return probabilities


def public_policy_value(model: POMDPModel, horizon: int, policy: str) -> float:
    planner = ExactPlanner(model)
    terminals = terminal_mask(model)

    @lru_cache(maxsize=None)
    def recurse(belief_tuple: tuple[float, ...], observation: int, remaining: int) -> float:
        belief = np.asarray(belief_tuple, dtype=np.float64)
        support = np.flatnonzero(belief > 1e-14)
        if remaining <= 0 or bool(np.all(terminals[support])):
            return 0.0
        action_probabilities = public_policy_action_distribution(
            model, planner, belief, observation, remaining, policy
        )
        total = 0.0
        for action, action_probability in enumerate(action_probabilities):
            if action_probability <= 0.0:
                continue
            action_value = expected_reward(model, belief, action)
            if remaining > 1:
                continuation = 0.0
                for next_observation, probability in enumerate(
                    observation_distribution(model, belief, action)
                ):
                    if probability <= 1e-15:
                        continue
                    posterior = update_belief(
                        model, belief, action, next_observation
                    )[0]
                    continuation += probability * recurse(
                        belief_key(posterior), next_observation, remaining - 1
                    )
                action_value += model.discount * continuation
            total += action_probability * action_value
        return float(total)

    total = 0.0
    for observation, probability in enumerate(initial_observation_distribution(model)):
        if probability <= 1e-15:
            continue
        belief = condition_initial(model, observation)[0]
        total += probability * recurse(belief_key(belief), observation, horizon)
    return float(total)


def fully_observed_oracle_value(model: POMDPModel, horizon: int) -> float:
    terminals = terminal_mask(model)

    @lru_cache(maxsize=None)
    def recurse(state: int, remaining: int) -> float:
        if remaining <= 0 or terminals[state]:
            return 0.0
        action_values = []
        for action in range(len(model.actions)):
            value = 0.0
            for successor, probability in enumerate(model.transition[action, state]):
                if probability <= 0.0:
                    continue
                value += probability * (
                    model.reward[action, state, successor]
                    + model.discount * recurse(successor, remaining - 1)
                )
            action_values.append(value)
        return float(max(action_values))

    return float(sum(probability * recurse(state, horizon) for state, probability in enumerate(model.initial)))


def return_extrema(model: POMDPModel, horizon: int) -> tuple[float, float]:
    terminals = terminal_mask(model)

    @lru_cache(maxsize=None)
    def recurse(state: int, remaining: int) -> tuple[float, float]:
        if remaining <= 0 or terminals[state]:
            return 0.0, 0.0
        lows, highs = [], []
        for action in range(len(model.actions)):
            for successor, probability in enumerate(model.transition[action, state]):
                if probability <= 0.0:
                    continue
                child_low, child_high = recurse(successor, remaining - 1)
                reward = model.reward[action, state, successor]
                lows.append(reward + model.discount * child_low)
                highs.append(reward + model.discount * child_high)
        return min(lows), max(highs)

    reachable = np.flatnonzero(model.initial > 0.0)
    lows, highs = zip(*(recurse(int(state), horizon) for state in reachable), strict=True)
    return float(min(lows)), float(max(highs))


def bellman_residual(
    model: POMDPModel, planner: ExactPlanner, belief: np.ndarray, horizon: int
) -> float:
    decision = planner.decision(belief, horizon)
    if horizon <= 0:
        return abs(decision.value)
    recomposed = []
    for action in range(len(model.actions)):
        value = expected_reward(model, belief, action)
        if horizon > 1:
            for observation, probability in enumerate(
                observation_distribution(model, belief, action)
            ):
                if probability > 1e-15:
                    posterior = update_belief(model, belief, action, observation)[0]
                    value += (
                        model.discount
                        * probability
                        * planner.decision(posterior, horizon - 1).value
                    )
        recomposed.append(value)
    return float(max(abs(a - b) for a, b in zip(recomposed, decision.q_values, strict=True)))


def all_positive_observation_beliefs(
    model: POMDPModel, planner: ExactPlanner, horizon: int
) -> Iterable[tuple[np.ndarray, int]]:
    for (belief_tuple, remaining) in planner.reachable_decisions(horizon):
        yield np.asarray(belief_tuple, dtype=np.float64), remaining
