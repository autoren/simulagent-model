#!/usr/bin/env python3
"""Independent bounded policy-execution verification utilities for V67.

This module intentionally does not import the V62 parser, V64 filter, V66 planner,
or V66 evaluator.  It independently reconstructs the pinned family and executes
archived contingent policies under an exact joint posterior.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

import numpy as np


POBAX_MODEL_PATH = Path(
    "data/v63-external-unknown-dynamics/source-checkout/pobax/envs/classic/"
    "POMDP/4x3_nonterminating.POMDP"
)
POBAX_MODEL_SHA256 = "0fa62301931960d682b02961ffd38f4dd6b8e8835bc0203f4a12f849c267d6ff"
THETA_SUPPORT = (0.6, 0.95)
IDENTITY_NAMES = ("clockwise_failure", "counterclockwise_failure")
CANONICAL_ACTION_NAMES = ("n", "e", "s", "w")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def policy_tree_hash(policy: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(policy).encode()).hexdigest()


@dataclass(frozen=True)
class IndependentModel:
    states: tuple[str, ...]
    actions: tuple[str, ...]
    observations: tuple[str, ...]
    discount: float
    initial: np.ndarray
    transition: np.ndarray
    observation: np.ndarray
    reward: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            "initial": (len(self.states),),
            "transition": (len(self.actions), len(self.states), len(self.states)),
            "observation": (
                len(self.actions), len(self.states), len(self.observations)
            ),
            "reward": (len(self.actions), len(self.states), len(self.states)),
        }
        for name, shape in shapes.items():
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} != {shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} is not finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount outside [0, 1]")


@dataclass(frozen=True)
class IndependentFamily:
    model: IndependentModel
    theta: np.ndarray
    theta_weights: np.ndarray
    static_prior: np.ndarray
    transitions: np.ndarray
    permutations: np.ndarray
    canonical_action_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        count = len(self.theta)
        states = len(self.model.states)
        actions = len(self.model.actions)
        shapes = {
            "theta": (count,),
            "theta_weights": (count,),
            "static_prior": (2, count),
            "transitions": (2, count, actions, states, states),
            "permutations": (2, actions),
        }
        for name, shape in shapes.items():
            array = np.asarray(getattr(self, name))
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} != {shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} is not finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if abs(float(self.static_prior.sum()) - 1.0) > 1e-12:
            raise ValueError("static prior does not normalize")
        if np.min(self.transitions) < -1e-15:
            raise ValueError("negative family transition")
        if np.max(np.abs(self.transitions.sum(axis=-1) - 1.0)) > 1e-12:
            raise ValueError("family transition rows do not normalize")


def _clean_source_lines(text: str) -> list[str]:
    result = []
    for raw in text.splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            result.append(value)
    return result


def _symbol_list(tokens: Sequence[str]) -> tuple[str, ...]:
    if len(tokens) == 1 and tokens[0].isdigit():
        return tuple(str(index) for index in range(int(tokens[0])))
    values = tuple(tokens)
    if not values or len(values) != len(set(values)):
        raise ValueError("symbols must be nonempty and unique")
    return values


def _selected_indices(symbols: tuple[str, ...], token: str) -> tuple[int, ...]:
    if token == "*":
        return tuple(range(len(symbols)))
    if token not in symbols:
        raise ValueError(f"unknown symbol {token!r}")
    return (symbols.index(token),)


def independent_parse_pomdp_text(text: str) -> IndependentModel:
    """Parse only the matrix/wildcard Cassandra subset used by the pinned file."""
    lines = _clean_source_lines(text)
    headers: dict[str, list[str]] = {}
    names = ("discount", "values", "states", "actions", "observations")
    for line in lines:
        for name in names:
            marker = name + ":"
            if line.startswith(marker):
                if name in headers:
                    raise ValueError(f"duplicate header {name}")
                headers[name] = line[len(marker):].strip().split()
                break
    if set(headers) != set(names) or headers["values"] != ["reward"]:
        raise ValueError("missing header or non-reward POMDP")
    states = _symbol_list(headers["states"])
    actions = _symbol_list(headers["actions"])
    observations = _symbol_list(headers["observations"])
    discount = float(headers["discount"][0])
    state_count, action_count = len(states), len(actions)
    observation_count = len(observations)
    transition = np.zeros((action_count, state_count, state_count))
    observation = np.zeros((action_count, state_count, observation_count))
    reward = np.zeros((action_count, state_count, state_count))
    initial: np.ndarray | None = None
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        if any(line.startswith(name + ":") for name in names):
            cursor += 1
            continue
        if line.startswith("start:"):
            body = line[6:].strip()
            cursor += 1
            values = body.split() if body else lines[cursor].split()
            if not body:
                cursor += 1
            initial = np.asarray([float(value) for value in values])
            if initial.shape != (state_count,):
                raise ValueError("wrong initial distribution length")
            continue
        if line.startswith("T:") or line.startswith("O:"):
            kind, token = (part.strip() for part in line.split(":", 1))
            cursor += 1
            width = state_count if kind == "T" else observation_count
            rows = []
            for _ in range(state_count):
                row = [float(value) for value in lines[cursor].split()]
                if len(row) != width:
                    raise ValueError(f"wrong {kind} matrix width")
                rows.append(row)
                cursor += 1
            matrix = np.asarray(rows)
            for action in _selected_indices(actions, token):
                if kind == "T":
                    transition[action] = matrix
                else:
                    observation[action] = matrix
            continue
        if line.startswith("R:"):
            parts = [part.strip() for part in line.split(":")]
            if len(parts) != 5:
                raise ValueError("wrong reward entry")
            tail = parts[4].split()
            if len(tail) != 2 or tail[0] != "*":
                raise ValueError("only observation-independent rewards are supported")
            value = float(tail[1])
            for action in _selected_indices(actions, parts[1]):
                for state in _selected_indices(states, parts[2]):
                    for successor in _selected_indices(states, parts[3]):
                        reward[action, state, successor] = value
            cursor += 1
            continue
        raise ValueError(f"unsupported POMDP line: {line}")
    if initial is None:
        raise ValueError("explicit start distribution required")
    return IndependentModel(
        states=states,
        actions=actions,
        observations=observations,
        discount=discount,
        initial=initial,
        transition=transition,
        observation=observation,
        reward=reward,
    )


def independent_parse_pomdp_file(path: str | Path) -> IndependentModel:
    return independent_parse_pomdp_text(Path(path).read_text())


def validate_source_model(model: IndependentModel, atol: float = 1e-12) -> dict[str, bool]:
    return {
        "initial_normalized": bool(
            np.min(model.initial) >= -atol
            and abs(float(model.initial.sum()) - 1.0) <= atol
        ),
        "transitions_normalized": bool(
            np.min(model.transition) >= -atol
            and np.max(np.abs(model.transition.sum(axis=-1) - 1.0)) <= atol
        ),
        "observations_normalized": bool(
            np.min(model.observation) >= -atol
            and np.max(np.abs(model.observation.sum(axis=-1) - 1.0)) <= atol
        ),
        "finite_reward_and_discount": bool(
            np.all(np.isfinite(model.reward)) and math.isfinite(model.discount)
        ),
    }


def independent_scaled_beta_2_2_quadrature(
    nodes: int, low: float = THETA_SUPPORT[0], high: float = THETA_SUPPORT[1],
    *, mutation: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if nodes < 2 or not 0.0 <= low < high <= 1.0:
        raise ValueError("invalid quadrature")
    roots, weights = np.polynomial.legendre.leggauss(nodes)
    theta = low + (roots + 1.0) * (high - low) / 2.0
    unit = (theta - low) / (high - low)
    density = 6.0 * unit * (1.0 - unit) / (high - low)
    result = weights * (high - low) * 0.5 * density
    if mutation == "corrupt_quadrature_weight":
        result[0] *= 1.5
    result /= result.sum()
    return theta.astype(np.float64), result.astype(np.float64)


def construct_independent_family(
    model: IndependentModel, *, nodes: int = 257,
    mutation: str | None = None,
) -> IndependentFamily:
    if set(model.actions) != set(CANONICAL_ACTION_NAMES):
        raise ValueError("family requires n/s/e/w source actions")
    index = {name: model.actions.index(name) for name in CANONICAL_ACTION_NAMES}
    clockwise_names = {"n": "e", "e": "s", "s": "w", "w": "n"}
    counter_names = {"n": "w", "w": "s", "s": "e", "e": "n"}
    permutations = np.asarray([
        [model.actions.index(clockwise_names[name]) for name in model.actions],
        [model.actions.index(counter_names[name]) for name in model.actions],
    ], dtype=np.int64)
    if mutation == "swap_clockwise_and_counterclockwise_identity":
        permutations = permutations[::-1].copy()
    theta, theta_weights = independent_scaled_beta_2_2_quadrature(
        nodes, mutation=mutation
    )
    transitions = np.empty(
        (2, nodes, len(model.actions), len(model.states), len(model.states)),
        dtype=np.float64,
    )
    for identity in range(2):
        for node, value in enumerate(theta):
            for action in range(len(model.actions)):
                transitions[identity, node, action] = (
                    value * model.transition[action]
                    + (1.0 - value) * model.transition[permutations[identity, action]]
                )
    static_prior = np.stack((0.5 * theta_weights, 0.5 * theta_weights))
    return IndependentFamily(
        model=model,
        theta=theta,
        theta_weights=theta_weights,
        static_prior=static_prior,
        transitions=transitions,
        permutations=permutations,
        canonical_action_ids=tuple(index[name] for name in CANONICAL_ACTION_NAMES),
    )


def load_pinned_family(project_root: str | Path, *, nodes: int = 257) -> IndependentFamily:
    path = Path(project_root) / POBAX_MODEL_PATH
    if file_sha256(path) != POBAX_MODEL_SHA256:
        raise RuntimeError("pinned POBAX model hash mismatch")
    model = independent_parse_pomdp_file(path)
    if not all(validate_source_model(model).values()):
        raise RuntimeError("pinned source model failed normalization")
    return construct_independent_family(model, nodes=nodes)


def normalize_belief(value: np.ndarray) -> tuple[np.ndarray, float]:
    mass = float(np.asarray(value).sum())
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("nonpositive or nonfinite belief mass")
    result = np.asarray(value, dtype=np.float64) / mass
    result[np.abs(result) < 1e-16] = 0.0
    return result, mass


def scalar_step(
    family: IndependentFamily, belief: np.ndarray, action: int,
    observation: int, *, mutation: str | None = None,
) -> tuple[np.ndarray, float, float]:
    expected_shape = (2, len(family.theta), len(family.model.states))
    if belief.shape != expected_shape or abs(float(belief.sum()) - 1.0) > 1e-10:
        raise ValueError("invalid joint belief")
    successor_mass = np.zeros_like(belief)
    reward_numerator = 0.0
    static = belief.sum(axis=2)
    mean_transition = None
    if mutation == "replace_persistent_static_model_with_per_step_mean_transition":
        mean_transition = np.zeros_like(family.model.transition[action])
        for identity in range(2):
            for node in range(len(family.theta)):
                mean_transition += static[identity, node] * family.transitions[
                    identity, node, action
                ]
    for identity in range(2):
        used_identity = (
            1 - identity
            if mutation == "swap_clockwise_and_counterclockwise_identity"
            else identity
        )
        for node in range(len(family.theta)):
            transition = (
                mean_transition
                if mean_transition is not None
                else family.transitions[used_identity, node, action]
            )
            for source in range(len(family.model.states)):
                source_mass = float(belief[identity, node, source])
                if source_mass == 0.0:
                    continue
                for successor in range(len(family.model.states)):
                    joint = (
                        source_mass * float(transition[source, successor])
                        * float(family.model.observation[action, successor, observation])
                    )
                    successor_mass[identity, node, successor] += joint
                    reward_source = (
                        successor
                        if mutation == "use_successor_instead_of_current_state_reward_index"
                        else source
                    )
                    reward = float(family.model.reward[action, reward_source, successor])
                    if mutation == "corrupt_source_reward":
                        reward += 0.01
                    reward_numerator += joint * reward
    posterior, probability = normalize_belief(successor_mass)
    return posterior, probability, reward_numerator / probability


def vector_step(
    family: IndependentFamily, belief: np.ndarray, action: int,
    observation: int,
) -> tuple[np.ndarray, float, float]:
    transition = family.transitions[:, :, action]
    likelihood = family.model.observation[action, :, observation]
    successor = np.einsum("ins,inst->int", belief, transition) * likelihood[None, None, :]
    probability = float(successor.sum())
    posterior, _ = normalize_belief(successor)
    reward_joint = np.einsum(
        "ins,inst,st,t->", belief, transition,
        family.model.reward[action], likelihood,
    )
    return posterior, probability, float(reward_joint) / probability


def _observation_id(model: IndependentModel, value: int | str) -> int:
    if isinstance(value, int):
        if value not in range(len(model.observations)):
            raise ValueError("invalid observation index")
        return value
    if value not in model.observations:
        raise ValueError(f"unknown observation {value!r}")
    return model.observations.index(value)


def _action_id(model: IndependentModel, value: int | str) -> int:
    if isinstance(value, int):
        if value not in range(len(model.actions)):
            raise ValueError("invalid action index")
        return value
    if value not in model.actions:
        raise ValueError(f"unknown action {value!r}")
    return model.actions.index(value)


def condition_public_history(
    family: IndependentFamily, record: dict[str, Any],
    *, mutation: str | None = None,
) -> tuple[np.ndarray, float]:
    forbidden = {"identity", "theta", "state", "truth", "audit"}
    if forbidden & set(record):
        raise ValueError("truth-like field in public record")
    actions = list(record["actions"])
    observations = list(record["observations"])
    if len(actions) != len(observations) or len(actions) != int(record["prefix_length"]):
        raise ValueError("history length mismatch")
    base = family.static_prior[:, :, None] * family.model.initial[None, None, :]
    if mutation != "omit_reset_initial_observation":
        if not np.array_equal(
            family.model.observation,
            np.broadcast_to(family.model.observation[0], family.model.observation.shape),
        ):
            raise ValueError("reset observation kernel depends on action")
        initial_observation = _observation_id(family.model, record["initial_observation"])
        base = base * family.model.observation[
            0, :, initial_observation
        ][None, None, :]
    belief, evidence = normalize_belief(base)
    for action_value, observation_value in zip(actions, observations, strict=True):
        action = _action_id(family.model, action_value)
        observation = _observation_id(family.model, observation_value)
        belief, probability, _ = scalar_step(family, belief, action, observation)
        evidence *= probability
    return belief, evidence


def _policy_action(
    family: IndependentFamily, node: dict[str, Any], *, mutation: str | None = None,
) -> int:
    if mutation == "replace_archived_selected_action_with_first_optimal_action":
        action = int(node["optimal_actions"][0])
    else:
        action = int(node["selected_action"])
    name = str(node["selected_action_name"])
    if action not in family.canonical_action_ids:
        raise ValueError("archived action is outside canonical action set")
    if (
        mutation != "replace_archived_selected_action_with_first_optimal_action"
        and family.model.actions[action] != name
    ):
        raise ValueError("archived action id/name mismatch")
    return action


def _validate_policy_node(
    node: dict[str, Any], expected_horizon: int, *, mutation: str | None = None,
) -> None:
    if mutation != "accept_wrong_policy_horizon" and int(node["horizon"]) != expected_horizon:
        raise ValueError("policy horizon mismatch")
    if bool(node["terminal"]):
        raise ValueError("unexpected archived terminal node")
    if expected_horizon < 1:
        raise ValueError("invalid expected horizon")


def execute_policy_scalar(
    family: IndependentFamily, root_belief: np.ndarray,
    policy: dict[str, Any], horizon: int = 3, *, mutation: str | None = None,
) -> dict[str, Any]:
    counters = {
        "reachable_nodes": 0,
        "node_invariants": 0,
        "node_invariant_passes": 0,
        "positive_branches": 0,
        "total_branches": 0,
        "total_branch_passes": 0,
    }

    def visit(belief: np.ndarray, node: dict[str, Any], remaining: int, depth: int) -> float:
        counters["reachable_nodes"] += 1
        counters["node_invariants"] += 1
        _validate_policy_node(node, remaining, mutation=mutation)
        action = _policy_action(family, node, mutation=mutation)
        counters["node_invariant_passes"] += 1
        outcomes = []
        for observation in range(len(family.model.observations)):
            try:
                child_belief, probability, reward = scalar_step(
                    family, belief, action, observation, mutation=mutation
                )
            except ValueError:
                continue
            if probability > 1e-15:
                outcomes.append((observation, child_belief, probability, reward))
        branches = node.get("branches", {})
        positive_keys = {str(row[0]) for row in outcomes}
        expected_keys = positive_keys if remaining > 1 else set()
        counters["positive_branches"] += len(positive_keys)
        counters["total_branches"] += 1
        if set(branches) != expected_keys:
            raise ValueError("policy observation branches are not exact and total")
        counters["total_branch_passes"] += 1
        value = 0.0
        for observation, child_belief, probability, reward in outcomes:
            immediate = reward
            if mutation == "omit_discount":
                discount = 1.0
            elif mutation == "discount_by_remaining_horizon":
                discount = family.model.discount ** (remaining - 1)
            else:
                discount = family.model.discount ** depth
            continuation = 0.0
            if remaining > 1:
                continuation = visit(
                    child_belief, branches[str(observation)], remaining - 1, depth + 1
                )
            value += probability * (discount * immediate + continuation)
        return value

    value = visit(np.asarray(root_belief), policy, horizon, 0)
    return {"value": float(value), **counters}


def compile_policy_dtmc(
    family: IndependentFamily, root_belief: np.ndarray,
    policy: dict[str, Any], horizon: int = 3, *, mutation: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    states: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    checks = {
        "node_invariants": 0,
        "node_invariant_passes": 0,
        "branch_totality_checks": 0,
        "branch_totality_passes": 0,
        "transition_normalization_checks": 0,
        "transition_normalization_passes": 0,
        "nonterminal_deadlocks": 0,
    }

    def allocate(kind: str, depth: int, path: tuple[int, ...]) -> int:
        identifier = len(states)
        states.append({"id": identifier, "kind": kind, "depth": depth, "path": list(path)})
        return identifier

    def visit(
        belief: np.ndarray, node: dict[str, Any], remaining: int,
        depth: int, path: tuple[int, ...], state_id: int,
    ) -> None:
        checks["node_invariants"] += 1
        _validate_policy_node(node, remaining, mutation=mutation)
        action = _policy_action(family, node, mutation=mutation)
        states[state_id]["action"] = action
        states[state_id]["action_name"] = family.model.actions[action]
        states[state_id]["remaining_horizon"] = remaining
        checks["node_invariant_passes"] += 1
        outcomes = []
        for observation in range(len(family.model.observations)):
            try:
                child_belief, probability, reward = vector_step(
                    family, belief, action, observation
                )
            except ValueError:
                continue
            if probability > 1e-15:
                outcomes.append((observation, child_belief, probability, reward))
        branches = node.get("branches", {})
        expected_keys = {str(row[0]) for row in outcomes} if remaining > 1 else set()
        checks["branch_totality_checks"] += 1
        if set(branches) != expected_keys:
            raise ValueError("compiled policy observation branches are not exact and total")
        checks["branch_totality_passes"] += 1
        archived = np.asarray(node.get("observation_probabilities", []), dtype=np.float64)
        probability_override = None
        if mutation == "use_archived_SMC2_branch_probabilities_instead_of_exact_probabilities":
            if archived.shape != (len(family.model.observations),):
                raise ValueError("missing archived probabilities")
            probability_override = archived
        local_edges = []
        first_target: int | None = None
        for outcome_index, (observation, child_belief, probability, reward) in enumerate(outcomes):
            if remaining > 1:
                target = allocate("policy", depth + 1, path + (observation,))
                visit(
                    child_belief, branches[str(observation)], remaining - 1,
                    depth + 1, path + (observation,), target,
                )
            else:
                target = allocate("terminal", depth + 1, path + (observation,))
            if first_target is None:
                first_target = target
            if mutation == "merge_distinct_observation_branches" and outcome_index > 0:
                target = int(first_target)
            edge_probability = (
                float(probability_override[observation])
                if probability_override is not None else probability
            )
            local_edges.append({
                "source": state_id,
                "target": target,
                "probability": edge_probability,
                "reward": (family.model.discount ** depth) * reward,
                "observation": observation,
            })
        if mutation == "drop_positive_observation_branch" and local_edges:
            local_edges.pop(0)
        transitions.extend(local_edges)
        checks["transition_normalization_checks"] += 1
        mass = sum(float(edge["probability"]) for edge in local_edges)
        if abs(mass - 1.0) <= 1e-10:
            checks["transition_normalization_passes"] += 1
        else:
            checks["nonterminal_deadlocks"] += int(mass <= 1e-15)

    root = allocate("policy", 0, ())
    visit(np.asarray(root_belief), policy, horizon, 0, (), root)
    done = allocate("done", horizon + 1, ())
    if mutation != "omit_done_transition":
        for state in states:
            if state["kind"] == "terminal":
                transitions.append({
                    "source": state["id"], "target": done,
                    "probability": 1.0, "reward": 0.0, "observation": None,
                })
    else:
        checks["nonterminal_deadlocks"] += sum(
            state["kind"] == "terminal" for state in states
        )
    transitions.append({
        "source": done, "target": done, "probability": 1.0,
        "reward": 0.0, "observation": None,
    })
    model = {
        "states": states,
        "transitions": transitions,
        "root_state": root,
        "done_state": done,
    }
    return model, checks


def dtmc_statistics(model: dict[str, Any]) -> dict[str, float]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for edge in model["transitions"]:
        grouped.setdefault(int(edge["source"]), []).append(edge)
    done = int(model["done_state"])
    memo: dict[int, tuple[float, float]] = {}

    def visit(state: int) -> tuple[float, float]:
        if state == done:
            return 1.0, 0.0
        if state in memo:
            return memo[state]
        edges = grouped.get(state, [])
        if not edges:
            return 0.0, 0.0
        termination = 0.0
        value = 0.0
        for edge in edges:
            child_termination, child_value = visit(int(edge["target"]))
            probability = float(edge["probability"])
            termination += probability * child_termination
            value += probability * (float(edge["reward"]) + child_value)
        memo[state] = termination, value
        return memo[state]

    termination, value = visit(int(model["root_state"]))
    return {"termination_probability": termination, "expected_return": value}


def write_explicit_dtmc(model: dict[str, Any], directory: str | Path) -> None:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    edges = sorted(
        model["transitions"],
        key=lambda edge: (int(edge["source"]), int(edge["target"])),
    )
    (output / "model.tra").write_text("dtmc\n" + "".join(
        f'{edge["source"]} {edge["target"]} {float(edge["probability"]):.17g}\n'
        for edge in edges
    ))
    labels = ["#DECLARATION", "init done", "#END"]
    labels.append(f'{model["root_state"]} init')
    labels.append(f'{model["done_state"]} done')
    (output / "model.lab").write_text("\n".join(labels) + "\n")
    (output / "model.rew").write_text("".join(
        f'{edge["source"]} {edge["target"]} {float(edge["reward"]):.17g}\n'
        for edge in edges
    ))


def run_storm_property(directory: str | Path, property_text: str) -> float:
    path = Path(directory)
    completed = subprocess.run([
        "storm", "--explicit", str(path / "model.tra"), str(path / "model.lab"),
        "--transrew", str(path / "model.rew"), "--prop", property_text,
        "--precision", "1e-14",
    ], check=True, capture_output=True, text=True)
    matches = re.findall(
        r"Result \((?:for )?initial states?\):\s*([-+0-9.eE/]+)",
        completed.stdout,
    )
    if len(matches) != 1:
        raise RuntimeError("unable to parse exactly one Storm result")
    token = matches[0]
    return float(Fraction(token)) if "/" in token else float(token)


def run_storm_properties(directory: str | Path) -> dict[str, float]:
    return {
        "termination_probability": run_storm_property(directory, 'P=? [F "done"]'),
        "expected_return": run_storm_property(directory, 'R=? [F "done"]'),
    }


def storm_version() -> str:
    completed = subprocess.run(
        ["storm", "--version"], check=True, capture_output=True, text=True
    )
    match = re.search(r"Storm\s+(\d+\.\d+\.\d+)", completed.stdout)
    if not match:
        raise RuntimeError("unable to parse Storm version")
    return match.group(1)


def finite_dtmc(model: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(edge[key]))
        for edge in model["transitions"]
        for key in ("probability", "reward")
    )
