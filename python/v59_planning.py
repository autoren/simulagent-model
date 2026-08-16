"""Budgeted root-sampled observation-contingent tree search for V59."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable

from v22_relational import canonical_json
from v42_stateful import world_signature
from v53_smc2 import continuous_unit_transition, instantiate_program
from v55_planning import candidate_actions


FORBIDDEN_SEARCH_KEYS = frozenset({
    "truth", "target_program_index", "target_program_key",
    "target_program_ordinal", "target_theta", "query_configuration_key",
    "query_world", "query_queue", "future_observation", "realized_outcome",
})


def assert_search_payload_is_public(value):
    if isinstance(value, dict):
        overlap = set(value) & FORBIDDEN_SEARCH_KEYS
        if overlap:
            raise PermissionError(
                "V59 search payload contains forbidden fields: "
                + ", ".join(sorted(overlap))
            )
        for child in value.values():
            assert_search_payload_is_public(child)
    elif isinstance(value, list):
        for child in value:
            assert_search_payload_is_public(child)


def forbidden_latent_conditioned_rollout(*_args, **_kwargs):
    raise PermissionError(
        "V59 rollout actions may not read hidden program, parameter, world, or queue"
    )


@dataclass
class ActionStats:
    visits: int = 0
    total_return: float = 0.0
    children: dict[str, "HistoryNode"] = field(default_factory=dict)

    @property
    def mean_return(self) -> float:
        return self.total_return / self.visits if self.visits else 0.0


@dataclass
class HistoryNode:
    visits: int = 0
    actions: dict[str, ActionStats] = field(default_factory=dict)


@dataclass
class SearchResult:
    root: HistoryNode
    budget: int
    simulations_run: int
    selected_action: dict
    selected_action_key: str
    root_action_rows: list[dict]
    root_sample_counts: dict[str, int]
    tree_nodes: int
    branching_action_nodes: int
    visited_action_nodes: int
    tree_sha256: str
    merge_observations: bool
    seed: int


def _seed(*parts) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def _weighted_sample(rows: list[dict], draw: float) -> dict:
    total = sum(float(row["weight"]) for row in rows)
    if total <= 0:
        raise RuntimeError("V59 root belief has zero mass")
    threshold = draw * total
    cumulative = 0.0
    for row in rows:
        cumulative += float(row["weight"])
        if threshold < cumulative:
            return copy.deepcopy(row)
    return copy.deepcopy(rows[-1])


def sample_root_counts(
    rows: list[dict], samples: int, seed: int,
    label_fn: Callable[[dict], str],
) -> dict[str, int]:
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for _ in range(samples):
        label = label_fn(_weighted_sample(rows, rng.random()))
        counts[label] = counts.get(label, 0) + 1
    return counts


def _initialize_actions(node: HistoryNode, action_rows: list[dict]) -> None:
    if not node.actions:
        node.actions = {row["key"]: ActionStats() for row in action_rows}


def _tree_action(
    node: HistoryNode, action_rows: list[dict], exploration_constant: float
) -> dict:
    _initialize_actions(node, action_rows)
    by_key = {row["key"]: row for row in action_rows}
    unvisited = sorted(
        key for key, stats in node.actions.items() if stats.visits == 0
    )
    if unvisited:
        return by_key[unvisited[0]]
    log_visits = math.log(max(1, node.visits))
    scored = []
    for key, stats in node.actions.items():
        bonus = exploration_constant * math.sqrt(log_visits / stats.visits)
        scored.append((stats.mean_return + bonus, key))
    maximum = max(score for score, _ in scored)
    selected_key = min(key for score, key in scored if score == maximum)
    return by_key[selected_key]


def _deployment_action(node: HistoryNode | None, action_rows: list[dict], history, fallback_seed: int):
    by_key = {row["key"]: row for row in action_rows}
    if node is not None:
        visited = [
            (key, stats) for key, stats in node.actions.items()
            if stats.visits > 0
        ]
        if visited:
            key, _ = min(
                visited,
                key=lambda item: (
                    -item[1].visits, -item[1].mean_return, item[0]
                ),
            )
            return by_key[key]
    token = canonical_json(history)
    index = _seed("v59-fallback", fallback_seed, token) % len(action_rows)
    return sorted(action_rows, key=lambda row: row["key"])[index]


def _rollout_action(action_rows: list[dict], history, rng: random.Random) -> dict:
    del history
    return action_rows[rng.randrange(len(action_rows))]


def _route_observation(observation: str, merge_observations: bool) -> str:
    return "*" if merge_observations else observation


def _rollout(
    state: dict,
    remaining: int,
    tick: int,
    history: list[dict],
    action_rows: list[dict],
    rng: random.Random,
    transition_fn: Callable[[dict, dict, int, float], tuple[dict, str]],
    terminal_fn: Callable[[dict], float],
    action_cost_fn: Callable[[dict], float],
    omit_action_cost: bool,
) -> float:
    if remaining == 0:
        return float(terminal_fn(state))
    row = _rollout_action(action_rows, history, rng)
    next_state, observation = transition_fn(
        state, row["action"], tick, rng.random()
    )
    cost = 0.0 if omit_action_cost else float(action_cost_fn(row["action"]))
    return -cost + _rollout(
        next_state, remaining - 1, tick + 1,
        [*history, {"action_key": row["key"], "observation": observation}],
        action_rows, rng, transition_fn, terminal_fn, action_cost_fn,
        omit_action_cost,
    )


def _simulate(
    node: HistoryNode,
    state: dict,
    remaining: int,
    tick: int,
    history: list[dict],
    action_rows: list[dict],
    rng: random.Random,
    transition_fn: Callable[[dict, dict, int, float], tuple[dict, str]],
    terminal_fn: Callable[[dict], float],
    action_cost_fn: Callable[[dict], float],
    exploration_constant: float,
    merge_observations: bool,
    omit_action_cost: bool,
) -> float:
    if remaining == 0:
        return float(terminal_fn(state))
    node.visits += 1
    row = _tree_action(node, action_rows, exploration_constant)
    stats = node.actions[row["key"]]
    next_state, observation = transition_fn(
        state, row["action"], tick, rng.random()
    )
    routed = _route_observation(observation, merge_observations)
    next_history = [
        *history, {"action_key": row["key"], "observation": observation}
    ]
    if routed not in stats.children:
        stats.children[routed] = HistoryNode()
        continuation = _rollout(
            next_state, remaining - 1, tick + 1, next_history,
            action_rows, rng, transition_fn, terminal_fn, action_cost_fn,
            omit_action_cost,
        )
    else:
        continuation = _simulate(
            stats.children[routed], next_state, remaining - 1, tick + 1,
            next_history, action_rows, rng, transition_fn, terminal_fn,
            action_cost_fn, exploration_constant, merge_observations,
            omit_action_cost,
        )
    cost = 0.0 if omit_action_cost else float(action_cost_fn(row["action"]))
    value = -cost + continuation
    stats.visits += 1
    stats.total_return += value
    return value


def tree_payload(node: HistoryNode) -> dict:
    return {
        "visits": node.visits,
        "actions": {
            key: {
                "visits": stats.visits,
                "total_return": stats.total_return,
                "mean_return": stats.mean_return,
                "children": {
                    observation: tree_payload(child)
                    for observation, child in sorted(stats.children.items())
                },
            }
            for key, stats in sorted(node.actions.items())
        },
    }


def tree_census(node: HistoryNode) -> tuple[int, int, int]:
    nodes, branching, visited_actions = 1, 0, 0
    for stats in node.actions.values():
        if stats.visits:
            visited_actions += 1
        if len(stats.children) > 1:
            branching += 1
        for child in stats.children.values():
            child_nodes, child_branching, child_actions = tree_census(child)
            nodes += child_nodes
            branching += child_branching
            visited_actions += child_actions
    return nodes, branching, visited_actions


def run_root_sampled_uct(
    root_rows: list[dict],
    action_rows: list[dict],
    horizon: int,
    tick: int,
    budget: int,
    seed: int,
    transition_fn: Callable[[dict, dict, int, float], tuple[dict, str]],
    terminal_fn: Callable[[dict], float],
    action_cost_fn: Callable[[dict], float],
    static_label_fn: Callable[[dict], str],
    exploration_constant: float = math.sqrt(2.0),
    merge_observations: bool = False,
    omit_action_cost: bool = False,
    simulation_limit_override: int | None = None,
) -> SearchResult:
    if horizon <= 0 or budget <= 0:
        raise ValueError("V59 search requires positive horizon and budget")
    if not root_rows or abs(sum(float(row["weight"]) for row in root_rows) - 1.0) > 1e-8:
        raise ValueError("V59 root rows must be a normalized nonempty belief")
    if len({row["key"] for row in action_rows}) != len(action_rows):
        raise ValueError("V59 action keys must be unique")
    root = HistoryNode()
    rng = random.Random(seed)
    root_counts: dict[str, int] = {}
    simulations = budget if simulation_limit_override is None else simulation_limit_override
    for _ in range(simulations):
        state = _weighted_sample(root_rows, rng.random())
        label = static_label_fn(state)
        root_counts[label] = root_counts.get(label, 0) + 1
        _simulate(
            root, state, horizon, tick, [], action_rows, rng,
            transition_fn, terminal_fn, action_cost_fn,
            exploration_constant, merge_observations, omit_action_cost,
        )
    selected = _deployment_action(root, action_rows, [], seed)
    payload = tree_payload(root)
    nodes, branching, visited_actions = tree_census(root)
    return SearchResult(
        root=root,
        budget=budget,
        simulations_run=simulations,
        selected_action=copy.deepcopy(selected["action"]),
        selected_action_key=selected["key"],
        root_action_rows=[
            {
                "action_key": key,
                "visits": stats.visits,
                "mean_return": stats.mean_return,
            }
            for key, stats in sorted(root.actions.items())
        ],
        root_sample_counts=dict(sorted(root_counts.items())),
        tree_nodes=nodes,
        branching_action_nodes=branching,
        visited_action_nodes=visited_actions,
        tree_sha256=hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        merge_observations=merge_observations,
        seed=seed,
    )


def static_atom_label(atom: dict) -> str:
    return canonical_json({
        "program_index": atom["program_index"],
        "node_index": atom["node_index"],
        "theta": float(atom["theta"]).hex(),
    })


def domain_transition_factory(registry: list[dict], entity_rows: list[dict]):
    program_cache: dict[tuple, dict] = {}

    def transition(state: dict, action: dict, tick: int, draw: float):
        key = (
            state["program_index"], state["node_index"],
            float(state["theta"]).hex(),
        )
        if key not in program_cache:
            program_cache[key] = instantiate_program(
                registry[state["program_index"]]["template"], state["theta"]
            )
        branches = continuous_unit_transition(
            program_cache[key], entity_rows, state["world"], state["queue"],
            action, tick,
        )
        ordered = sorted(branches.items())
        cumulative = 0.0
        selected = ordered[-1][1]
        for _, branch in ordered:
            cumulative += float(branch["mass"])
            if draw < cumulative:
                selected = branch
                break
        next_state = {
            "program_index": state["program_index"],
            "node_index": state["node_index"],
            "theta": state["theta"],
            "configuration_key": canonical_json({
                "world": sorted(selected["world"].items()),
                "queue": sorted(selected["queue"], key=canonical_json),
            }),
            "world": dict(selected["world"]),
            "queue": list(selected["queue"]),
            "weight": state.get("weight", 1.0),
        }
        return next_state, world_signature(next_state["world"])

    return transition


def plan_domain(
    atoms: list[dict], registry: list[dict], entity_rows: list[dict], goal: dict,
    horizon: int, tick: int, budget: int, seed: int, config: dict,
    merge_observations: bool = False,
) -> SearchResult:
    action_rows = candidate_actions(entity_rows)
    transition = domain_transition_factory(registry, entity_rows)
    return run_root_sampled_uct(
        atoms, action_rows, horizon, tick, budget, seed, transition,
        lambda state: float(state["world"][goal["atom"]] is goal["value"]),
        lambda action: float(config["planningModel"]["actionCost"][action["id"]]),
        static_atom_label,
        exploration_constant=config["candidateSearch"]["explorationConstant"],
        merge_observations=merge_observations,
    )


def _policy_episode(
    search: SearchResult,
    root_rows: list[dict],
    action_rows: list[dict],
    horizon: int,
    tick: int,
    root_draw: float,
    transition_draws: list[float],
    transition_fn: Callable[[dict, dict, int, float], tuple[dict, str]],
    terminal_fn: Callable[[dict], float],
    action_cost_fn: Callable[[dict], float],
    fallback_seed: int,
) -> float:
    state = _weighted_sample(root_rows, root_draw)
    node: HistoryNode | None = search.root
    history: list[dict] = []
    value = 0.0
    for depth in range(horizon):
        row = _deployment_action(node, action_rows, history, fallback_seed)
        value -= float(action_cost_fn(row["action"]))
        next_state, observation = transition_fn(
            state, row["action"], tick + depth, transition_draws[depth]
        )
        routed = _route_observation(observation, search.merge_observations)
        next_node = None
        if node is not None and row["key"] in node.actions:
            next_node = node.actions[row["key"]].children.get(routed)
        history.append({"action_key": row["key"], "observation": observation})
        state, node = next_state, next_node
    return value + float(terminal_fn(state))


def evaluate_policy_pair(
    candidate: SearchResult,
    control: SearchResult,
    root_rows: list[dict],
    action_rows: list[dict],
    horizon: int,
    tick: int,
    episodes: int,
    seed: int,
    transition_fn: Callable[[dict, dict, int, float], tuple[dict, str]],
    terminal_fn: Callable[[dict], float],
    action_cost_fn: Callable[[dict], float],
    fallback_seed: int,
) -> dict:
    candidate_returns, control_returns, differences = [], [], []
    for episode in range(episodes):
        rng = random.Random(_seed("v59-evaluation", seed, episode))
        root_draw = rng.random()
        transition_draws = [rng.random() for _ in range(horizon)]
        candidate_value = _policy_episode(
            candidate, root_rows, action_rows, horizon, tick, root_draw,
            transition_draws, transition_fn, terminal_fn, action_cost_fn,
            fallback_seed,
        )
        control_value = _policy_episode(
            control, root_rows, action_rows, horizon, tick, root_draw,
            transition_draws, transition_fn, terminal_fn, action_cost_fn,
            fallback_seed,
        )
        candidate_returns.append(candidate_value)
        control_returns.append(control_value)
        differences.append(candidate_value - control_value)
    mean_candidate = sum(candidate_returns) / episodes
    mean_control = sum(control_returns) / episodes
    mean_difference = sum(differences) / episodes
    variance = (
        sum((value - mean_difference) ** 2 for value in differences)
        / (episodes - 1)
        if episodes > 1 else 0.0
    )
    standard_error = math.sqrt(variance / episodes)
    return {
        "episodes": episodes,
        "candidate_mean_return": mean_candidate,
        "control_mean_return": mean_control,
        "paired_mean_difference": mean_difference,
        "paired_standard_error": standard_error,
        "paired_lower_95": mean_difference - 1.96 * standard_error,
        "paired_upper_95": mean_difference + 1.96 * standard_error,
        "finite": all(math.isfinite(value) for value in [
            *candidate_returns, *control_returns,
        ]),
    }


def evaluate_domain_policy_pair(
    candidate: SearchResult, control: SearchResult, atoms: list[dict],
    registry: list[dict], entity_rows: list[dict], goal: dict,
    horizon: int, tick: int, episodes: int, seed: int, config: dict,
) -> dict:
    transition = domain_transition_factory(registry, entity_rows)
    action_rows = candidate_actions(entity_rows)
    return evaluate_policy_pair(
        candidate, control, atoms, action_rows, horizon, tick, episodes, seed,
        transition,
        lambda state: float(state["world"][goal["atom"]] is goal["value"]),
        lambda action: float(config["planningModel"]["actionCost"][action["id"]]),
        fallback_seed=config["evaluation"]["evaluationSeed"],
    )

