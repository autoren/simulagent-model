"""SMC² belief conversion and semantics-preserving fast root sampling for V60."""
from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import random
from collections import defaultdict

from scipy.stats import wasserstein_distance

import v59_planning as v59
from v22_relational import canonical_json
from v53_smc2 import normalize_float_map, normalize_float_sequence, theta_bin
from v55_planning import candidate_actions


def _cdf(rows: list[dict]) -> tuple[list[float], float]:
    cumulative, total = [], 0.0
    for row in rows:
        total += float(row["weight"])
        cumulative.append(total)
    if total <= 0:
        raise RuntimeError("V60 root belief has zero mass")
    return cumulative, total


def _cdf_sample(rows: list[dict], cumulative: list[float], total: float, draw: float) -> dict:
    index = bisect.bisect_right(cumulative, draw * total)
    return copy.deepcopy(rows[min(index, len(rows) - 1)])


def run_root_sampled_uct_fast(
    root_rows: list[dict], action_rows: list[dict], horizon: int, tick: int,
    budget: int, seed: int, transition_fn, terminal_fn, action_cost_fn,
    static_label_fn, exploration_constant: float = math.sqrt(2.0),
    merge_observations: bool = False,
) -> v59.SearchResult:
    """V59 search semantics with O(log n), rather than O(n), root draws."""
    if horizon <= 0 or budget <= 0:
        raise ValueError("V60 search requires positive horizon and budget")
    if not root_rows or abs(sum(float(row["weight"]) for row in root_rows) - 1.0) > 1e-8:
        raise ValueError("V60 root rows must be a normalized nonempty belief")
    if len({row["key"] for row in action_rows}) != len(action_rows):
        raise ValueError("V60 action keys must be unique")
    cumulative, total = _cdf(root_rows)
    root = v59.HistoryNode()
    rng = random.Random(seed)
    root_counts: dict[str, int] = {}
    for _ in range(budget):
        state = _cdf_sample(root_rows, cumulative, total, rng.random())
        label = static_label_fn(state)
        root_counts[label] = root_counts.get(label, 0) + 1
        v59._simulate(
            root, state, horizon, tick, [], action_rows, rng,
            transition_fn, terminal_fn, action_cost_fn,
            exploration_constant, merge_observations, False,
        )
    selected = v59._deployment_action(root, action_rows, [], seed)
    payload = v59.tree_payload(root)
    nodes, branching, visited_actions = v59.tree_census(root)
    return v59.SearchResult(
        root=root,
        budget=budget,
        simulations_run=budget,
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


def plan_domain_fast(
    atoms: list[dict], registry: list[dict], entity_rows: list[dict], goal: dict,
    horizon: int, tick: int, budget: int, seed: int, config: dict,
    merge_observations: bool = False,
) -> v59.SearchResult:
    action_rows = candidate_actions(entity_rows)
    transition = v59.domain_transition_factory(registry, entity_rows)
    return run_root_sampled_uct_fast(
        atoms, action_rows, horizon, tick, budget, seed, transition,
        lambda state: float(state["world"][goal["atom"]] is goal["value"]),
        lambda action: float(config["planningModel"]["actionCost"][action["id"]]),
        v59.static_atom_label,
        exploration_constant=config["candidateSearch"]["explorationConstant"],
        merge_observations=merge_observations,
    )


def smc2_atoms_for_planning(pooled: dict) -> list[dict]:
    """Decode and merge a pooled V53 posterior without changing its measure."""
    grouped: dict[tuple[int, str, str], float] = defaultdict(float)
    theta_keys = set()
    for atom in pooled["atoms"]:
        program_index = int(atom["program_index"])
        theta_key = float(atom["theta"]).hex()
        configuration_key = atom["configuration_key"]
        grouped[(program_index, theta_key, configuration_key)] += float(atom["weight"])
        theta_keys.add((program_index, theta_key))
    node_indices = {
        key: index for index, key in enumerate(sorted(theta_keys))
    }
    rows = []
    for (program_index, theta_key, configuration_key), weight in sorted(grouped.items()):
        payload = json.loads(configuration_key)
        rows.append({
            "program_index": program_index,
            "node_index": node_indices[(program_index, theta_key)],
            "theta": float.fromhex(theta_key),
            "configuration_key": configuration_key,
            "world": dict(payload["world"]),
            "queue": list(payload["queue"]),
            "weight": weight,
        })
    total = sum(row["weight"] for row in rows)
    if total <= 0:
        raise RuntimeError("V60 converted SMC² belief has zero mass")
    for row in rows:
        row["weight"] /= total
    if abs(sum(row["weight"] for row in rows) - 1.0) > 1e-12:
        raise RuntimeError("V60 converted SMC² belief does not normalize")
    return rows


def atom_marginals(atoms: list[dict], program_count: int, theta_bins: int, parameter_model: dict) -> dict:
    program = [0.0] * program_count
    theta_values, theta_weights = [], []
    joint, configuration = {}, {}
    for atom in atoms:
        weight = float(atom["weight"])
        index = int(atom["program_index"])
        program[index] += weight
        theta_values.append(float(atom["theta"]))
        theta_weights.append(weight)
        bin_key = f"{index}:{theta_bin(atom['theta'], parameter_model, theta_bins)}"
        joint[bin_key] = joint.get(bin_key, 0.0) + weight
        key = atom["configuration_key"]
        configuration[key] = configuration.get(key, 0.0) + weight
    return {
        "program": normalize_float_sequence(program),
        "theta_values": theta_values,
        "theta_weights": normalize_float_sequence(theta_weights),
        "joint_bins": normalize_float_map(joint),
        "configuration": normalize_float_map(configuration),
    }


def sequence_tv(left, right) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))


def map_tv(left, right) -> float:
    return 0.5 * sum(
        abs(left.get(key, 0.0) - right.get(key, 0.0))
        for key in set(left) | set(right)
    )


def belief_comparison(exact: dict, estimate: dict) -> dict:
    return {
        "program_tv": sequence_tv(exact["program"], estimate["program"]),
        "theta_wasserstein": float(wasserstein_distance(
            exact["theta_values"], estimate["theta_values"],
            u_weights=exact["theta_weights"], v_weights=estimate["theta_weights"],
        )),
        "binned_program_theta_tv": map_tv(
            exact["joint_bins"], estimate["joint_bins"]
        ),
        "configuration_tv": map_tv(
            exact["configuration"], estimate["configuration"]
        ),
    }


def normalized_inference(result: dict, tolerance: float = 1e-10) -> bool:
    values = (
        sum(result["program"]), sum(result["theta_weights"]),
        sum(result["joint_bins"].values()),
        sum(result["configuration"].values()),
        sum(atom["weight"] for atom in result["atoms"]),
    )
    return all(abs(value - 1.0) <= tolerance for value in values)


def forbidden_truth_conditioned_belief(*_args, **_kwargs):
    raise PermissionError("V60 candidate inference may not condition on audit truth")
