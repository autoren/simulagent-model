#!/usr/bin/env python3
"""Construct the sealed V55r1 delayed-consequence confirmation population."""
from __future__ import annotations

import argparse
import json
import random

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text, unary_atom
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import (
    action_bindings,
    atom_universe,
    deterministic_world,
    entities,
    epistemic_rows,
)
from v46_stochastic import _configuration_key
from v49_belief import masked_trace
from v53_smc2 import (
    continuous_unit_transition,
    instantiate_program,
    scaled_beta_sample,
    stream_seed,
)
from v55_planning import assert_planning_payload_is_public
from v55r1_planning import planning_registry


def observation_design_key(row: dict) -> str:
    return sha256_text(canonical_json({
        key: row[key] for key in ("entities", "initial_state", "actions", "masks")
    }))


def _visit_designs(value, result: set[str]) -> None:
    if isinstance(value, dict):
        if {"entities", "initial_state", "actions", "masks"} <= set(value):
            result.add(observation_design_key(value))
        for child in value.values():
            _visit_designs(child, result)
    elif isinstance(value, list):
        for child in value:
            _visit_designs(child, result)


def prior_observation_design_keys() -> set[str]:
    result: set[str] = set()
    for version in range(46, 56):
        for path in sorted((PROJECT_ROOT / "data").glob(f"v{version}*/*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    _visit_designs(json.loads(line), result)
    return result


def history_class_for_record(record: int) -> str:
    return "prior_like_all_wait" if record % 2 == 0 else "mixed_informative"


def target_assignments(config: dict, registry_size: int = 8) -> list[int]:
    count = config["population"]["confirmationTasks"]
    copies = config["population"]["tasksPerGeneratingTemplate"]
    values = [index for index in range(registry_size) for _ in range(copies)]
    if len(values) != count:
        raise RuntimeError("V55r1 truth allocation does not match task count")
    random.Random(config["population"]["truthAssignmentSeed"]).shuffle(values)
    return values


def goal_assignments(config: dict) -> list[dict]:
    count = config["population"]["confirmationTasks"]
    entity_rows = entities(config["population"]["entityCount"])
    values = [
        {"atom": unary_atom("active", entity["id"]), "value": value}
        for entity in entity_rows for value in (False, True) for _ in range(4)
    ]
    if len(values) != count:
        raise RuntimeError("V55r1 goal allocation does not match task count")
    random.Random(config["population"]["goalSeed"]).shuffle(values)
    return values


def mask_schedule(entity_rows, steps: int, token: str, config: dict) -> list[list[str]]:
    atoms = list(atom_universe(entity_rows))
    count = max(1, min(len(atoms) - 1, len(atoms) // 2))
    seed = config["population"]["historySeed"]
    return [
        sorted(sorted(atoms, key=lambda atom: sha256_text(
            f"v55r1-mask|{seed}|{token}|{tick}|{atom}"
        ))[:count])
        for tick in range(steps)
    ]


def action_schedule(
    entity_rows, steps: int, token: str, history_class: str, config: dict
) -> list[dict]:
    if history_class == "prior_like_all_wait":
        return [{"id": "wait", "binding": {}} for _ in range(steps)]
    bindings = action_bindings(entity_rows)
    seed = config["population"]["historySeed"]
    actions = []
    for tick in range(steps):
        action_id = ("pulse", "route", "wait")[
            int(sha256_text(f"v55r1-action|{seed}|{token}|{tick}")[:12], 16) % 3
        ]
        if action_id == "wait":
            actions.append({"id": "wait", "binding": {}})
            continue
        binding = bindings[int(sha256_text(
            f"v55r1-binding|{seed}|{token}|{tick}"
        )[:12], 16) % len(bindings)]
        actions.append({"id": action_id, "binding": dict(binding)})
    if steps:
        actions[0] = {"id": "pulse", "binding": dict(bindings[0])}
    if steps > 1:
        actions[1] = {"id": "route", "binding": dict(bindings[-1])}
    return actions


def make_case(
    record: int,
    ordinal: int,
    length: int,
    history_class: str,
    config: dict,
    kind: str,
    used: set[str],
    prior: set[str],
    query_goal: dict | None = None,
) -> dict:
    nonce = 0
    while True:
        token = (
            f"v55r1|{config['population']['generatorSeed']}|{kind}|{record}|"
            f"{ordinal}|{history_class}|{nonce}"
        )
        entity_rows = entities(config["population"]["entityCount"])
        world = deterministic_world(entity_rows, token)
        if query_goal is not None:
            for entity in entity_rows:
                world[unary_atom("active", entity["id"])] = not query_goal["value"]
        row = {
            "id": f"{kind}_{sha256_text(token)[:16]}",
            "entities": entity_rows,
            "initial_world": world,
            "initial_state": epistemic_rows(world),
            "actions": action_schedule(
                entity_rows, length, token, history_class, config
            ),
            "masks": mask_schedule(entity_rows, length, token, config),
            "sequence_length": length,
            "entity_count": len(entity_rows),
        }
        key = observation_design_key(row)
        if key not in used and key not in prior:
            row["observation_design_key"] = key
            used.add(key)
            return row
        nonce += 1


def sample_branch(branches: dict, seed: int) -> dict:
    ordered = sorted(branches.items())
    draw = random.Random(seed).random()
    cumulative = 0.0
    for _, row in ordered:
        cumulative += float(row["mass"])
        if draw < cumulative:
            return row
    return ordered[-1][1]


def simulate_case(target: dict, theta: float, case: dict, config: dict, token: str) -> dict:
    program = instantiate_program(target["template"], theta)
    world, queue, history = dict(case["initial_world"]), [], []
    for tick, action in enumerate(case["actions"]):
        branches = continuous_unit_transition(
            program, case["entities"], world, queue, action, tick
        )
        selected = sample_branch(branches, stream_seed(
            config["population"]["trajectorySeed"], token, tick, "history"
        ))
        world, queue = dict(selected["world"]), list(selected["queue"])
        history.append(selected["history"][-1])
    return {"world": world, "queue": queue, "history": history}


def public_case(case: dict, sampled: dict) -> dict:
    return {
        key: case[key]
        for key in (
            "id", "entities", "initial_state", "actions", "masks",
            "sequence_length", "entity_count", "observation_design_key",
        )
    } | {"observations": masked_trace(sampled["history"], case["masks"])}


def record_theta(record: int, config: dict) -> float:
    return scaled_beta_sample(stream_seed(
        config["population"]["thetaPriorSeed"],
        "v55r1", record, "theta-prior",
    ))


def build_record(
    record: int,
    target_index: int,
    goal: dict,
    registry: list[dict],
    config: dict,
    used: set[str],
    prior: set[str],
) -> dict:
    history_class = history_class_for_record(record)
    target = registry[target_index]
    theta = record_theta(record, config)
    supports = []
    lengths = config["population"]["supportSequenceLengths"]
    for ordinal in range(config["population"]["supportEpisodesPerTask"]):
        length = lengths[(record + ordinal) % len(lengths)]
        case = make_case(
            record, ordinal, length, history_class, config,
            "support", used, prior,
        )
        sampled = simulate_case(
            target, theta, case, config,
            f"v55r1|{record}|support|{ordinal}",
        )
        supports.append(public_case(case, sampled))
    prefixes = config["population"]["ongoingHistoryPrefixLengths"]
    prefix = prefixes[record % len(prefixes)]
    case = make_case(
        record, 0, prefix, history_class, config,
        "query", used, prior, query_goal=goal,
    )
    sampled = simulate_case(
        target, theta, case, config, f"v55r1|{record}|query"
    )
    query = public_case(case, sampled)
    query["prefix_length"] = prefix
    public = {"supports": supports, "query": query, "goal": dict(goal)}
    assert_planning_payload_is_public(public)
    return {
        "id": f"planning_r1_{record:05d}",
        "schema_version": 55,
        "revision": "r1",
        "population": "delayed_consequence_confirmation",
        "record": record,
        "history_class": history_class,
        "public": public,
        "truth": {
            "target_program_index": target_index,
            "target_program_key": target["key"],
            "target_program_ordinal": target["program_ordinal"],
            "target_theta": theta,
            "query_configuration_key": _configuration_key(
                sampled["world"], sampled["queue"]
            ),
            "query_world": sampled["world"],
            "query_queue": sampled["queue"],
        },
    }


def build_population(config: dict) -> list[dict]:
    registry = planning_registry(config)
    used, prior = set(), prior_observation_design_keys()
    assignments = target_assignments(config, len(registry))
    goals = goal_assignments(config)
    return [
        build_record(
            record, assignments[record], goals[record],
            registry, config, used, prior,
        )
        for record in range(config["population"]["confirmationTasks"])
    ]


def population_hash(rows: list[dict]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v55r1-implementation-lock.json"
    )
    parser.add_argument(
        "--output-dir",
        default="data/v55r1-delayed-consequence-adequacy-confirmation",
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_v55r1_population"]:
        raise RuntimeError("V55r1 implementation lock does not authorize population construction")
    for relative, digest in lock["implementation_files_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"V55r1 frozen implementation changed: {relative}")
    design = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())
    rows = build_population(design["config_payload"])
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "planning.jsonl"
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))
    manifest = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_population_manifest",
        "count": len(rows),
        "file": {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
        },
        "population_hash": population_hash(rows),
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
