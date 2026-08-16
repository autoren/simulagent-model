#!/usr/bin/env python3
"""Construct the exact V42 oracle sequential development population."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from typing import Any, Sequence

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json, sha256_text
from v42_stateful import (
    ACTIONS,
    ONTOLOGY,
    action_bindings,
    atom_universe,
    compatible_worlds,
    deterministic_world,
    entities,
    epistemic_rows,
    execute_partial,
    execute_sequence,
    mechanic_registry,
    memoryless_execute_sequence,
    world_signature,
)


def complete_state_rows(world: dict[str, bool]) -> list[dict[str, Any]]:
    return [{"atom": atom, "value": value} for atom, value in sorted(world.items())]


def action_sequence(
    entity_rows: Sequence[dict[str, str]], length: int, token: str,
    action_pattern: Sequence[str] | None = None, same_binding: bool = False,
) -> list[dict[str, Any]]:
    bindings = action_bindings(entity_rows)
    start = int(sha256_text(f"binding|{token}")[:8], 16) % len(bindings)
    if action_pattern is None:
        offset = int(sha256_text(f"action|{token}")[:2], 16) % 2
        action_pattern = [ACTIONS[(index + offset) % 2] for index in range(length)]
        if int(sha256_text(f"repeat|{token}")[:2], 16) % 4 == 0:
            action_pattern = [ACTIONS[offset]] * length
    if len(action_pattern) != length:
        raise ValueError("V42 action pattern length mismatch")
    return [
        {
            "id": action,
            "binding": dict(bindings[start] if same_binding else bindings[(start + index) % len(bindings)]),
        }
        for index, action in enumerate(action_pattern)
    ]


def structural_key(
    entity_rows: Sequence[dict[str, str]], state_rows: Sequence[dict[str, Any]],
    actions: Sequence[dict[str, Any]],
) -> str:
    return sha256_text(canonical_json({
        "entities": list(entity_rows), "initial_state": list(state_rows), "actions": list(actions),
    }))


def support_case(index: int, seed: int) -> dict[str, Any]:
    count = 2 + (index % 2)
    entity_rows = entities(count)
    token = f"support-pool|{seed}|{index}"
    world = deterministic_world(entity_rows, token)
    length = 2 + (index % 3)
    actions = action_sequence(entity_rows, length, token, same_binding=index % 3 == 0)
    return {
        "id": f"support_case_{sha256_text(token)[:16]}",
        "entities": entity_rows,
        "initial_world": world,
        "actions": actions,
        "structural_key": structural_key(entity_rows, epistemic_rows(world), actions),
    }


def trajectory_key(program: dict[str, Any], case: dict[str, Any]) -> tuple[str, ...]:
    return tuple(world_signature(world) for world in execute_sequence(
        program, case["entities"], case["initial_world"], case["actions"]
    ))


def identifying_support(
    target: dict[str, Any], registry: Sequence[dict[str, Any]], seed: int, maximum: int,
    pool: Sequence[dict[str, Any]] | None = None,
    signatures: dict[tuple[str, str], tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    pool = list(pool) if pool is not None else [support_case(index, seed) for index in range(512)]
    signatures = signatures if signatures is not None else {
        (mechanic["id"], case["id"]): trajectory_key(mechanic["program"], case)
        for mechanic in registry for case in pool
    }
    survivors = list(registry)
    selected = []
    used = set()
    while len(survivors) > 1:
        choices = []
        for case in pool:
            if case["id"] in used:
                continue
            target_signature = signatures[(target["id"], case["id"])]
            matching = [
                mechanic for mechanic in survivors
                if signatures[(mechanic["id"], case["id"])] == target_signature
            ]
            if len(matching) < len(survivors):
                choices.append((len(matching), sha256_text(f"{target['id']}|{case['id']}"), case, matching))
        if not choices:
            raise RuntimeError(f"V42 target is not identifiable in support pool: {target['id']}")
        _, _, case, survivors = min(choices, key=lambda row: (row[0], row[1]))
        used.add(case["id"])
        trajectory = execute_sequence(target["program"], case["entities"], case["initial_world"], case["actions"])
        selected.append({
            "id": case["id"],
            "entities": case["entities"],
            "initial_state": epistemic_rows(case["initial_world"]),
            "actions": case["actions"],
            "observed_step_states": [complete_state_rows(world) for world in trajectory],
            "structural_key": case["structural_key"],
        })
        if len(selected) > maximum:
            raise RuntimeError(f"V42 target exceeds support budget: {target['id']}")
    if survivors[0]["id"] != target["id"]:
        raise RuntimeError("V42 greedy support selected the wrong unique mechanic")
    return selected


def _query_pattern(length: int, pair_index: int) -> list[str]:
    if length == 2:
        return ["pulse", "route"]
    if length == 3:
        return ["pulse", "route", "route"] if pair_index % 2 == 0 else ["pulse", "pulse", "route"]
    return ["pulse", "route", "pulse", "route"] if pair_index % 2 == 0 else ["pulse", "pulse", "route", "route"]


def query_pair(
    mechanic: dict[str, Any], pair_index: int, seed: int, forbidden_keys: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    length = 2 + (pair_index % 3)
    count = 2 + (pair_index % 4)
    entity_rows = entities(count)
    pattern = _query_pattern(length, pair_index)
    for attempt in range(4096):
        token = f"query|{seed}|{mechanic['id']}|{pair_index}|{attempt}"
        world = deterministic_world(entity_rows, token)
        forward_actions = action_sequence(entity_rows, length, token, pattern, same_binding=True)
        reverse_actions = list(reversed(forward_actions))
        unknown_atoms: list[str] = []
        if pair_index % 3 == 2:
            universe = atom_universe(entity_rows)
            unknown_atoms = [universe[int(sha256_text(f"unknown|{token}")[:8], 16) % len(universe)]]
        initial = epistemic_rows(world, unknown_atoms)
        forward_key = structural_key(entity_rows, initial, forward_actions)
        reverse_key = structural_key(entity_rows, initial, reverse_actions)
        if forward_key in forbidden_keys or reverse_key in forbidden_keys or forward_key == reverse_key:
            continue
        forward_target = execute_partial([mechanic["program"]], entity_rows, initial, forward_actions)
        reverse_target = execute_partial([mechanic["program"]], entity_rows, initial, reverse_actions)
        forward_memoryless = execute_partial([mechanic["program"]], entity_rows, initial, forward_actions, memoryless=True)
        reverse_memoryless = execute_partial([mechanic["program"]], entity_rows, initial, reverse_actions, memoryless=True)
        memoryless_failures = sum((
            forward_target["possible_final_observations"] != forward_memoryless["possible_final_observations"],
            reverse_target["possible_final_observations"] != reverse_memoryless["possible_final_observations"],
        ))
        order_effect = forward_target["possible_final_observations"] != reverse_target["possible_final_observations"]
        if memoryless_failures == 0:
            continue
        if mechanic["family"] == "order_sensitive_composition" and pair_index == 0 and not order_effect:
            continue
        group = f"order_{sha256_text(f'{mechanic['id']}|{pair_index}')[:16]}"

        def row(role: str, actions: list[dict[str, Any]], target: dict[str, Any], key: str) -> dict[str, Any]:
            return {
                "id": f"query_{sha256_text(f'{token}|{role}')[:16]}",
                "entities": entity_rows,
                "initial_state": initial,
                "actions": actions,
                "structural_key": key,
                "sequence_length": length,
                "entity_count": count,
                "partial_initial_state": bool(unknown_atoms),
                "order_counterfactual_group": group,
                "order_counterfactual_role": role,
                "order_effect": order_effect,
                "target": target,
            }

        forbidden_keys.update((forward_key, reverse_key))
        return (
            row("forward", forward_actions, forward_target, forward_key),
            row("reversed", reverse_actions, reverse_target, reverse_key),
        )
    raise RuntimeError(f"Could not construct V42 query pair for {mechanic['id']}/{pair_index}")


def build_population(config: dict[str, Any]) -> list[dict[str, Any]]:
    registry = mechanic_registry()
    seed = config["population"]["generatorSeed"]
    support_pool = [support_case(index, seed) for index in range(512)]
    support_signatures = {
        (mechanic["id"], case["id"]): trajectory_key(mechanic["program"], case)
        for mechanic in registry for case in support_pool
    }
    records = []
    for mechanic in registry:
        support = identifying_support(
            mechanic, registry, seed,
            config["population"]["supportSequencesPerMechanicMaximum"],
            support_pool, support_signatures,
        )
        forbidden = {row["structural_key"] for row in support}
        queries = []
        for pair_index in range(config["population"]["querySequencesPerMechanic"] // 2):
            queries.extend(query_pair(mechanic, pair_index, seed, forbidden))
        if len(queries) != config["population"]["querySequencesPerMechanic"]:
            raise RuntimeError("V42 query quota failed")
        split = "development_fit" if mechanic["ordinal"] < 6 else "development_evaluation"
        order_pairs = Counter()
        for query in queries:
            if query["order_effect"]:
                order_pairs[query["order_counterfactual_group"]] += 1
        if mechanic["family"] == "order_sensitive_composition" and not any(count == 2 for count in order_pairs.values()):
            raise RuntimeError("V42 order-sensitive mechanic lacks a causal order pair")
        records.append({
            "id": mechanic["id"],
            "schema_version": 42,
            "split": split,
            "construction_family": mechanic["family"],
            "agent_input": {
                "task": "infer_a_stateful_lifted_mechanic_and_predict_each_query_trajectory",
                "ontology": ONTOLOGY,
                "action_schemas": [
                    {"id": action, "parameters": ONTOLOGY["action"]["parameters"], "distinct_parameters": True}
                    for action in ACTIONS
                ],
                "dsl_contract": config["statefulDsl"],
                "support_sequences": support,
                "queries": [
                    {key: value for key, value in query.items() if key != "target"}
                    for query in queries
                ],
            },
            "target": {
                "program": mechanic["program"],
                "program_key": mechanic["key"],
            },
            "oracle_queries": [
                {"id": query["id"], "target": query["target"]}
                for query in queries
            ],
            "oracle_metadata": {
                "family_ordinal": mechanic["ordinal"],
                "support_sequences": len(support),
                "query_sequences": len(queries),
                "causal_order_pairs": sum(count == 2 for count in order_pairs.values()),
            },
        })
    if len(records) != 40 or len({row["target"]["program_key"] for row in records}) != 40:
        raise RuntimeError("V42 population must contain 40 unique mechanics")
    return sorted(records, key=lambda row: row["id"])


def corpus_hash(rows: Sequence[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v42-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_development_population"]:
        raise RuntimeError("V42 implementation lock does not authorize construction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V42 locked implementation changed: {path}")
    output = PROJECT_ROOT / "data/v42-sequential-state-foundation"
    if output.exists():
        raise RuntimeError("V42 development population already exists")
    rows = build_population(lock["config_payload"])
    if corpus_hash(rows) != lock["expected_corpus_sha256"]:
        raise RuntimeError("V42 population differs from implementation lock")
    output.mkdir(parents=True)
    artifacts = {}
    for split in ("development_fit", "development_evaluation"):
        selected = [row for row in rows if row["split"] == split]
        path = output / f"{split}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in selected))
        artifacts[split] = {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(selected), "sha256": file_sha256(path)}
    manifest = {
        "schema_version": 42,
        "experiment": config["experiment"] if (config := lock["config_payload"]) else None,
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "counts": {
            "families": dict(Counter(row["construction_family"] for row in rows)),
            "splits": dict(Counter(row["split"] for row in rows)),
            "support_sequences": sum(len(row["agent_input"]["support_sequences"]) for row in rows),
            "query_sequences": sum(len(row["agent_input"]["queries"]) for row in rows),
            "sequence_lengths": dict(Counter(str(query["sequence_length"]) for row in rows for query in row["agent_input"]["queries"])),
            "entity_counts": dict(Counter(str(query["entity_count"]) for row in rows for query in row["agent_input"]["queries"])),
            "partial_queries": sum(query["partial_initial_state"] for row in rows for query in row["agent_input"]["queries"]),
            "causal_order_pairs": sum(row["oracle_metadata"]["causal_order_pairs"] for row in rows),
        },
        "data_access": {"oracle_development_runs": 0, "language_model_forward_passes": 0, "adapter_training_runs": 0, "v41_records_read": 0},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
