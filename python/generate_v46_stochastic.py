#!/usr/bin/env python3
"""Construct the sealed V46 oracle stochastic-transition population."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from typing import Any, Sequence

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json, sha256_text
from v42_stateful import ONTOLOGY, action_bindings, deterministic_world, entities, epistemic_rows
from v46_stochastic import (
    ACTIONS, distribution_key, execute_distribution, map_determinized, mechanic_registry,
    uniformized,
)


def make_actions(entity_rows, pattern, token):
    bindings = action_bindings(entity_rows)
    start = int(sha256_text(f"binding|{token}")[:8], 16) % len(bindings)
    rows = []
    for index, action in enumerate(pattern):
        rows.append(
            {"id": "wait", "binding": {}}
            if action == "wait"
            else {"id": action, "binding": dict(bindings[(start + index) % len(bindings)])}
        )
    return rows


def complete_rows(world):
    return epistemic_rows(world)


def structural_key(entity_rows, state, actions):
    return sha256_text(canonical_json({"entities": entity_rows, "initial_state": state, "actions": actions}))


def support_case(index, seed):
    length = 2 + index % 3
    patterns = [pattern for pattern in product(ACTIONS, repeat=length) if "pulse" in pattern or "route" in pattern]
    pattern = patterns[(index // 3) % len(patterns)]
    count = 2 + index % 2
    entity_rows = entities(count)
    token = f"v46-support|{seed}|{index}"
    world = deterministic_world(entity_rows, token)
    actions = make_actions(entity_rows, pattern, token)
    state = complete_rows(world)
    return {
        "id": f"support_{sha256_text(token)[:16]}",
        "entities": entity_rows,
        "initial_world": world,
        "initial_state": state,
        "actions": actions,
        "structural_key": structural_key(entity_rows, state, actions),
    }


def trajectory(program, case):
    return execute_distribution(program, case["entities"], case["initial_world"], case["actions"])


def identifying_support(target, registry, seed, maximum):
    pool = [support_case(index, seed) for index in range(1024)]
    survivors = list(registry)
    selected, used = [], set()
    cache: dict[tuple[str, str], str] = {}

    def signature(mechanic, case):
        key = (mechanic["id"], case["id"])
        if key not in cache:
            cache[key] = distribution_key(trajectory(mechanic["program"], case))
        return cache[key]

    while len(survivors) > 1:
        choices = []
        for case in pool:
            if case["id"] in used:
                continue
            expected = signature(target, case)
            matching = [mechanic for mechanic in survivors if signature(mechanic, case) == expected]
            if len(matching) < len(survivors):
                choices.append((len(matching), sha256_text(f"{target['id']}|{case['id']}"), case, matching))
                if len(matching) == 1:
                    break
        if not choices:
            raise RuntimeError(f"V46 target not identifiable: {target['id']}")
        _, _, case, survivors = min(choices, key=lambda row: (row[0], row[1]))
        used.add(case["id"])
        selected.append({
            "id": case["id"],
            "entities": case["entities"],
            "initial_state": case["initial_state"],
            "actions": case["actions"],
            "observed_step_distributions": trajectory(target["program"], case),
            "structural_key": case["structural_key"],
        })
        if len(selected) > maximum:
            raise RuntimeError(f"V46 support budget exceeded: {target['id']}")
    if survivors[0]["id"] != target["id"]:
        raise RuntimeError("V46 support selected the wrong mechanic")
    return selected


def stochastic_rule(mechanic):
    for rule in mechanic["program"]["rules"]:
        branch = rule["stochastic_immediate"] or rule["stochastic_delayed"]
        if branch:
            return rule["action"], (branch[0].get("delay", 0))
    raise RuntimeError("V46 mechanic has no stochastic rule")


def query_pattern(mechanic, length, ordinal):
    trigger, delay = stochastic_rule(mechanic)
    other = "route" if trigger == "pulse" else "pulse"
    if delay and length > delay:
        base = [trigger, *(["wait"] * delay)]
        while len(base) < length:
            base.insert(-delay, other if ordinal % 2 else trigger)
        return base[:length]
    if delay:
        return [trigger, *([other] * (length - 1))]
    patterns = ([trigger, other, trigger, "wait"], [other, trigger, "wait", trigger], [trigger, "wait", other, trigger])
    return list(patterns[ordinal % len(patterns)][:length])


def query_case(mechanic, ordinal, seed, forbidden):
    length = 2 + ordinal % 3
    count = 2 + ordinal % 3
    entity_rows = entities(count)
    pattern = query_pattern(mechanic, length, ordinal)
    trigger, delay = stochastic_rule(mechanic)
    observable_expected = not delay or length > delay
    for attempt in range(8192):
        token = f"v46-query|{seed}|{mechanic['id']}|{ordinal}|{attempt}"
        world = deterministic_world(entity_rows, token)
        actions = make_actions(entity_rows, pattern, token)
        state = complete_rows(world)
        key = structural_key(entity_rows, state, actions)
        if key in forbidden:
            continue
        target = execute_distribution(mechanic["program"], entity_rows, world, actions)
        map_sensitive = distribution_key(target) != distribution_key(map_determinized(target))
        if observable_expected and not map_sensitive:
            continue
        forbidden.add(key)
        return {
            "id": f"query_{sha256_text(token)[:16]}",
            "entities": entity_rows,
            "initial_state": state,
            "actions": actions,
            "structural_key": key,
            "sequence_length": length,
            "entity_count": count,
            "probability_sensitive": map_sensitive,
            "uniform_sensitive": distribution_key(target) != distribution_key(uniformized(target)),
            "timing_sensitive": bool(delay and map_sensitive),
            "target": target,
        }
    raise RuntimeError(f"Could not construct V46 query: {mechanic['id']}/{ordinal}")


def build_population(config):
    registry = mechanic_registry()
    seed = config["population"]["generatorSeed"]
    records = []
    for mechanic in registry:
        supports = identifying_support(
            mechanic, registry, seed, config["population"]["supportSequencesPerMechanicMaximum"]
        )
        forbidden = {row["structural_key"] for row in supports}
        queries = [
            query_case(mechanic, ordinal, seed, forbidden)
            for ordinal in range(config["population"]["querySequencesPerMechanic"])
        ]
        if not any(row["probability_sensitive"] for row in queries):
            raise RuntimeError("V46 mechanic lacks a probability-sensitive query")
        if mechanic["timing"] == "delayed" and not any(row["timing_sensitive"] for row in queries):
            raise RuntimeError("V46 delayed mechanic lacks a timing-sensitive query")
        split = "development_fit" if mechanic["ordinal"] < 6 else "development_evaluation"
        records.append({
            "id": mechanic["id"],
            "schema_version": 46,
            "split": split,
            "construction_family": mechanic["family"],
            "agent_input": {
                "task": "infer_an_exact_stochastic_transition_mechanic_and_predict_each_distribution_valued_trajectory",
                "ontology": ONTOLOGY,
                "action_schemas": [
                    {"id": "pulse", "parameters": ONTOLOGY["action"]["parameters"]},
                    {"id": "route", "parameters": ONTOLOGY["action"]["parameters"]},
                    {"id": "wait", "parameters": []},
                ],
                "probability_vocabulary": config["probabilitySemantics"]["probabilityVocabulary"],
                "support_sequences": supports,
                "queries": [{key: value for key, value in row.items() if key != "target"} for row in queries],
            },
            "target": {"program": mechanic["program"], "program_key": mechanic["key"]},
            "oracle_queries": [{"id": row["id"], "target": row["target"]} for row in queries],
            "oracle_metadata": {
                "family_ordinal": mechanic["ordinal"],
                "probability": mechanic["probability"],
                "timing": mechanic["timing"],
                "support_sequences": len(supports),
                "query_sequences": len(queries),
                "probability_sensitive_queries": sum(row["probability_sensitive"] for row in queries),
                "uniform_sensitive_queries": sum(row["uniform_sensitive"] for row in queries),
                "timing_sensitive_queries": sum(row["timing_sensitive"] for row in queries),
            },
        })
    if len(records) != 40 or len({row["target"]["program_key"] for row in records}) != 40:
        raise RuntimeError("V46 population must contain 40 unique mechanics")
    return sorted(records, key=lambda row: row["id"])


def corpus_hash(rows: Sequence[dict[str, Any]]):
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v46-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_development_population"]:
        raise RuntimeError("V46 implementation lock does not authorize construction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V46 implementation changed: {path}")
    output = PROJECT_ROOT / "data/v46-oracle-stochastic-transitions"
    if output.exists():
        raise RuntimeError("V46 population already exists")
    rows = build_population(lock["config_payload"])
    if corpus_hash(rows) != lock["expected_corpus_sha256"]:
        raise RuntimeError("V46 corpus differs from implementation lock")
    output.mkdir(parents=True)
    artifacts = {}
    for split in ("development_fit", "development_evaluation"):
        selected = [row for row in rows if row["split"] == split]
        path = output / f"{split}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in selected))
        artifacts[split] = {
            "path": str(path.relative_to(PROJECT_ROOT)), "records": len(selected), "sha256": file_sha256(path)
        }
    manifest = {
        "schema_version": 46,
        "experiment": lock["config_payload"]["experiment"],
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "counts": {
            "mechanics": len(rows),
            "support_sequences": sum(len(row["agent_input"]["support_sequences"]) for row in rows),
            "query_sequences": sum(len(row["agent_input"]["queries"]) for row in rows),
            "probability_sensitive_queries": sum(row["oracle_metadata"]["probability_sensitive_queries"] for row in rows),
            "uniform_sensitive_queries": sum(row["oracle_metadata"]["uniform_sensitive_queries"] for row in rows),
            "timing_sensitive_queries": sum(row["oracle_metadata"]["timing_sensitive_queries"] for row in rows),
            "families": dict(Counter(row["construction_family"] for row in rows)),
            "splits": dict(Counter(row["split"] for row in rows)),
            "probabilities": dict(Counter(row["oracle_metadata"]["probability"] for row in rows)),
        },
        "data_access": {"oracle_development_runs": 0, "sampled_realizations": 0, "model_forward_passes": 0, "adapter_training_runs": 0},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
