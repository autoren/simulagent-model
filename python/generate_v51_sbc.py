#!/usr/bin/env python3
"""Construct fresh prior-predictive replication specifications for V51."""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import product
from typing import Any, Sequence

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import action_bindings, atom_universe, deterministic_world, entities, epistemic_rows
from v46_stochastic import ACTIONS, _configuration_key
from v49_belief import _configuration_key_with_history, advance_configurations, masked_trace
from v51_sbc import categorical_sample, mechanic_registry


def structural_key(entity_rows, state, actions):
    return sha256_text(canonical_json({"entities": entity_rows, "initial_state": state, "actions": actions}))


def mask_schedule(entity_rows, steps: int, token: str, mask_seed: int):
    atoms = list(atom_universe(entity_rows))
    count = max(1, min(len(atoms) - 1, len(atoms) // 2))
    return [
        sorted(sorted(atoms, key=lambda atom: sha256_text(
            f"v51-mask|{mask_seed}|{token}|{step}|{atom}"
        ))[:count])
        for step in range(steps)
    ]


def make_case(replication: int, ordinal: int, length: int, config, kind: str):
    simulation = config["simulation"]
    entity_count = simulation["entityCounts"][(replication + ordinal) % len(simulation["entityCounts"])]
    entity_rows = entities(entity_count)
    token = f"v51-{kind}|{simulation['generatorSeed']}|{replication}|{ordinal}"
    world = deterministic_world(entity_rows, token)
    state = epistemic_rows(world)
    patterns = [pattern for pattern in product(ACTIONS, repeat=length) if "pulse" in pattern and "route" in pattern]
    pattern = patterns[int(sha256_text(f"v51-pattern|{token}")[:12], 16) % len(patterns)]
    bindings = action_bindings(entity_rows)
    start = int(sha256_text(f"v51-binding|{token}")[:12], 16) % len(bindings)
    actions = [
        {"id": "wait", "binding": {}}
        if action == "wait"
        else {"id": action, "binding": dict(bindings[(start + index) % len(bindings)])}
        for index, action in enumerate(pattern)
    ]
    masks = mask_schedule(entity_rows, length, token, simulation["generatorSeed"])
    return {
        "id": f"{kind}_{sha256_text(token)[:16]}",
        "entities": entity_rows,
        "initial_world": world,
        "initial_state": state,
        "actions": actions,
        "masks": masks,
        "sequence_length": length,
        "entity_count": entity_count,
        "structural_key": structural_key(entity_rows, state, actions),
    }


def sample_configuration(program, case, action_count: int, seed: int):
    initial = {
        _configuration_key_with_history(case["initial_world"], [], []): {
            "world": dict(case["initial_world"]), "queue": [], "history": [], "mass": Fraction(1)
        }
    }
    configurations = advance_configurations(
        program, case["entities"], initial, case["actions"][:action_count], 0
    )
    masses = {key: row["mass"] for key, row in configurations.items()}
    selected_key = categorical_sample(masses, seed)
    return configurations[selected_key]


def public_case(case):
    return {
        key: case[key]
        for key in (
            "id", "entities", "initial_state", "actions", "masks", "sequence_length",
            "entity_count", "structural_key",
        )
    }


def build_replications(config):
    registry = mechanic_registry()
    simulation = config["simulation"]
    rows = []
    for replication in range(simulation["replications"]):
        target_index = int(sha256_text(
            f"v51-prior|{simulation['priorSeed']}|{replication}"
        ), 16) % len(registry)
        target = registry[target_index]
        supports = []
        for ordinal in range(simulation["supportInterventionsPerReplication"]):
            length = simulation["sequenceLengths"][(replication + ordinal) % len(simulation["sequenceLengths"])]
            case = make_case(replication, ordinal, length, config, "support")
            sampled = sample_configuration(
                target["program"], case, length,
                int(sha256_text(
                    f"v51-support-trajectory|{simulation['trajectorySeed']}|{replication}|{ordinal}"
                ), 16),
            )
            supports.append({
                **public_case(case),
                "observations": masked_trace(sampled["history"], case["masks"]),
            })
        pair_index = replication % len(simulation["sequenceLengths"])
        length = simulation["sequenceLengths"][pair_index]
        prefix_length = simulation["queryPrefixLengths"][pair_index]
        query_case = make_case(replication, 0, length, config, "query")
        sampled_prefix = sample_configuration(
            target["program"], query_case, prefix_length,
            int(sha256_text(
                f"v51-query-trajectory|{simulation['trajectorySeed']}|{replication}"
            ), 16),
        )
        query = {
            **public_case(query_case),
            "prefix_length": prefix_length,
            "observations": masked_trace(
                sampled_prefix["history"], query_case["masks"][:prefix_length]
            ),
            "true_configuration_key": _configuration_key(
                sampled_prefix["world"], sampled_prefix["queue"]
            ),
        }
        rows.append({
            "id": f"replication_{replication:05d}",
            "schema_version": 51,
            "replication": replication,
            "target_program_index": target_index,
            "target_program_key": target["key"],
            "target_program_ordinal": target["program_ordinal"],
            "target_probability_ordinal": target["probability_ordinal"],
            "family": target["family"],
            "probability": target["probability"],
            "timing": target["timing"],
            "supports": supports,
            "query": query,
        })
    return rows


def corpus_hash(rows: Sequence[dict[str, Any]]):
    return sha256_text("".join(canonical_json(row) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v51-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_calibration_replications"]:
        raise RuntimeError("V51 implementation lock does not authorize construction")
    output = PROJECT_ROOT / "data/v51-simulation-based-calibration"
    if output.exists():
        raise RuntimeError("V51 calibration corpus already exists")
    rows = build_replications(lock["config_payload"])
    if corpus_hash(rows) != lock["expected_corpus_sha256"]:
        raise RuntimeError("V51 calibration corpus differs from implementation precommit")
    output.mkdir(parents=True)
    path = output / "replications.jsonl"
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))
    manifest = {
        "schema_version": 51,
        "experiment": "v51_calibration_population",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "corpus_sha256": corpus_hash(rows),
        "artifact": {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
            "replications": len(rows),
        },
        "counts": {
            "replications": len(rows),
            "support_observations": sum(len(row["supports"]) for row in rows),
            "query_observations": len(rows),
        },
        "data_access": {
            "calibration_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
