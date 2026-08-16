#!/usr/bin/env python3
"""Construct V53 continuous-parameter exact, SBC, and scale populations."""
from __future__ import annotations

import argparse
import json
import random

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import action_bindings, atom_universe, deterministic_world, entities, epistemic_rows
from v46_stochastic import ACTIONS, _configuration_key
from v49_belief import masked_trace
from v53_smc2 import (
    continuous_unit_transition,
    instantiate_program,
    mechanic_registry,
    scaled_beta_sample,
    stream_seed,
)


def observation_design_key(row):
    return sha256_text(canonical_json({
        key: row[key] for key in ("entities", "initial_state", "actions", "masks")
    }))


def _visit_designs(value, result):
    if isinstance(value, dict):
        if {"entities", "initial_state", "actions", "masks"} <= set(value):
            result.add(observation_design_key(value))
        for child in value.values():
            _visit_designs(child, result)
    elif isinstance(value, list):
        for child in value:
            _visit_designs(child, result)


def prior_observation_design_keys():
    result = set()
    for version in range(46, 53):
        for path in sorted((PROJECT_ROOT / "data").glob(f"v{version}-*/*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    _visit_designs(json.loads(line), result)
    return result


def mask_schedule(entity_rows, steps, token, mask_seed):
    atoms = list(atom_universe(entity_rows))
    count = max(1, min(len(atoms) - 1, len(atoms) // 2))
    return [
        sorted(sorted(atoms, key=lambda atom: sha256_text(
            f"v53-mask|{mask_seed}|{token}|{tick}|{atom}"
        ))[:count])
        for tick in range(steps)
    ]


def action_schedule(entity_rows, steps, token, all_wait=False):
    if all_wait:
        return [{"id": "wait", "binding": {}} for _ in range(steps)]
    action_ids = [
        ACTIONS[int(sha256_text(f"v53-action|{token}|{tick}")[:12], 16) % len(ACTIONS)]
        for tick in range(steps)
    ]
    if steps:
        action_ids[0] = "pulse"
    if steps > 1:
        action_ids[1] = "route"
    bindings = action_bindings(entity_rows)
    start = int(sha256_text(f"v53-binding|{token}")[:12], 16) % len(bindings)
    return [
        {"id": "wait", "binding": {}}
        if action == "wait"
        else {"id": action, "binding": dict(bindings[(start + tick) % len(bindings)])}
        for tick, action in enumerate(action_ids)
    ]


def make_case(
    population, record, ordinal, length, entity_counts, config, kind, used, prior,
    all_wait=False,
):
    nonce = 0
    while True:
        token = (
            f"v53|{config['population']['generatorSeed']}|{population}|{kind}|"
            f"{record}|{ordinal}|{nonce}"
        )
        entity_count = entity_counts[(record + ordinal + nonce) % len(entity_counts)]
        entity_rows = entities(entity_count)
        world = deterministic_world(entity_rows, token)
        row = {
            "id": f"{kind}_{sha256_text(token)[:16]}",
            "entities": entity_rows,
            "initial_world": world,
            "initial_state": epistemic_rows(world),
            "actions": action_schedule(entity_rows, length, token, all_wait),
            "masks": mask_schedule(
                entity_rows, length, token, config["population"]["maskSeed"]
            ),
            "sequence_length": length,
            "entity_count": entity_count,
        }
        key = observation_design_key(row)
        if key not in used and key not in prior:
            row["observation_design_key"] = key
            used.add(key)
            return row
        nonce += 1


def sample_branch(branches, seed):
    ordered = sorted(branches.items())
    draw = random.Random(seed).random()
    cumulative = 0.0
    for _, row in ordered:
        cumulative += float(row["mass"])
        if draw < cumulative:
            return row
    return ordered[-1][1]


def simulate_trace(template, theta, case, steps, config, token):
    program = instantiate_program(template, theta)
    world, queue, history = dict(case["initial_world"]), [], []
    for tick, action in enumerate(case["actions"][:steps]):
        branches = continuous_unit_transition(
            program, case["entities"], world, queue, action, tick
        )
        selected = sample_branch(
            branches,
            stream_seed(
                config["population"]["trajectorySeed"], token, tick, "trajectory"
            ),
        )
        world, queue = dict(selected["world"]), list(selected["queue"])
        history.append(selected["history"][-1])
    return {"world": world, "queue": queue, "history": history}


def public_case(case):
    return {
        key: case[key]
        for key in (
            "id", "entities", "initial_state", "actions", "masks",
            "sequence_length", "entity_count", "observation_design_key",
        )
    }


def observed_episode(target, theta, case, config, token, steps=None):
    steps = case["sequence_length"] if steps is None else steps
    sampled = simulate_trace(target["template"], theta, case, steps, config, token)
    return {
        **public_case(case),
        "observations": masked_trace(sampled["history"], case["masks"][:steps]),
    }, sampled


def target_fields(target, target_index, theta):
    return {
        "target_program_index": target_index,
        "target_program_key": target["key"],
        "target_program_ordinal": target["program_ordinal"],
        "target_theta": theta,
        "family": target["family"],
        "timing": target["timing"],
    }


def record_theta(config, population, record):
    return scaled_beta_sample(stream_seed(
        config["population"]["thetaPriorSeed"], population, record, "theta-prior"
    ))


def build_exact(registry, config, used, prior):
    specification = config["exactBenchmark"]
    rows = []
    for target_index, target in enumerate(registry):
        for replicate in range(specification["recordsPerTemplate"]):
            record_index = target_index * specification["recordsPerTemplate"] + replicate
            ambiguity_probe = replicate in specification.get(
                "ambiguityProbeReplicateOrdinalsPerTemplate", []
            )
            theta = record_theta(config, "exact", record_index)
            supports = []
            for ordinal in range(specification["supportEpisodesPerRecord"]):
                length = specification["supportSequenceLengths"][(record_index + ordinal) % 2]
                case = make_case(
                    "exact", record_index, ordinal, length, specification["entityCounts"],
                    config, "support", used, prior, ambiguity_probe,
                )
                observed, _ = observed_episode(
                    target, theta, case, config, f"exact|{record_index}|support|{ordinal}"
                )
                supports.append(observed)
            pair = record_index % len(specification["querySequenceLengths"])
            length = specification["querySequenceLengths"][pair]
            prefix = specification["queryPrefixLengths"][pair]
            case = make_case(
                "exact", record_index, 0, length, specification["entityCounts"],
                config, "query", used, prior, ambiguity_probe,
            )
            query, sampled = observed_episode(
                target, theta, case, config, f"exact|{record_index}|query", prefix
            )
            query.update({
                "prefix_length": prefix,
                "true_configuration_key": _configuration_key(
                    sampled["world"], sampled["queue"]
                ),
            })
            rows.append({
                "id": f"exact_{record_index:05d}",
                "schema_version": 53,
                "population": "exact",
                "record": record_index,
                **target_fields(target, target_index, theta),
                "supports": supports,
                "query": query,
                "pmcmc_reference": record_index % 2 == 0,
                "ambiguity_probe": ambiguity_probe,
            })
    return rows


def build_sbc(registry, config, used, prior):
    specification = config["sbc"]
    rows = []
    for replication in range(specification["replications"]):
        target_index = int(sha256_text(
            f"v53-sbc-program|{config['population']['generatorSeed']}|{replication}"
        ), 16) % len(registry)
        target = registry[target_index]
        theta = record_theta(config, "sbc", replication)
        supports = []
        for ordinal in range(specification["supportEpisodesPerReplication"]):
            length = specification["supportSequenceLengths"][(replication + ordinal) % 2]
            case = make_case(
                "sbc", replication, ordinal, length, specification["entityCounts"],
                config, "support", used, prior,
            )
            observed, _ = observed_episode(
                target, theta, case, config, f"sbc|{replication}|support|{ordinal}"
            )
            supports.append(observed)
        pair = replication % len(specification["querySequenceLengths"])
        length = specification["querySequenceLengths"][pair]
        prefix = specification["queryPrefixLengths"][pair]
        case = make_case(
            "sbc", replication, 0, length, specification["entityCounts"],
            config, "query", used, prior,
        )
        query, sampled = observed_episode(
            target, theta, case, config, f"sbc|{replication}|query", prefix
        )
        query.update({
            "prefix_length": prefix,
            "true_configuration_key": _configuration_key(sampled["world"], sampled["queue"]),
        })
        rows.append({
            "id": f"sbc_{replication:05d}",
            "schema_version": 53,
            "population": "sbc",
            "replication": replication,
            **target_fields(target, target_index, theta),
            "supports": supports,
            "query": query,
        })
    return rows


def build_scale(registry, config, used, prior):
    specification = config["scaleStress"]
    rows = []
    for target_index, target in enumerate(registry):
        for replicate in range(specification["recordsPerTemplate"]):
            record_index = target_index * specification["recordsPerTemplate"] + replicate
            theta = record_theta(config, "scale", record_index)
            episodes = []
            for ordinal, length in enumerate(specification["sequenceLengths"]):
                case = make_case(
                    "scale", record_index, ordinal, length, specification["entityCounts"],
                    config, "episode", used, prior,
                )
                observed, sampled = observed_episode(
                    target, theta, case, config,
                    f"scale|{record_index}|episode|{ordinal}",
                )
                observed.update({
                    "prefix_length": length,
                    "true_configuration_key": _configuration_key(
                        sampled["world"], sampled["queue"]
                    ),
                })
                episodes.append(observed)
            rows.append({
                "id": f"scale_{record_index:05d}",
                "schema_version": 53,
                "population": "scale",
                "record": record_index,
                **target_fields(target, target_index, theta),
                "supports": episodes[:-1],
                "query": episodes[-1],
            })
    return rows


def build_populations(config):
    registry = mechanic_registry(config["population"]["templateSeed"])
    prior = prior_observation_design_keys()
    used = set()
    return {
        "exact": build_exact(registry, config, used, prior),
        "sbc": build_sbc(registry, config, used, prior),
        "scale": build_scale(registry, config, used, prior),
    }


def population_hash(populations):
    return sha256_text("".join(
        canonical_json(row) + "\n"
        for name in ("exact", "sbc", "scale") for row in populations[name]
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v53r2-implementation-lock.json"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_v53r2_populations"]:
        raise RuntimeError("V53r2 implementation lock does not authorize population construction")
    output = PROJECT_ROOT / "data/v53r2-continuous-parameter-smc2"
    if output.exists():
        raise RuntimeError("V53 populations already exist")
    populations = build_populations(lock["config_payload"])
    if population_hash(populations) != lock["expected_population_sha256"]:
        raise RuntimeError("V53 populations differ from implementation precommit")
    output.mkdir(parents=True)
    artifacts = {}
    for name, rows in populations.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        artifacts[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
            "records": len(rows),
        }
    manifest = {
        "schema_version": 53,
        "experiment": "v53r2_continuous_parameter_populations",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "population_sha256": population_hash(populations),
        "artifacts": artifacts,
        "data_access": {
            "smc_squared_evaluation_runs": 0,
            "pmcmc_reference_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
