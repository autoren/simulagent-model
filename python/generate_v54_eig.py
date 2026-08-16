#!/usr/bin/env python3
"""Construct the sealed V54 selection and adaptive-SBC populations."""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import atom_universe, deterministic_world, entities, epistemic_rows
from v46_stochastic import _configuration_key
from v49_belief import _configuration_key_with_history, masked_trace
from v53_smc2 import (
    continuous_advance_configurations,
    continuous_unit_transition,
    exact_inference,
    instantiate_program,
    mechanic_registry,
    scaled_beta_sample,
    stream_seed,
)
from v54_eig import (
    assert_selection_payload_is_public,
    belief_atoms_from_exact,
    candidate_interventions,
    score_all_interventions,
    select_score,
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
    for version in range(49, 54):
        for path in sorted((PROJECT_ROOT / "data").glob(f"v{version}*/*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    _visit_designs(json.loads(line), result)
    return result


def history_class_for_record(record: int):
    within_template = record % 8
    if within_template < 2:
        return "prior_like_all_wait"
    if within_template < 6:
        return "mixed_informative"
    return "pending_delayed_event"


def selection_population_class_counts():
    counts = {
        "prior_like_all_wait": 0,
        "mixed_informative": 0,
        "pending_delayed_event": 0,
    }
    for record in range(64):
        counts[history_class_for_record(record)] += 1
    return counts


def mask_schedule(entity_rows, steps, token, mask_seed):
    atoms = list(atom_universe(entity_rows))
    count = max(1, min(len(atoms) - 1, len(atoms) // 2))
    return [
        sorted(sorted(atoms, key=lambda atom: sha256_text(
            f"v54-mask|{mask_seed}|{token}|{tick}|{atom}"
        ))[:count])
        for tick in range(steps)
    ]


def action_schedule(entity_rows, steps, token, history_class, query=False):
    bindings = [row["action"]["binding"] for row in candidate_interventions(entity_rows)
                if row["action"]["id"] == "pulse"]
    if history_class == "prior_like_all_wait":
        return [{"id": "wait", "binding": {}} for _ in range(steps)]
    result = []
    for tick in range(steps):
        action_id = ("pulse", "route", "wait")[
            int(sha256_text(f"v54-action|{token}|{tick}")[:12], 16) % 3
        ]
        if action_id == "wait":
            result.append({"id": "wait", "binding": {}})
        else:
            index = int(sha256_text(f"v54-binding|{token}|{tick}")[:12], 16) % len(bindings)
            result.append({"id": action_id, "binding": dict(bindings[index])})
    if steps:
        result[0] = {"id": "pulse", "binding": dict(bindings[0])}
    if steps > 1 and history_class == "mixed_informative":
        result[1] = {"id": "route", "binding": dict(bindings[-1])}
    if query and steps and history_class == "pending_delayed_event":
        result[-1] = {"id": "pulse", "binding": dict(bindings[-1])}
    return result


def make_case(population, record, ordinal, length, entity_count, history_class,
              config, kind, used, prior, query=False):
    nonce = 0
    while True:
        token = (
            f"v54|{config['population']['generatorSeed']}|{population}|{kind}|"
            f"{record}|{ordinal}|{history_class}|{nonce}"
        )
        entity_rows = entities(entity_count)
        world = deterministic_world(entity_rows, token)
        row = {
            "id": f"{kind}_{sha256_text(token)[:16]}",
            "entities": entity_rows,
            "initial_world": world,
            "initial_state": epistemic_rows(world),
            "actions": action_schedule(
                entity_rows, length, token, history_class, query=query
            ),
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


def simulate_case(target, theta, case, steps, config, token):
    program = instantiate_program(target["template"], theta)
    world, queue, history = dict(case["initial_world"]), [], []
    for tick, action in enumerate(case["actions"][:steps]):
        branches = continuous_unit_transition(
            program, case["entities"], world, queue, action, tick
        )
        selected = sample_branch(branches, stream_seed(
            config["population"]["trajectorySeed"], token, tick, "history"
        ))
        world, queue = dict(selected["world"]), list(selected["queue"])
        history.append(selected["history"][-1])
    return {"world": world, "queue": queue, "history": history}


def public_case(case, sampled, steps=None):
    steps = case["sequence_length"] if steps is None else steps
    return {
        key: case[key]
        for key in (
            "id", "entities", "initial_state", "actions", "masks",
            "sequence_length", "entity_count", "observation_design_key",
        )
    } | {
        "actions": case["actions"][:steps],
        "masks": case["masks"][:steps],
        "sequence_length": steps,
        "observations": masked_trace(sampled["history"], case["masks"][:steps]),
    }


def record_theta(config, population, record):
    return scaled_beta_sample(stream_seed(
        config["population"]["thetaPriorSeed"], population, record, "theta-prior"
    ))


def target_index_for_record(population, record, registry, config):
    if population == "selection":
        return record // config["population"]["recordsPerGeneratingTemplate"]
    return int(sha256_text(
        f"v54-sbc-program|{config['population']['templateAssignmentSeed']}|{record}"
    ), 16) % len(registry)


def build_history(population, record, registry, config, used, prior):
    history_class = (
        history_class_for_record(record) if population == "selection"
        else history_class_for_record(record % 64)
    )
    target_index = target_index_for_record(population, record, registry, config)
    target = registry[target_index]
    theta = record_theta(config, population, record)
    entity_counts = config["population"]["entityCounts"]
    entity_count = entity_counts[record % len(entity_counts)]
    supports = []
    for ordinal in range(config["population"]["supportEpisodesPerRecord"]):
        lengths = config["population"]["supportSequenceLengths"]
        length = lengths[(record + ordinal) % len(lengths)]
        case = make_case(
            population, record, ordinal, length, entity_count, history_class,
            config, "support", used, prior,
        )
        sampled = simulate_case(
            target, theta, case, length, config,
            f"{population}|{record}|support|{ordinal}",
        )
        supports.append(public_case(case, sampled))
    prefix_lengths = config["population"]["ongoingHistoryPrefixLengths"]
    prefix = prefix_lengths[record % len(prefix_lengths)]
    case = make_case(
        population, record, 0, prefix, entity_count, history_class,
        config, "query", used, prior, query=True,
    )
    sampled = simulate_case(
        target, theta, case, prefix, config, f"{population}|{record}|query"
    )
    query = public_case(case, sampled, prefix)
    query["prefix_length"] = prefix
    public_history = {"supports": supports, "query": query}
    assert_selection_payload_is_public(public_history)
    return {
        "id": f"{population}_{record:05d}",
        "schema_version": 54,
        "population": population,
        "record": record,
        "history_class": history_class,
        "public_history": public_history,
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


def inference_record(public_history):
    return {
        "supports": public_history["supports"],
        "query": public_history["query"],
    }


def simulate_selected_outcome(row, registry, intervention, config):
    truth, query = row["truth"], row["public_history"]["query"]
    target = registry[truth["target_program_index"]]
    program = instantiate_program(target["template"], truth["target_theta"])
    world, queue, history = (
        dict(truth["query_world"]), list(truth["query_queue"]), []
    )
    start_tick = query["prefix_length"]
    for offset, action in enumerate(intervention["assay"]):
        local = {
            _configuration_key_with_history(world, queue, []): {
                "world": dict(world), "queue": list(queue),
                "history": [], "mass": Fraction(1),
            }
        }
        branches = continuous_advance_configurations(
            program, query["entities"], local, [action], start_tick + offset
        )
        selected = sample_branch(branches, stream_seed(
            config["population"]["outcomeSeed"], row["id"],
            intervention["key"], offset, "selected-outcome",
        ))
        world, queue = dict(selected["world"]), list(selected["queue"])
        history.append(selected["history"][-1])
    masks = [list(atom_universe(query["entities"])) for _ in history]
    return {
        "observations": masked_trace(history, masks),
        "masks": masks,
        "final_configuration_key": _configuration_key(world, queue),
    }


def attach_adaptive_selection_and_outcome(row, registry, v53_config, v54_config):
    exact = exact_inference(
        registry, inference_record(row["public_history"]), v53_config
    )
    atoms = belief_atoms_from_exact(exact)
    query = row["public_history"]["query"]
    scores = score_all_interventions(
        atoms, registry, query["entities"], query["prefix_length"]
    )
    selected = select_score(
        scores, v54_config["targetAndObjective"]["tieToleranceNats"]
    )
    intervention = next(
        candidate for candidate in candidate_interventions(query["entities"])
        if candidate["key"] == selected["selected"]["intervention_key"]
    )
    outcome = simulate_selected_outcome(row, registry, intervention, v54_config)
    row["selected_intervention"] = intervention
    row["realized_outcome"] = outcome
    return row


def build_populations(v54_config, v53_config):
    registry = mechanic_registry(5303)
    used, prior = set(), prior_observation_design_keys()
    selection = [
        build_history("selection", record, registry, v54_config, used, prior)
        for record in range(v54_config["population"]["selectionRecords"])
    ]
    adaptive = []
    for record in range(v54_config["adaptiveSbc"]["replications"]):
        row = build_history(
            "adaptive_sbc", record, registry, v54_config, used, prior
        )
        adaptive.append(attach_adaptive_selection_and_outcome(
            row, registry, v53_config, v54_config
        ))
    return {"selection": selection, "adaptive_sbc": adaptive}


def population_hash(populations):
    return sha256_text("".join(
        canonical_json(row) + "\n"
        for name in ("selection", "adaptive_sbc")
        for row in populations[name]
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v54-implementation-lock.json")
    parser.add_argument("--output-dir", default="data/v54-exact-one-step-eig")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_v54_active_populations"]:
        raise RuntimeError("V54 implementation lock does not authorize population construction")
    for relative, digest in lock["implementation_files_sha256"].items():
        if file_sha256(PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"V54 frozen implementation changed: {relative}")
    design = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())
    v54_config = design["config_payload"]
    v53_config = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    populations = build_populations(v54_config, v53_config)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in populations.items():
        (output_dir / f"{name}.jsonl").write_text(
            "".join(canonical_json(row) + "\n" for row in rows)
        )
    manifest = {
        "schema_version": 54,
        "experiment": "v54_population_manifest",
        "counts": {name: len(rows) for name, rows in populations.items()},
        "files": {
            name: {
                "path": str((output_dir / f"{name}.jsonl").relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(output_dir / f"{name}.jsonl"),
            }
            for name in populations
        },
        "population_hash": population_hash(populations),
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
