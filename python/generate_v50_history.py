#!/usr/bin/env python3
"""Construct the sealed V50 history-dependent passive-belief population."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, Sequence

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import ONTOLOGY, action_bindings, atom_universe, deterministic_world, entities, epistemic_rows
from v46_stochastic import ACTIONS
from v47_sampling import sample_trajectory
from v49_belief import conditional_suffix_from_map, masked_trace, signature_world, trace_key, trajectory_map
from v50_belief import (
    decimal_map,
    fraction_rows,
    kl_divergence,
    latest_only_evidence,
    map_latent_predictive,
    mechanic_registry,
    time_shuffled_evidence,
    total_variation,
)


def make_actions(entity_rows, pattern, token):
    bindings = action_bindings(entity_rows)
    start = int(sha256_text(f"v50-binding|{token}")[:8], 16) % len(bindings)
    return [
        {"id": "wait", "binding": {}}
        if action == "wait"
        else {"id": action, "binding": dict(bindings[(start + index) % len(bindings)])}
        for index, action in enumerate(pattern)
    ]


def structural_key(entity_rows, state, actions):
    return sha256_text(canonical_json({"entities": entity_rows, "initial_state": state, "actions": actions}))


def base_case(index: int, config, prefix: str, length: int):
    population = config["population"]
    entity_count = population["entityCounts"][index % len(population["entityCounts"])]
    entity_rows = entities(entity_count)
    token = f"v50-{prefix}|{population['generatorSeed']}|{index}"
    world = deterministic_world(entity_rows, token)
    state = epistemic_rows(world)
    case_id = f"{prefix}_{sha256_text(token)[:16]}"
    return {
        "id": case_id,
        "entities": entity_rows,
        "initial_world": world,
        "initial_state": state,
        "sequence_length": length,
        "entity_count": entity_count,
    }


def support_mask_schedule(entity_rows, steps: int, token: str, mask_seed: int):
    """Known time-varying masks, fixed before any trajectory is sampled."""
    atoms = list(atom_universe(entity_rows))
    count = max(1, min(len(atoms) - 1, len(atoms) // 2))
    return [
        sorted(sorted(atoms, key=lambda atom: sha256_text(
            f"v50-support-mask|{mask_seed}|{token}|{step}|{atom}"
        ))[:count])
        for step in range(steps)
    ]


def support_case(index: int, config):
    population = config["population"]
    length = population["sequenceLengths"][index % len(population["sequenceLengths"])]
    patterns = [pattern for pattern in product(ACTIONS, repeat=length) if "pulse" in pattern and "route" in pattern]
    pattern = patterns[(index // len(population["sequenceLengths"])) % len(patterns)]
    case = base_case(index, config, "support", length)
    case["actions"] = make_actions(case["entities"], pattern, case["id"])
    case["masks"] = support_mask_schedule(
        case["entities"], length, case["id"], population["maskSeed"]
    )
    case["structural_key"] = structural_key(case["entities"], case["initial_state"], case["actions"])
    return case


def prior_structural_keys() -> set[str]:
    keys: set[str] = set()
    for split in ("development_fit", "development_evaluation"):
        v47_path = PROJECT_ROOT / f"data/v47-sampled-transition-estimation/{split}.jsonl"
        if v47_path.is_file():
            for line in v47_path.read_text().splitlines():
                record = json.loads(line)
                for row in record["agent_input"]["support_interventions"] + record["agent_input"]["queries"]:
                    keys.add(row["structural_key"])
        v48_path = PROJECT_ROOT / f"data/v48-stochastic-language-composition/{split}.jsonl"
        if v48_path.is_file():
            for line in v48_path.read_text().splitlines():
                record = json.loads(line)
                for group in (record["reference"]["support_interventions"], record["reference"]["queries"]):
                    keys.update(row["source_structural_key"] for row in group)
        v49_path = PROJECT_ROOT / f"data/v49-passive-partial-observation/{split}.jsonl"
        if v49_path.is_file():
            for line in v49_path.read_text().splitlines():
                record = json.loads(line)
                for row in record["agent_input"]["support_interventions"] + record["agent_input"]["queries"]:
                    keys.add(row["structural_key"])
    return keys


def observation_key_map(full: dict[str, Fraction], masks):
    result: dict[str, Fraction] = {}
    for key, mass in full.items():
        observed = trace_key(masked_trace(json.loads(key), masks))
        result[observed] = result.get(observed, Fraction(0)) + mass
    return result


def informative_supports(target, registry, pool, signatures, count: int):
    ranked = []
    for case in pool:
        target_signature, target_outcomes = signatures[(target["id"], case["id"])]
        if target_outcomes < 2:
            continue
        matches = sum(signatures[(candidate["id"], case["id"])][0] == target_signature for candidate in registry)
        ranked.append((matches, sha256_text(f"{target['id']}|{case['id']}"), case))
    selected = [row[2] for row in sorted(ranked)[:count]]
    if len(selected) != count:
        raise RuntimeError(f"V50 lacks informative supports for {target['id']}")
    return sorted(selected, key=lambda row: row["id"])


def outcome_catalog(values: Sequence[Any], prefix: str):
    catalog, reverse = {}, {}
    for value in values:
        key = canonical_json(value)
        identifier = f"{prefix}_{sha256_text(key)[:16]}"
        catalog[identifier] = value
        reverse[key] = identifier
    return catalog, reverse


def v50_trial_seed(sampling_seed: int, mechanic_id: str, case_id: str, trial: int):
    return int(sha256_text(f"v50-trial|{sampling_seed}|{mechanic_id}|{case_id}|{trial}"), 16)


def support_rows(mechanic, case, trials: int, sampling_seed: int):
    full = trajectory_map(mechanic["program"], case["entities"], case["initial_world"], case["actions"])
    trajectories = [json.loads(key) for key in full]
    full_catalog, full_reverse = outcome_catalog(trajectories, "trajectory")
    masked_values = [masked_trace(value, case["masks"]) for value in trajectories]
    masked_catalog, masked_reverse = outcome_catalog(masked_values, "trace")
    full_ids, masked_ids = [], []
    for trial in range(trials):
        trajectory = sample_trajectory(
            mechanic["program"], case["entities"], case["initial_world"], case["actions"],
            v50_trial_seed(sampling_seed, mechanic["id"], case["id"], trial),
        )
        full_ids.append(full_reverse[canonical_json(trajectory)])
        masked_ids.append(masked_reverse[canonical_json(masked_trace(trajectory, case["masks"]))])
    shared = {
        key: case[key]
        for key in ("id", "entities", "initial_state", "actions", "masks", "sequence_length", "entity_count", "structural_key")
    }
    return (
        {**shared, "masked_trace_catalog": masked_catalog, "realized_masked_trace_ids": masked_ids},
        {**shared, "full_trajectory_catalog": full_catalog, "realized_full_trajectory_ids": full_ids},
    )


def sample_key(distribution: dict[str, Fraction], seed: int):
    rng = random.Random(seed)
    draw = rng.random()
    cumulative = 0.0
    for key, probability in sorted(distribution.items()):
        cumulative += float(probability)
        if draw < cumulative:
            return key
    return sorted(distribution)[-1]


def compatible_full_distribution(full, evidence, prefix_length):
    selected = {
        key: mass
        for key, mass in full.items()
        if all(
            signature_world(json.loads(key)[step]).get(row["atom"]) is row["value"]
            for step, observed in enumerate(evidence)
            for row in observed
        )
    }
    total = sum(selected.values(), Fraction(0))
    return {key: value / total for key, value in selected.items()} if total else {}


def _invariant_atoms(trajectories, step: int):
    worlds = [signature_world(trajectory[step]) for trajectory in trajectories]
    return [atom for atom in sorted(worlds[0]) if len({world[atom] for world in worlds}) == 1]


def query_case(mechanic, ordinal: int, config, forbidden: set[str]):
    population = config["population"]
    delay = mechanic["delay"]
    stochastic_rule = next(
        rule for rule in mechanic["program"]["rules"]
        if rule["stochastic_immediate"] or rule["stochastic_delayed"]
    )
    trigger = stochastic_rule["action"]
    other = "route" if trigger == "pulse" else "pulse"
    modes = [{
        "name": "two_stochastic_updates",
        "prefix_length": 2 + delay,
        "informative_step": delay,
        "head": [trigger, trigger],
        "require_ambiguous_suffix": True,
    }]
    # A fair toggle followed by another fair toggle erases the first result.
    # The fallback observes one stochastic update after a deterministic setup
    # action; for delayed mechanics the waits cover delivery and one later
    # observation. Shuffling assigns the revealed fact to a pre-delivery time.
    modes.append({
        "name": "single_update_with_prestate_control",
        "prefix_length": 3 + delay,
        "informative_step": 1 + delay,
        "head": [other, trigger, *(["wait"] * (1 + delay))],
        "require_ambiguous_suffix": False,
    })
    stochastic_branch = (
        stochastic_rule["stochastic_immediate"] or stochastic_rule["stochastic_delayed"]
    )[0]
    if mechanic["probability"] == "1/2" and stochastic_branch["effect"]["op"] == "toggle":
        modes.reverse()

    for mode_index, mode in enumerate(modes):
        prefix_length = mode["prefix_length"]
        informative_step = mode["informative_step"]
        allowed_lengths = [value for value in population["sequenceLengths"] if value > prefix_length]
        if not allowed_lengths:
            continue
        length = allowed_lengths[ordinal % len(allowed_lengths)]
        for attempt in range(512):
            index = (
                50000 + mode_index * 500000 + ordinal
                + attempt * population["queryEpisodesPerMechanic"]
            )
            case = base_case(index, config, f"query-{mechanic['id']}-{ordinal}-{mode['name']}", length)
            bindings = action_bindings(case["entities"])
            binding = bindings[int(sha256_text(f"v50-query-binding|{case['id']}")[:8], 16) % len(bindings)]
            pattern = [*mode["head"], *(["wait"] * (length - len(mode["head"])))]
            case["actions"] = [
                {"id": "wait", "binding": {}}
                if action == "wait"
                else {"id": action, "binding": dict(binding)}
                for action in pattern
            ]
            case["structural_key"] = structural_key(case["entities"], case["initial_state"], case["actions"])
            if case["structural_key"] in forbidden:
                continue
            full = trajectory_map(mechanic["program"], case["entities"], case["initial_world"], case["actions"])
            if len(full) < 2:
                continue
            trajectories = [json.loads(key) for key in full]
            worlds = [signature_world(trajectory[informative_step]) for trajectory in trajectories]
            variable_atoms = [
                atom for atom in sorted(worlds[0]) if len({world[atom] for world in worlds}) > 1
            ]
            invariant_atoms = _invariant_atoms(trajectories, prefix_length - 1)
            for atom in variable_atoms:
                latest_atom = next((candidate for candidate in invariant_atoms if candidate != atom), None)
                if latest_atom is None:
                    continue
                masks = [[] for _ in range(length)]
                masks[informative_step] = [atom]
                masks[prefix_length - 1] = [latest_atom]
                # Qualify the schedule over every observation it can emit before
                # drawing the anchor. This makes the public masks independent of
                # realized values while guaranteeing a genuinely informative task.
                histories = {}
                schedule_valid = True
                for trajectory in trajectories:
                    candidate_evidence = masked_trace(trajectory, masks)[:prefix_length]
                    evidence_key = canonical_json(candidate_evidence)
                    if evidence_key in histories:
                        continue
                    evidence_mass, truth_fraction = conditional_suffix_from_map(
                        full, candidate_evidence, prefix_length
                    )
                    latest = latest_only_evidence(candidate_evidence)
                    latest_mass, latest_fraction = conditional_suffix_from_map(full, latest, prefix_length)
                    shuffled = time_shuffled_evidence(candidate_evidence, informative_step)
                    shuffled_mass, shuffled_fraction = conditional_suffix_from_map(full, shuffled, prefix_length)
                    if not evidence_mass or not latest_mass:
                        schedule_valid = False
                        break
                    truth = decimal_map(truth_fraction)
                    latest_truth = decimal_map(latest_fraction)
                    shuffled_truth = decimal_map(shuffled_fraction) if shuffled_mass else {}
                    oracle_tv = total_variation(truth, latest_truth)
                    history_kl = kl_divergence(truth, latest_truth)
                    shuffled_kl = kl_divergence(truth, shuffled_truth)
                    if oracle_tv < 0.10 or history_kl < 0.05 or shuffled_kl < 0.05:
                        schedule_valid = False
                        break
                    histories[evidence_key] = {
                        "truth": truth_fraction,
                        "latest_truth": latest_fraction,
                        "shuffled_truth": shuffled_fraction,
                        "latest": latest,
                        "shuffled": shuffled,
                        "oracle_tv": oracle_tv,
                        "history_kl": history_kl,
                        "shuffled_kl": shuffled_kl,
                    }
                if not schedule_valid or not histories:
                    continue
                anchor_key = sample_key(
                    full, v50_trial_seed(population["samplingSeed"], mechanic["id"], case["id"], -1)
                )
                anchor = json.loads(anchor_key)
                evidence = masked_trace(anchor, masks)[:prefix_length]
                selected = histories[canonical_json(evidence)]
                truth_fraction = selected["truth"]
                latest_fraction = selected["latest_truth"]
                shuffled_fraction = selected["shuffled_truth"]
                latest = selected["latest"]
                shuffled = selected["shuffled"]
                truth = decimal_map(truth_fraction)
                collapsed = map_latent_predictive(
                    [mechanic], [Decimal(1)], case["entities"], case["initial_world"], case["actions"],
                    evidence, prefix_length,
                )
                oracle_tv = selected["oracle_tv"]
                history_kl = selected["history_kl"]
                shuffled_kl = selected["shuffled_kl"]
                collapsed_kl = kl_divergence(truth, collapsed)
                case["masks"] = masks
                case["informative_step"] = informative_step
                case["earlier_evidence_distance"] = prefix_length - 1 - informative_step
                case["construction_mode"] = mode["name"]
                forbidden.add(case["structural_key"])
                return {
                    "case": case,
                    "prefix_length": prefix_length,
                    "evidence": evidence,
                    "latest_evidence": latest,
                    "shuffled_evidence": shuffled,
                    "anchor": anchor,
                    "full": full,
                    "truth": truth_fraction,
                    "latest_truth": latest_fraction,
                    "shuffled_truth": shuffled_fraction,
                    "compatible": compatible_full_distribution(full, evidence, prefix_length),
                    "oracle_tv": oracle_tv,
                    "history_kl": history_kl,
                    "shuffled_kl": shuffled_kl,
                    "collapsed_kl": collapsed_kl,
                }
    raise RuntimeError(f"V50 lacks history-dependent query {mechanic['id']}/{ordinal}")


def query_rows(mechanic, ordinal, config, forbidden):
    built = query_case(mechanic, ordinal, config, forbidden)
    case = built["case"]
    prefix_length = built["prefix_length"]
    population = config["population"]
    catalog, reverse = outcome_catalog([json.loads(key) for key in built["compatible"]], "full")
    heldout_ids = []
    for trial in range(population["heldoutConditionalContinuationsPerQuery"]):
        key = sample_key(
            built["compatible"],
            v50_trial_seed(population["samplingSeed"], mechanic["id"], case["id"], 1000 + trial),
        )
        heldout_ids.append(reverse[key])
    public = {
        key: case[key]
        for key in (
            "id", "entities", "initial_state", "actions", "masks", "sequence_length", "entity_count",
            "structural_key", "informative_step", "earlier_evidence_distance", "construction_mode",
        )
    }
    public.update({
        "query_prefix_length": prefix_length,
        "masked_prefix_observations": built["evidence"],
        "latest_only_prefix_observations": built["latest_evidence"],
        "time_shuffled_prefix_observations": built["shuffled_evidence"],
    })
    oracle = {
        "id": case["id"],
        "anchor_full_prefix": built["anchor"][:prefix_length],
        "true_complete_history_conditional_suffix_distribution": fraction_rows(built["truth"]),
        "true_latest_only_conditional_suffix_distribution": fraction_rows(built["latest_truth"]),
        "true_time_shuffled_conditional_suffix_distribution": fraction_rows(built["shuffled_truth"]),
        "compatible_full_trajectory_catalog": catalog,
        "heldout_full_trajectory_ids": heldout_ids,
        "oracle_full_history_vs_latest_only_tv": built["oracle_tv"],
        "oracle_history_value_kl_nats": built["history_kl"],
        "oracle_time_shuffled_kl_nats": built["shuffled_kl"],
        "oracle_map_collapse_kl_nats": built["collapsed_kl"],
    }
    return public, oracle


def build_population(config):
    registry = mechanic_registry()
    population = config["population"]
    forbidden_prior = prior_structural_keys()
    pool = [support_case(index, config) for index in range(768)]
    unique_pool = {}
    for case in pool:
        if case["structural_key"] not in forbidden_prior:
            unique_pool.setdefault(case["structural_key"], case)
    pool = list(unique_pool.values())
    signatures = {}
    for candidate in registry:
        for case in pool:
            distribution = observation_key_map(
                trajectory_map(candidate["program"], case["entities"], case["initial_world"], case["actions"]),
                case["masks"],
            )
            signatures[(candidate["id"], case["id"])] = (
                canonical_json(fraction_rows(distribution, "observation")), len(distribution)
            )

    rows = []
    for mechanic in registry:
        support_cases = informative_supports(
            mechanic, registry, pool, signatures, population["supportInterventionsPerMechanic"]
        )
        partial_supports, full_supports = [], []
        for case in support_cases:
            partial, full = support_rows(
                mechanic, case, population["realizedTrialsPerSupportIntervention"], population["samplingSeed"]
            )
            partial_supports.append(partial)
            full_supports.append(full)
        forbidden = forbidden_prior | {row["structural_key"] for row in partial_supports}
        queries, oracle_queries = [], []
        for ordinal in range(population["queryEpisodesPerMechanic"]):
            query, oracle = query_rows(mechanic, ordinal, config, forbidden)
            queries.append(query)
            oracle_queries.append(oracle)
        split = "development_fit" if mechanic["ordinal"] < 6 else "development_evaluation"
        rows.append({
            "id": mechanic["id"],
            "schema_version": 50,
            "split": split,
            "construction_family": mechanic["family"],
            "agent_input": {
                "task": "infer_history_dependent_joint_belief_over_hidden_world_queue_program_and_probability",
                "ontology": ONTOLOGY,
                "probability_vocabulary": ["1/4", "1/2", "3/4"],
                "support_interventions": partial_supports,
                "queries": queries,
            },
            "reference": {"matched_fully_observed_support_interventions": full_supports},
            "target": {"program": mechanic["program"], "program_key": mechanic["key"]},
            "oracle_queries": oracle_queries,
            "oracle_metadata": {
                "family_ordinal": mechanic["ordinal"],
                "probability": mechanic["probability"],
                "timing": mechanic["timing"],
                "delay": mechanic["delay"],
                "support_interventions": len(partial_supports),
                "support_trials": len(partial_supports) * population["realizedTrialsPerSupportIntervention"],
                "queries": len(queries),
                "heldout_conditional_continuations": len(queries)
                * population["heldoutConditionalContinuationsPerQuery"],
            },
        })
    keys = {row["target"]["program_key"] for row in rows}
    support_keys = {item["structural_key"] for row in rows for item in row["agent_input"]["support_interventions"]}
    query_keys = {item["structural_key"] for row in rows for item in row["agent_input"]["queries"]}
    if len(rows) != 48 or len(keys) != 48 or support_keys & query_keys:
        raise RuntimeError("V50 population is not fresh or support/query disjoint")
    if (support_keys | query_keys) & forbidden_prior:
        raise RuntimeError("V50 structural case overlaps V47 through V49")
    return sorted(rows, key=lambda row: row["id"])


def corpus_hash(rows: Sequence[dict[str, Any]]):
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v50-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_development_population"]:
        raise RuntimeError("V50 implementation lock does not authorize construction")
    output = PROJECT_ROOT / "data/v50-history-dependent-belief-filtering"
    if output.exists():
        raise RuntimeError("V50 corpus already exists")
    rows = build_population(lock["config_payload"])
    output.mkdir(parents=True)
    artifacts = {}
    for split in ("development_fit", "development_evaluation"):
        path = output / f"{split}.jsonl"
        selected = [row for row in rows if row["split"] == split]
        path.write_text("".join(canonical_json(row) + "\n" for row in selected))
        artifacts[split] = {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path), "records": len(selected)}
    counts = {
        "mechanics": len(rows),
        "support_interventions": sum(len(row["agent_input"]["support_interventions"]) for row in rows),
        "support_trials": sum(
            len(item["realized_masked_trace_ids"])
            for row in rows for item in row["agent_input"]["support_interventions"]
        ),
        "queries": sum(len(row["agent_input"]["queries"]) for row in rows),
        "heldout_conditional_continuations": sum(
            len(item["heldout_full_trajectory_ids"])
            for row in rows for item in row["oracle_queries"]
        ),
    }
    manifest = {
        "schema_version": 50,
        "experiment": "v50_history_dependent_belief_filtering_population",
        "corpus_sha256": corpus_hash(rows),
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifacts": artifacts,
        "counts": counts,
        "data_access": {
            "development_runs": 0,
            "support_realizations_constructed": counts["support_trials"],
            "heldout_realizations_constructed": counts["heldout_conditional_continuations"],
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
