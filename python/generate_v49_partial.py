#!/usr/bin/env python3
"""Construct the sealed V49 passive-partial-observation population."""
from __future__ import annotations

import argparse
import json
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
from v49_belief import (
    conditional_suffix_from_map,
    decimal_map,
    fraction_rows,
    full_evidence,
    masked_trace,
    map_latent_predictive,
    mechanic_registry,
    trace_key,
    trajectory_map,
)


def make_actions(entity_rows, pattern, token):
    bindings = action_bindings(entity_rows)
    start = int(sha256_text(f"v49-binding|{token}")[:8], 16) % len(bindings)
    rows = []
    for index, action in enumerate(pattern):
        rows.append(
            {"id": "wait", "binding": {}}
            if action == "wait"
            else {"id": action, "binding": dict(bindings[(start + index) % len(bindings)])}
        )
    return rows


def structural_key(entity_rows, state, actions):
    return sha256_text(canonical_json({"entities": entity_rows, "initial_state": state, "actions": actions}))


def mask_schedule(entity_rows, steps: int, visible_fraction: float, token: str, mask_seed: int):
    atoms = list(atom_universe(entity_rows))
    count = max(1, min(len(atoms) - 1, round(len(atoms) * visible_fraction)))
    result = []
    for step in range(steps):
        ranked = sorted(atoms, key=lambda atom: sha256_text(f"v49-mask|{mask_seed}|{token}|{step}|{atom}"))
        result.append(sorted(ranked[:count]))
    return result


def intervention(index: int, config, prefix: str):
    population = config["population"]
    length = population["sequenceLengths"][index % len(population["sequenceLengths"])]
    patterns = [pattern for pattern in product(ACTIONS, repeat=length) if "pulse" in pattern and "route" in pattern]
    pattern = patterns[(index // len(population["sequenceLengths"])) % len(patterns)]
    entity_count = population["entityCounts"][(index // 3) % len(population["entityCounts"])]
    entity_rows = entities(entity_count)
    token = f"v49-{prefix}|{population['generatorSeed']}|{index}"
    world = deterministic_world(entity_rows, token)
    state = epistemic_rows(world)
    actions = make_actions(entity_rows, pattern, token)
    case_id = f"{prefix}_{sha256_text(token)[:16]}"
    visible_fraction = population["visibleFractions"][(index // 9) % len(population["visibleFractions"])]
    masks = mask_schedule(entity_rows, length, visible_fraction, case_id, population["maskSeed"])
    return {
        "id": case_id,
        "entities": entity_rows,
        "initial_world": world,
        "initial_state": state,
        "actions": actions,
        "masks": masks,
        "visible_fraction": visible_fraction,
        "sequence_length": length,
        "entity_count": entity_count,
        "structural_key": structural_key(entity_rows, state, actions),
    }


def prior_structural_keys() -> set[str]:
    keys: set[str] = set()
    for split in ("development_fit", "development_evaluation"):
        v47 = PROJECT_ROOT / f"data/v47-sampled-transition-estimation/{split}.jsonl"
        if v47.is_file():
            for line in v47.read_text().splitlines():
                record = json.loads(line)
                for row in record["agent_input"]["support_interventions"] + record["agent_input"]["queries"]:
                    keys.add(row["structural_key"])
        v48 = PROJECT_ROOT / f"data/v48-stochastic-language-composition/{split}.jsonl"
        if v48.is_file():
            for line in v48.read_text().splitlines():
                record = json.loads(line)
                for group in (record["reference"]["support_interventions"], record["reference"]["queries"]):
                    keys.update(row["source_structural_key"] for row in group)
    return keys


def observation_key_map(full: dict[str, Fraction], masks):
    result: dict[str, Fraction] = {}
    for key, mass in full.items():
        observed = trace_key(masked_trace(json.loads(key), masks))
        result[observed] = result.get(observed, Fraction(0)) + mass
    return result


def informative_supports(target, registry, pool, signatures, count: int):
    by_visibility = {value: [] for value in (0.25, 0.5, 0.75)}
    for case in pool:
        target_signature, target_outcomes = signatures[(target["id"], case["id"])]
        if target_outcomes < 2:
            continue
        matches = sum(signatures[(candidate["id"], case["id"])][0] == target_signature for candidate in registry)
        by_visibility[case["visible_fraction"]].append(
            (matches, sha256_text(f"{target['id']}|{case['id']}"), case)
        )
    selected = []
    quota = count // len(by_visibility)
    for visibility in sorted(by_visibility):
        selected.extend(row[2] for row in sorted(by_visibility[visibility])[:quota])
    if len(selected) != count:
        raise RuntimeError(f"V49 lacks informative balanced supports for {target['id']}")
    return sorted(selected, key=lambda row: row["id"])


def outcome_catalog(values: Sequence[Any], prefix: str):
    catalog = {}
    reverse = {}
    for value in values:
        key = canonical_json(value)
        identifier = f"{prefix}_{sha256_text(key)[:16]}"
        catalog[identifier] = value
        reverse[key] = identifier
    return catalog, reverse


def v49_trial_seed(sampling_seed: int, mechanic_id: str, case_id: str, trial: int):
    return int(sha256_text(f"v49-trial|{sampling_seed}|{mechanic_id}|{case_id}|{trial}"), 16)


def support_rows(mechanic, case, trials: int, sampling_seed: int):
    full = trajectory_map(mechanic["program"], case["entities"], case["initial_world"], case["actions"])
    trajectories = [json.loads(key) for key in full]
    full_catalog, full_reverse = outcome_catalog(trajectories, "trajectory")
    masked_values = [masked_trace(value, case["masks"]) for value in trajectories]
    masked_catalog, masked_reverse = outcome_catalog(masked_values, "trace")
    full_ids = []
    masked_ids = []
    for trial in range(trials):
        trajectory = sample_trajectory(
            mechanic["program"], case["entities"], case["initial_world"], case["actions"],
            v49_trial_seed(sampling_seed, mechanic["id"], case["id"], trial),
        )
        full_ids.append(full_reverse[canonical_json(trajectory)])
        masked_ids.append(masked_reverse[canonical_json(masked_trace(trajectory, case["masks"]))])
    shared = {
        key: case[key]
        for key in (
            "id", "entities", "initial_state", "actions", "masks", "visible_fraction",
            "sequence_length", "entity_count", "structural_key",
        )
    }
    partial = {
        **shared,
        "masked_trace_catalog": masked_catalog,
        "realized_masked_trace_ids": masked_ids,
    }
    full_reference = {
        **shared,
        "full_trajectory_catalog": full_catalog,
        "realized_full_trajectory_ids": full_ids,
    }
    return partial, full_reference


def sample_key(distribution: dict[str, Fraction], seed: int):
    rng = random.Random(seed)
    draw = rng.random()
    cumulative = 0.0
    for key, probability in sorted(distribution.items()):
        cumulative += float(probability)
        if draw < cumulative:
            return key
    return sorted(distribution)[-1]


def conditional_full_distribution(full, evidence, prefix_length):
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


def signature_world(signature):
    return {row["atom"]: row["value"] for row in json.loads(signature)}


def query_case(mechanic, ordinal: int, config, forbidden: set[str]):
    population = config["population"]
    prefix_length = population["queryEvidencePrefixLengths"][ordinal // 8]
    visible_fraction = population["visibleFractions"][ordinal % 3]
    length = population["sequenceLengths"][(ordinal // 3) % 3]
    stochastic_rule = next(
        rule for rule in mechanic["program"]["rules"]
        if rule["stochastic_immediate"] or rule["stochastic_delayed"]
    )
    trigger = stochastic_rule["action"]
    other = "route" if trigger == "pulse" else "pulse"
    delay = (stochastic_rule["stochastic_delayed"] or [{}])[0].get("delay", 0)
    for attempt in range(8192):
        index = 20000 + ordinal + attempt * 24
        case = intervention(index, config, f"query-{mechanic['id']}-{ordinal}")
        case["visible_fraction"] = visible_fraction
        pattern = [trigger]
        for step in range(1, length):
            pattern.append("wait" if delay and step <= delay else other)
        bindings = action_bindings(case["entities"])
        binding = bindings[int(sha256_text(f"v49-query-binding|{case['id']}")[:8], 16) % len(bindings)]
        case["actions"] = [
            {"id": "wait", "binding": {}} if action == "wait" else {"id": action, "binding": dict(binding)}
            for action in pattern
        ]
        case["sequence_length"] = length
        case["masks"] = mask_schedule(
            case["entities"], length, visible_fraction, case["id"], population["maskSeed"]
        )
        case["structural_key"] = structural_key(case["entities"], case["initial_state"], case["actions"])
        if case["structural_key"] in forbidden:
            continue
        full = trajectory_map(mechanic["program"], case["entities"], case["initial_world"], case["actions"])
        anchor_key = sample_key(
            full,
            v49_trial_seed(population["samplingSeed"], mechanic["id"], case["id"], -1),
        )
        anchor = json.loads(anchor_key)
        evidence = masked_trace(anchor, case["masks"])[:prefix_length]
        evidence_mass, suffix = conditional_suffix_from_map(full, evidence, prefix_length)
        if not evidence_mass or len(suffix) < 2:
            continue
        compatible = conditional_full_distribution(full, evidence, prefix_length)
        if len(compatible) < 2:
            continue
        collapsed = map_latent_predictive(
            [mechanic], [Decimal(1)], case["entities"], case["initial_world"], case["actions"],
            evidence, prefix_length,
        )
        truth_decimal = decimal_map(suffix)
        collapsed_tv = sum(
            abs(collapsed.get(key, Decimal(0)) - truth_decimal.get(key, Decimal(0)))
            for key in set(collapsed) | set(truth_decimal)
        ) / 2
        require_hidden_influence = visible_fraction < 0.75 or ordinal in (2, 5)
        if require_hidden_influence and collapsed_tv <= Decimal("1e-6"):
            continue
        forbidden.add(case["structural_key"])
        return case, prefix_length, evidence, anchor, full, compatible, suffix
    raise RuntimeError(f"V49 lacks ambiguous query {mechanic['id']}/{ordinal}")


def query_rows(mechanic, ordinal, config, forbidden):
    case, prefix_length, evidence, anchor, full, compatible, suffix = query_case(
        mechanic, ordinal, config, forbidden
    )
    population = config["population"]
    catalog, reverse = outcome_catalog([json.loads(key) for key in compatible], "full")
    heldout_ids = []
    for trial in range(population["heldoutConditionalContinuationsPerQuery"]):
        key = sample_key(
            compatible,
            v49_trial_seed(population["samplingSeed"], mechanic["id"], case["id"], 1000 + trial),
        )
        heldout_ids.append(reverse[key])
    public = {
        key: case[key]
        for key in (
            "id", "entities", "initial_state", "actions", "masks", "visible_fraction",
            "sequence_length", "entity_count", "structural_key",
        )
    }
    public.update({
        "query_prefix_length": prefix_length,
        "masked_prefix_observations": evidence,
    })
    oracle = {
        "id": case["id"],
        "anchor_full_prefix": anchor[:prefix_length],
        "true_partial_conditional_suffix_distribution": fraction_rows(suffix),
        "compatible_full_trajectory_catalog": catalog,
        "heldout_full_trajectory_ids": heldout_ids,
    }
    evidence_mass, _ = conditional_suffix_from_map(full, evidence, prefix_length)
    oracle["partial_evidence_mass"] = {
        "numerator": evidence_mass.numerator,
        "denominator": evidence_mass.denominator,
    }
    return public, oracle


def build_population(config):
    registry = mechanic_registry()
    population = config["population"]
    forbidden_prior = prior_structural_keys()
    pool = [intervention(index, config, "support") for index in range(432)]
    pool = [case for case in pool if case["structural_key"] not in forbidden_prior]
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
            "schema_version": 49,
            "split": split,
            "construction_family": mechanic["family"],
            "agent_input": {
                "task": "infer_a_joint_belief_over_hidden_state_stochastic_program_and_probability_from_passively_masked_trials",
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
                "support_interventions": len(partial_supports),
                "support_trials": len(partial_supports) * population["realizedTrialsPerSupportIntervention"],
                "queries": len(queries),
                "heldout_conditional_continuations": len(queries)
                * population["heldoutConditionalContinuationsPerQuery"],
            },
        })
    keys = {row["target"]["program_key"] for row in rows}
    support_keys = {
        item["structural_key"] for row in rows for item in row["agent_input"]["support_interventions"]
    }
    query_keys = {item["structural_key"] for row in rows for item in row["agent_input"]["queries"]}
    if len(rows) != 48 or len(keys) != 48 or support_keys & query_keys:
        raise RuntimeError("V49 population is not fresh or support/query disjoint")
    if (support_keys | query_keys) & forbidden_prior:
        raise RuntimeError("V49 structural case overlaps V47 or V48")
    return sorted(rows, key=lambda row: row["id"])


def corpus_hash(rows: Sequence[dict[str, Any]]):
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v49-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_development_population"]:
        raise RuntimeError("V49 implementation lock does not authorize construction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V49 implementation changed: {path}")
    output = PROJECT_ROOT / "data/v49-passive-partial-observation"
    if output.exists():
        raise RuntimeError("V49 population already exists")
    rows = build_population(lock["config_payload"])
    if corpus_hash(rows) != lock["expected_corpus_sha256"]:
        raise RuntimeError("V49 corpus differs from implementation lock")
    output.mkdir(parents=True)
    artifacts = {}
    for split in ("development_fit", "development_evaluation"):
        selected = [row for row in rows if row["split"] == split]
        path = output / f"{split}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in selected))
        artifacts[split] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "records": len(selected),
            "sha256": file_sha256(path),
        }
    counts = {
        "mechanics": len(rows),
        "support_interventions": sum(row["oracle_metadata"]["support_interventions"] for row in rows),
        "support_trials": sum(row["oracle_metadata"]["support_trials"] for row in rows),
        "queries": sum(row["oracle_metadata"]["queries"] for row in rows),
        "heldout_conditional_continuations": sum(
            row["oracle_metadata"]["heldout_conditional_continuations"] for row in rows
        ),
        "families": dict(Counter(row["construction_family"] for row in rows)),
        "probabilities": dict(Counter(row["oracle_metadata"]["probability"] for row in rows)),
        "visibility": dict(Counter(
            str(query["visible_fraction"])
            for row in rows for query in row["agent_input"]["queries"]
        )),
        "splits": dict(Counter(row["split"] for row in rows)),
    }
    manifest = {
        "schema_version": 49,
        "experiment": lock["config_payload"]["experiment"],
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
