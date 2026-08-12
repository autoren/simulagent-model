"""Leak-resistant language views and graph utilities for V22r2.

V22r2 is derived from the immutable V22 oracle corpus.  This module owns only the public
language/interface layer: opaque identifiers, deterministic permutations, controlled query
counterfactuals, surface banks, and conversion between language-grounding predictions and V22
epistemic graphs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from v22_relational import (
    canonical_json,
    canonical_state_hash,
    epistemic_from_world,
    epistemic_rows,
    execute_partial,
    parse_atom,
    relation_atom,
    rename_state,
    rows_to_epistemic,
    unary_atom,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWN_OPERATORS = ("affirmative_gold", "negated_opposite", "contrastive_both")
ENTITY_ALIASES = (
    "noru", "vela", "tavi", "soren", "mira", "keto", "luma", "pavo",
    "runi", "dexa", "zori", "bela", "cavo", "fira", "ganu", "hena",
    "jora", "lito", "mavo", "pira", "ravo", "sela", "tora", "vani",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def opaque_id(kind: str, token: str, length: int = 16) -> str:
    return f"{kind}_{sha256_text(token)[:length]}"


def deterministic_shuffle(values: Iterable[Any], token: str) -> list[Any]:
    result = list(values)
    random.Random(int(sha256_text(token)[:16], 16)).shuffle(result)
    return result


def truth_label(allowed_values: Sequence[bool]) -> str:
    values = tuple(allowed_values)
    if values == (False,):
        return "false"
    if values == (True,):
        return "true"
    if values == (False, True):
        return "unknown"
    raise ValueError(f"Invalid V22 epistemic values: {values}")


def atom_with_mapping(atom: str, mapping: dict[str, str]) -> str:
    values = parse_atom(atom)
    if values[0] == "u":
        return unary_atom(values[1], mapping[values[2]])
    return relation_atom(values[1], mapping[values[2]], mapping[values[3]])


def world_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, bool]:
    return {row["atom"]: bool(row["value"]) for row in rows}


def update_query_answer(
    query: dict[str, Any], target_program: dict[str, Any], v22_config: dict[str, Any]
) -> None:
    entities = query["entities"]
    world = world_from_rows(query["reference_complete_world"])
    unknown = list(query.get("unknown_atoms", []))
    state = epistemic_from_world(v22_config, entities, world, unknown)
    answer = execute_partial(
        [target_program], v22_config, entities, state, query["action_binding"],
        v22_config["limits"]["maximumUnknownAtomsPerQuery"],
    )
    query["epistemic_state"] = epistemic_rows(state)
    query["reference_complete_world"] = [
        {"atom": atom, "value": value} for atom, value in sorted(world.items())
    ]
    query["canonical_state_binding_hash"] = canonical_state_hash(
        v22_config, entities, world, query["action_binding"]
    )
    query.update(answer)


def controlled_queries(
    episode: dict[str, Any], v22_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Replace V22's orientation/topology pairs with background-matched pairs."""

    queries = copy.deepcopy(episode["oracle_grounding"]["queries"])
    target_program = episode["target"]["program"]

    orientation = [row for row in queries if row["query_axis"] == "relation_orientation"]
    if len(orientation) != 2:
        raise ValueError("Every V22 episode must have exactly two orientation queries")
    base_world = world_from_rows(orientation[0]["reference_complete_world"])
    binding = orientation[0]["action_binding"]
    direct_atom = relation_atom("linked", binding["actor"], binding["target"])
    reverse_atom = relation_atom("linked", binding["target"], binding["actor"])
    support_hashes = {
        row["canonical_state_binding_hash"] for row in episode["oracle_grounding"]["support"]
    }
    background_atoms = sorted(set(base_world) - {direct_atom, reverse_atom})
    chosen_background = None
    for attempt in range(1 << len(background_atoms)):
        candidate = dict(base_world)
        for index, atom in enumerate(background_atoms):
            if (attempt >> index) & 1:
                candidate[atom] = not candidate[atom]
        hashes = set()
        for direct, reverse in ((True, False), (False, True)):
            world = {**candidate, direct_atom: direct, reverse_atom: reverse}
            hashes.add(canonical_state_hash(
                v22_config, orientation[0]["entities"], world, binding
            ))
        if not hashes & support_hashes:
            chosen_background = candidate
            break
    if chosen_background is None:
        raise RuntimeError("Unable to isolate orientation pair from support graphs")
    base_world = chosen_background
    for row, case, direct, reverse in zip(
        orientation,
        ("actor_to_target", "target_to_actor"),
        (True, False),
        (False, True),
        strict=True,
    ):
        world = dict(base_world)
        world[direct_atom] = direct
        world[reverse_atom] = reverse
        row["reference_complete_world"] = [
            {"atom": atom, "value": value} for atom, value in sorted(world.items())
        ]
        row["orientation_case"] = case
        row["counterfactual_group"] = f"{episode['id']}|relation_orientation"
        row["counterfactual_role"] = case
        update_query_answer(row, target_program, v22_config)

    topology = [row for row in queries if row["query_axis"] == "graph_topology"]
    if len(topology) != 2:
        raise ValueError("Every V22 episode must have exactly two topology queries")
    base_world = world_from_rows(topology[0]["reference_complete_world"])
    binding = topology[0]["action_binding"]
    middle = next(
        entity["id"] for entity in topology[0]["entities"]
        if entity["id"] not in set(binding.values())
    )
    linked_atoms = [
        atom for atom in base_world if parse_atom(atom)[:2] == ("r", "linked")
    ]
    cases = (
        ("directed_chain", {(binding["actor"], middle), (middle, binding["target"])}),
        ("common_parent_fork", {(middle, binding["actor"]), (middle, binding["target"])}),
    )
    for row, (case, edges) in zip(topology, cases, strict=True):
        world = dict(base_world)
        for atom in linked_atoms:
            parsed = parse_atom(atom)
            world[atom] = (parsed[2], parsed[3]) in edges
        row["reference_complete_world"] = [
            {"atom": atom, "value": value} for atom, value in sorted(world.items())
        ]
        row["topology_case"] = case
        row["counterfactual_group"] = f"{episode['id']}|graph_topology"
        row["counterfactual_role"] = case
        update_query_answer(row, target_program, v22_config)
    return queries


def positive_phrase(atom: str, orientation: str = "direct") -> str:
    values = parse_atom(atom)
    if values[0] == "u":
        return {
            "stable": f"{values[2]} is stable",
            "charged": f"{values[2]} is charged",
            "online": f"{values[2]} is online",
        }[values[1]]
    predicate, source, target = values[1], values[2], values[3]
    if predicate == "linked":
        return (
            f"{source} is linked to {target}" if orientation == "direct"
            else f"{target} receives a link from {source}"
        )
    if predicate == "feeds":
        return (
            f"{source} feeds {target}" if orientation == "direct"
            else f"{target} is fed by {source}"
        )
    raise ValueError(f"Unsupported predicate {predicate}")


def negative_phrase(atom: str, orientation: str = "direct") -> str:
    values = parse_atom(atom)
    if values[0] == "u":
        return {
            "stable": f"{values[2]} is unstable",
            "charged": f"{values[2]} is uncharged",
            "online": f"{values[2]} is offline",
        }[values[1]]
    predicate, source, target = values[1], values[2], values[3]
    if predicate == "linked":
        return (
            f"{source} is not linked to {target}" if orientation == "direct"
            else f"{target} does not receive a link from {source}"
        )
    if predicate == "feeds":
        return (
            f"{source} does not feed {target}" if orientation == "direct"
            else f"{target} is not fed by {source}"
        )
    raise ValueError(f"Unsupported predicate {predicate}")


def render_clause(
    atom: str, allowed_values: Sequence[bool], orientation: str,
    operator: str, surface_bank: str,
) -> str:
    label = truth_label(allowed_values)
    positive = positive_phrase(atom, orientation)
    negative = negative_phrase(atom, orientation)
    if label == "unknown":
        templates = {
            "fit_a": f"Current evidence leaves unresolved whether {positive}.",
            "fit_b": f"It has not been determined whether {positive}.",
            "eval_c": f"The record is inconclusive about whether {positive}.",
            "eval_d": f"Whether {positive} remains unknown.",
        }
        return templates[surface_bank]
    gold = positive if label == "true" else negative
    opposite = negative if label == "true" else positive
    if operator == "affirmative_gold":
        templates = {
            "fit_a": f"The inspection confirms that {gold}.",
            "fit_b": f"According to the current reading, {gold}.",
            "eval_c": f"The recorded state shows that {gold}.",
            "eval_d": f"Present status: {gold}.",
        }
    elif operator == "negated_opposite":
        templates = {
            "fit_a": f"It is not the case that {opposite}.",
            "fit_b": f"The claim that {opposite} is false.",
            "eval_c": f"One can rule out that {opposite}.",
            "eval_d": f"Contrary to the opposite description, it is false that {opposite}.",
        }
    elif operator == "contrastive_both":
        templates = {
            "fit_a": f"{gold.capitalize()}, rather than {opposite}.",
            "fit_b": f"The correct description is that {gold}; not that {opposite}.",
            "eval_c": f"{opposite.capitalize()} is wrong; instead, {gold}.",
            "eval_d": f"Between the two alternatives, {gold}, not {opposite}.",
        }
    else:
        raise ValueError(f"Unsupported known-value operator {operator}")
    return templates[surface_bank]


def semantic_signature(
    atom: str, allowed_values: Sequence[bool], orientation: str, operator: str,
) -> dict[str, Any]:
    values = parse_atom(atom)
    return {
        "predicate_kind": "unary" if values[0] == "u" else "relation",
        "predicate": values[1],
        "truth_status": truth_label(allowed_values),
        "semantic_operator": operator,
        "relation_orientation": orientation if values[0] == "r" else None,
    }


def entity_mapping(
    entities: Sequence[dict[str, str]], token: str,
) -> dict[str, str]:
    aliases = deterministic_shuffle(ENTITY_ALIASES, f"{token}|entity-aliases")
    mapping = {
        entity["id"]: aliases[index]
        for index, entity in enumerate(sorted(entities, key=lambda row: row["id"]))
    }
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("Opaque entity identifier collision")
    return mapping


def surface_bank_for(split: str, token: str, config: dict[str, Any]) -> str:
    banks = config["splits"][split]["surfaceBanks"]
    return banks[int(sha256_text(token)[:8], 16) % len(banks)]


def build_state_view(
    source_item: dict[str, Any], episode_id: str, split: str, role: str,
    config: dict[str, Any], v22_config: dict[str, Any], mapping_token: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_entities = source_item["entities"]
    source_world = world_from_rows(source_item["reference_complete_world"])
    mapping = entity_mapping(source_entities, mapping_token)
    entities, world, binding = rename_state(
        source_entities, source_world, source_item["action_binding"], mapping
    )
    unknown = [atom_with_mapping(atom, mapping) for atom in source_item.get("unknown_atoms", [])]
    state = epistemic_from_world(v22_config, entities, world, unknown)
    item_id = opaque_id("item", f"{config['seed']}|{source_item['id']}", 20)
    surface_rows = []
    for atom_ordinal, (atom, allowed) in enumerate(sorted(state.items())):
        parsed = parse_atom(atom)
        orientation = (
            ("direct", "inverse")[int(sha256_text(f"{item_id}|orientation|{atom}")[:8], 16) % 2]
            if parsed[0] == "r" else "direct"
        )
        label = truth_label(allowed)
        operator = (
            "explicit_unknown" if label == "unknown" else
            KNOWN_OPERATORS[
                int(sha256_text(f"{item_id}|operator|{atom_ordinal}|{atom}")[:8], 16)
                % len(KNOWN_OPERATORS)
            ]
        )
        bank = surface_bank_for(split, f"{item_id}|bank|{atom}", config)
        candidate_id = opaque_id("atom", f"{item_id}|candidate|{atom}")
        evidence_id = opaque_id("ev", f"{item_id}|evidence|{atom}")
        signature = semantic_signature(atom, allowed, orientation, operator)
        surface_rows.append({
            "atom": atom,
            "allowed_values": list(allowed),
            "truth_label": label,
            "candidate_id": candidate_id,
            "candidate_statement": positive_phrase(atom, "direct") + ".",
            "evidence_id": evidence_id,
            "evidence_text": render_clause(atom, allowed, orientation, operator, bank),
            "surface_bank": bank,
            "signature": signature,
        })

    candidates = deterministic_shuffle(surface_rows, f"{item_id}|candidate-order")
    evidence = deterministic_shuffle(surface_rows, f"{item_id}|evidence-order")
    candidate_index = {row["candidate_id"]: index for index, row in enumerate(candidates)}
    evidence_index = {row["evidence_id"]: index for index, row in enumerate(evidence)}
    public_entities = deterministic_shuffle(entities, f"{item_id}|entity-order")
    public = {
        "id": item_id,
        "entities": public_entities,
        "action": {"id": v22_config["action"]["id"], "binding": binding},
        "atom_candidates": [
            {"id": row["candidate_id"], "statement": row["candidate_statement"]}
            for row in candidates
        ],
        "evidence": [
            {"id": row["evidence_id"], "text": row["evidence_text"]}
            for row in evidence
        ],
        "observation": "\n".join(f"- {row['evidence_text']}" for row in evidence),
        "output_instruction": (
            "Align each evidence statement to one atom candidate and classify its status as "
            "true, false, or unknown."
        ),
    }
    groundings = [
        {
            "atom": row["atom"],
            "allowed_values": row["allowed_values"],
            "truth_label": row["truth_label"],
            "candidate_id": row["candidate_id"],
            "candidate_index": candidate_index[row["candidate_id"]],
            "evidence_id": row["evidence_id"],
            "evidence_index": evidence_index[row["evidence_id"]],
            "candidate_statement": row["candidate_statement"],
            "evidence_text": row["evidence_text"],
            "surface_bank": row["surface_bank"],
            **row["signature"],
        }
        for row in sorted(surface_rows, key=lambda value: value["atom"])
    ]
    oracle = {
        "id": item_id,
        "source_item_id": source_item["id"],
        "entities": sorted(entities, key=lambda value: value["id"]),
        "action_binding": binding,
        "epistemic_state": epistemic_rows(state),
        "reference_complete_world": [
            {"atom": atom, "value": value} for atom, value in sorted(world.items())
        ],
        "canonical_state_binding_hash": canonical_state_hash(
            v22_config, entities, world, binding
        ),
        "atom_groundings": groundings,
        "entity_mapping": mapping,
        "query_axis": source_item.get("query_axis"),
        "possible_transition_codes": source_item.get("possible_transition_codes"),
        "identifiable": source_item.get("identifiable"),
        "unknown_effect": source_item.get("unknown_effect"),
        "counterfactual_group": source_item.get("counterfactual_group"),
        "counterfactual_role": source_item.get("counterfactual_role"),
        "metamorphic_group": source_item.get("metamorphic_group"),
        "metamorphic_role": source_item.get("metamorphic_role"),
        "entity_count": len(entities),
    }
    scene = {
        "id": item_id,
        "schema_version": "22r2",
        "split": split,
        "episode_id": episode_id,
        "role": role,
        "agent_input": public,
        "target": {"atom_groundings": groundings},
        "oracle_metadata": {
            "source_item_id": source_item["id"],
            "query_axis": source_item.get("query_axis"),
            "counterfactual_group": source_item.get("counterfactual_group"),
            "counterfactual_role": source_item.get("counterfactual_role"),
            "metamorphic_group": source_item.get("metamorphic_group"),
            "metamorphic_role": source_item.get("metamorphic_role"),
            "entity_count": len(entities),
        },
    }
    return public, oracle, scene


def split_assignments(
    source_records: Sequence[dict[str, Any]], config: dict[str, Any]
) -> list[tuple[dict[str, Any], int, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        grouped[(record["construction_family"], record["split"])].append(record)
    result = []
    for family in sorted({record["construction_family"] for record in source_records}):
        fit = grouped[(family, "development_fit")]
        evaluation = grouped[(family, "development_evaluation")]
        if len(fit) != 3 or len(evaluation) != 3:
            raise ValueError(f"Unexpected V22 population for {family}")
        for ordinal, record in enumerate(fit):
            split = "grounding_fit" if ordinal < 2 else "grounding_calibration"
            result.append((record, ordinal, split))
        for local, record in enumerate(evaluation):
            result.append((record, local + 3, "grounding_evaluation"))
    return result


def build_corpus(
    source_records: Sequence[dict[str, Any]], config: dict[str, Any],
    v22_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    scenes = []
    for source, ordinal, split in split_assignments(source_records, config):
        episode_id = opaque_id("episode", f"{config['seed']}|{source['id']}", 20)
        query_rows = controlled_queries(source, v22_config)
        public_support = []
        oracle_support = []
        for row in source["oracle_grounding"]["support"]:
            public, oracle, scene = build_state_view(
                row, episode_id, split, "support", config, v22_config,
                f"{source['id']}|{row.get('metamorphic_group', row['id'])}",
            )
            public["observed_transition_code"] = row["transition_code"]
            oracle["transition_code"] = row["transition_code"]
            public_support.append(public)
            oracle_support.append(oracle)
            scenes.append(scene)
        public_queries = []
        oracle_queries = []
        for row in query_rows:
            pair_token = (
                row.get("counterfactual_group") or row.get("metamorphic_group") or row["id"]
            )
            public, oracle, scene = build_state_view(
                row, episode_id, split, "query", config, v22_config,
                f"{source['id']}|{pair_token}",
            )
            public_queries.append(public)
            oracle_queries.append(oracle)
            scenes.append(scene)
        public_support = deterministic_shuffle(public_support, f"{episode_id}|support-order")
        public_queries = deterministic_shuffle(public_queries, f"{episode_id}|query-order")
        oracle_support_lookup = {row["id"]: row for row in oracle_support}
        oracle_query_lookup = {row["id"]: row for row in oracle_queries}
        oracle_support = [oracle_support_lookup[row["id"]] for row in public_support]
        oracle_queries = [oracle_query_lookup[row["id"]] for row in public_queries]
        records.append({
            "id": episode_id,
            "schema_version": "22r2",
            "split": split,
            "agent_input": {
                "task": "ground_relational_state_then_induce_and_execute_schema",
                "entity_types": v22_config["entityTypes"],
                "action_schema": v22_config["action"],
                "dsl_contract": source["agent_input"]["dsl_contract"],
                "support_traces": public_support,
                "queries": public_queries,
                "output_instruction": source["agent_input"]["output_instruction"],
            },
            "oracle_grounding": {"support": oracle_support, "queries": oracle_queries},
            "target": copy.deepcopy(source["target"]),
            "oracle_metadata": {
                "source_episode_id": source["id"],
                "construction_family": source["construction_family"],
                "source_ordinal": ordinal,
            },
            "source": {
                "kind": "v22r2_derived_from_immutable_v22",
                "v22_schema_version": source["schema_version"],
                "v21_final_records_read": 0,
                "v21_final_model_results_read": 0,
            },
        })
    return records, scenes


def public_prompt_texts(scenes: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for scene in scenes:
        split = scene["split"]
        for row in scene["agent_input"]["atom_candidates"]:
            by_split[split].add(row["statement"])
        for row in scene["agent_input"]["evidence"]:
            by_split[split].add(row["text"])
    return dict(by_split)


def scene_prompt_layout(scene: dict[str, Any]) -> tuple[str, dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Return exact label-free content and character spans for candidates/evidence."""

    public = scene["agent_input"]
    parts = [
        "Typed entities: ",
        ", ".join(f"{row['id']} ({row['entity_type']})" for row in public["entities"]),
        ".\nAction binding: actor=", public["action"]["binding"]["actor"],
        ", target=", public["action"]["binding"]["target"],
        ".\nAtom candidates:\n",
    ]
    length = sum(len(value) for value in parts)
    candidate_spans: dict[str, tuple[int, int]] = {}
    for index, row in enumerate(public["atom_candidates"], start=1):
        prefix = f"Candidate {index}: "
        parts.append(prefix); length += len(prefix)
        start = length
        parts.append(row["statement"]); length += len(row["statement"])
        candidate_spans[row["id"]] = (start, length)
        parts.append("\n"); length += 1
    parts.append("Evidence statements:\n"); length += len("Evidence statements:\n")
    evidence_spans: dict[str, tuple[int, int]] = {}
    for index, row in enumerate(public["evidence"], start=1):
        prefix = f"Evidence {index}: "
        parts.append(prefix); length += len(prefix)
        start = length
        parts.append(row["text"]); length += len(row["text"])
        evidence_spans[row["id"]] = (start, length)
        if index != len(public["evidence"]):
            parts.append("\n"); length += 1
    text = "".join(parts)
    if len(text) != length:
        raise AssertionError("Scene prompt layout length mismatch")
    return text, candidate_spans, evidence_spans


def scene_prompt_text(scene: dict[str, Any]) -> str:
    """The exact label-free user content consumed by the frozen representation."""

    return scene_prompt_layout(scene)[0]


def grounding_lookup(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["evidence_id"]: row for row in scene["target"]["atom_groundings"]
    }


def predicted_epistemic_rows(
    scene: dict[str, Any], predictions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidate_atoms = {
        row["candidate_id"]: row["atom"] for row in scene["target"]["atom_groundings"]
    }
    values = {"false": [False], "true": [True], "unknown": [False, True]}
    result = []
    for prediction in predictions:
        result.append({
            "atom": candidate_atoms[prediction["candidate_id"]],
            "allowed_values": values[prediction["truth_label"]],
        })
    return sorted(result, key=lambda row: row["atom"])


def validate_scene_prediction(scene: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    expected = len(scene["target"]["atom_groundings"])
    if len(rows) != expected:
        raise ValueError("Prediction does not cover every evidence unit")
    evidence = [row["evidence_id"] for row in rows]
    candidates = [row["candidate_id"] for row in rows]
    if len(set(evidence)) != expected or len(set(candidates)) != expected:
        raise ValueError("Prediction must be a one-to-one evidence/candidate assignment")


def canonical_scene_graph(rows: Sequence[dict[str, Any]]) -> str:
    return canonical_json(sorted(rows, key=lambda row: row["atom"]))


def rows_to_state(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[bool, ...]]:
    return rows_to_epistemic(rows)
