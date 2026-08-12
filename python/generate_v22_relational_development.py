"""Generate the open, oracle-first V22 typed-relational development benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

from v10_protocol import file_sha256
from v22_relational import (
    ProgramHypothesis,
    action_bindings,
    atom_universe,
    canonical_json,
    canonical_state_hash,
    entities_for_layout,
    epistemic_from_world,
    epistemic_rows,
    enumerate_program_hypotheses,
    evaluate_program,
    execute_partial,
    extend_with_inert_entity,
    greedy_identifying_support,
    hashed_world,
    layout_key,
    parse_atom,
    program_key,
    rename_state,
    render_observation,
    sha256_text,
    target_hypotheses,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def support_layouts(config: dict[str, Any]) -> list[dict[str, int]]:
    allowed = set(config["supportEntityCounts"])
    return [
        value for value in config["entityLayouts"]
        if value["units"] + value["hubs"] in allowed
    ]


def selected_targets(config: dict[str, Any]) -> list[tuple[str, int, ProgramHypothesis]]:
    selected = []
    used = set()
    for family in config["constructionFamilies"]:
        bit_ordinals = Counter()
        for ordinal in range(config["episodesPerFamily"]):
            bits = config["outcomeBits"][ordinal % len(config["outcomeBits"])]
            candidates = sorted(
                target_hypotheses(family, bits),
                key=lambda value: sha256_text(
                    f"{config['seed']}|{family}|{bits}|{value.key}"
                ),
            )
            available = [value for value in candidates if value.key not in used]
            index = bit_ordinals[bits]
            if index >= len(available):
                raise RuntimeError(f"Insufficient unused V22 targets for {family}/{bits}")
            target = available[index]
            bit_ordinals[bits] += 1
            used.add(target.key)
            selected.append((family, ordinal, target))
    return selected


def serialize_state_item(
    config: dict[str, Any], identifier: str, entities: Sequence[dict[str, str]],
    world: dict[str, bool], binding: dict[str, str], unknown_atoms: Sequence[str], token: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = epistemic_from_world(config, entities, world, unknown_atoms)
    observation, signatures = render_observation(config, entities, state, sha256_text(token))
    public = {
        "id": identifier,
        "entities": list(entities),
        "action": {"id": config["action"]["id"], "binding": dict(binding)},
        "observation": observation,
    }
    oracle = {
        "id": identifier,
        "entities": list(entities),
        "action_binding": dict(binding),
        "epistemic_state": epistemic_rows(state),
        "reference_complete_world": [
            {"atom": atom, "value": value} for atom, value in sorted(world.items())
        ],
        "semantic_signatures": signatures,
        "canonical_state_binding_hash": canonical_state_hash(
            config, entities, world, binding
        ),
    }
    return public, oracle


def unused_world(
    config: dict[str, Any], layout: dict[str, int], token: str,
    forbidden_hashes: set[str], binding_index: int = 0,
) -> tuple[list[dict[str, str]], dict[str, bool], dict[str, str]]:
    entities = entities_for_layout(layout)
    bindings = action_bindings(config, entities)
    for attempt in range(1000):
        world = hashed_world(config, entities, f"{token}|{attempt}")
        binding = bindings[(binding_index + attempt) % len(bindings)]
        identity = canonical_state_hash(config, entities, world, binding)
        if identity not in forbidden_hashes:
            forbidden_hashes.add(identity)
            return entities, world, binding
    raise RuntimeError("Unable to generate a structurally fresh V22 world")


def set_linked_edges(
    config: dict[str, Any], entities: Sequence[dict[str, str]], world: dict[str, bool],
    edges: set[tuple[str, str]],
) -> dict[str, bool]:
    result = dict(world)
    for atom in atom_universe(config, entities):
        values = parse_atom(atom)
        if values[:2] == ("r", "linked"):
            result[atom] = (values[2], values[3]) in edges
    return result


def graph_topology(world: dict[str, bool]) -> str:
    edges = [
        (values[2], values[3])
        for atom, truth in world.items()
        if truth and (values := parse_atom(atom))[:2] == ("r", "linked")
    ]
    if not edges:
        return "empty"
    nodes = sorted(set(value for edge in edges for value in edge))
    indegree = Counter(target for _, target in edges)
    outdegree = Counter(source for source, _ in edges)
    if len(edges) == 1:
        return "single_edge"
    if len(edges) == len(nodes) and all(indegree[value] == outdegree[value] == 1 for value in nodes):
        return "cycle"
    if max(outdegree.values()) >= 2:
        return "out_fork"
    if max(indegree.values()) >= 2:
        return "in_fork"
    if len(edges) >= 2:
        return "chain_or_disjoint"
    return "other"


def extension_pair(
    config: dict[str, Any], target: ProgramHypothesis, token: str,
    require_change: bool,
) -> tuple[
    list[dict[str, str]], dict[str, bool], dict[str, str],
    list[dict[str, str]], dict[str, bool], dict[str, str], str,
]:
    base_layouts = [value for value in config["entityLayouts"] if value["units"] + value["hubs"] == 3]
    for base_layout in base_layouts:
        base_entities = entities_for_layout(base_layout)
        for entity_type in ("unit", "hub"):
            if len(base_entities) >= config["dsl"]["maximumEntityCount"]:
                continue
            for binding in action_bindings(config, base_entities):
                for attempt in range(128):
                    base_world = hashed_world(config, base_entities, f"{token}|base|{attempt}")
                    extended_entities, inert_world, added = extend_with_inert_entity(
                        config, base_entities, base_world, entity_type
                    )
                    new_atoms = sorted(set(inert_world) - set(base_world))
                    base_code = evaluate_program(
                        target.program, config, base_entities, base_world, binding
                    )
                    if not require_change:
                        if evaluate_program(
                            target.program, config, extended_entities, inert_world, binding
                        ) == base_code:
                            return (
                                base_entities, base_world, binding,
                                extended_entities, inert_world, binding, added,
                            )
                        continue
                    for values in product((False, True), repeat=len(new_atoms)):
                        extended_world = dict(inert_world)
                        extended_world.update(dict(zip(new_atoms, values, strict=True)))
                        if evaluate_program(
                            target.program, config, extended_entities, extended_world, binding
                        ) != base_code:
                            return (
                                base_entities, base_world, binding,
                                extended_entities, extended_world, binding, added,
                            )
    raise RuntimeError(
        f"Could not construct {'sensitive' if require_change else 'invariant'} entity extension"
    )


def partial_queries(
    config: dict[str, Any], target: ProgramHypothesis, token: str,
    forbidden_hashes: set[str],
) -> list[tuple[list[dict[str, str]], dict[str, bool], dict[str, str], list[str], str]]:
    entities = entities_for_layout({"units": 3, "hubs": 1})
    bindings = action_bindings(config, entities)
    for attempt in range(512):
        world = hashed_world(config, entities, f"{token}|partial|{attempt}")
        binding = bindings[attempt % len(bindings)]
        identity = canonical_state_hash(config, entities, world, binding)
        if identity in forbidden_hashes:
            continue
        sensitive = None
        invariant = None
        for atom in atom_universe(config, entities):
            state = epistemic_from_world(config, entities, world, [atom])
            answer = execute_partial([target.program], config, entities, state, binding, 1)
            if answer["identifiable"] and invariant is None:
                invariant = atom
            if not answer["identifiable"] and sensitive is None:
                sensitive = atom
        if sensitive is not None and invariant is not None:
            forbidden_hashes.add(identity)
            return [
                (entities, world, binding, [sensitive], "outcome_sensitive"),
                (entities, world, binding, [invariant], "outcome_invariant"),
            ]
    raise RuntimeError("V22 target lacks a constructed sensitive/invariant partial-state pair")


def query_payload(
    config: dict[str, Any], episode_id: str, index: int, axis: str,
    entities: Sequence[dict[str, str]], world: dict[str, bool], binding: dict[str, str],
    unknown_atoms: Sequence[str], target: ProgramHypothesis, metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_id = f"{episode_id}:query:{index:03d}"
    public, oracle = serialize_state_item(
        config, query_id, entities, world, binding, unknown_atoms,
        f"{episode_id}|{axis}|{index}",
    )
    state = epistemic_from_world(config, entities, world, unknown_atoms)
    answer = execute_partial(
        [target.program], config, entities, state, binding,
        config["limits"]["maximumUnknownAtomsPerQuery"],
    )
    public["query_axis"] = axis
    oracle.update({
        "query_axis": axis,
        "unknown_atoms": list(unknown_atoms),
        "unknown_effect": (
            "fully_observed" if not unknown_atoms else
            "outcome_invariant" if answer["identifiable"] else "outcome_sensitive"
        ),
        "entity_count": len(entities),
        "graph_topology": graph_topology(world),
        **answer,
        **metadata,
    })
    return public, oracle


def build_queries(
    config: dict[str, Any], episode_id: str, family: str, ordinal: int,
    target: ProgramHypothesis, support_hashes: set[str], split: str,
    structural_hashes: dict[str, dict[str, set[str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public_queries = []
    oracle_queries = []
    forbidden = set(support_hashes)

    def add(
        axis: str, entities: Sequence[dict[str, str]], world: dict[str, bool],
        binding: dict[str, str], unknown: Sequence[str] = (), **metadata: Any,
    ) -> None:
        public, oracle = query_payload(
            config, episode_id, len(public_queries), axis, entities, world, binding,
            unknown, target, metadata,
        )
        public_queries.append(public)
        oracle_queries.append(oracle)
        forbidden.add(oracle["canonical_state_binding_hash"])
        if axis in structural_hashes:
            structural_hashes[axis][split].add(oracle["canonical_state_binding_hash"])

    entities, world, binding = unused_world(
        config, {"units": 3, "hubs": 0}, f"{episode_id}|binding", forbidden,
        binding_index=ordinal,
    )
    add("binding_recombination", entities, world, binding, binding_role="new_action_binding")

    entities = entities_for_layout({"units": 2, "hubs": 0})
    binding = action_bindings(config, entities)[0]
    actor, target_id = binding["actor"], binding["target"]
    def fresh_orientation(edges: set[tuple[str, str]], label: str) -> dict[str, bool]:
        for attempt in range(128):
            candidate = set_linked_edges(
                config, entities,
                hashed_world(config, entities, f"{episode_id}|orientation|{label}|{attempt}"),
                edges,
            )
            if canonical_state_hash(config, entities, candidate, binding) not in forbidden:
                return candidate
        raise RuntimeError("Unable to construct support-disjoint relation orientation")

    direct = fresh_orientation({(actor, target_id)}, "direct")
    add("relation_orientation", entities, direct, binding, orientation_case="actor_to_target")
    reverse = fresh_orientation({(target_id, actor)}, "reverse")
    add("relation_orientation", entities, reverse, binding, orientation_case="target_to_actor")

    entities = entities_for_layout({"units": 3, "hubs": 0})
    binding = action_bindings(config, entities)[ordinal % len(action_bindings(config, entities))]
    actor, target_id = binding["actor"], binding["target"]
    middle = next(value["id"] for value in entities if value["id"] not in binding.values())
    def fresh_topology(edges: set[tuple[str, str]], label: str) -> dict[str, bool]:
        opposite = (
            "development_evaluation" if split == "development_fit" else "development_fit"
        )
        blocked = structural_hashes["graph_topology"][opposite]
        for attempt in range(512):
            candidate = set_linked_edges(
                config, entities,
                hashed_world(config, entities, f"{episode_id}|topology|{label}|{attempt}"),
                edges,
            )
            identity = canonical_state_hash(config, entities, candidate, binding)
            if identity not in blocked and identity not in forbidden:
                return candidate
        raise RuntimeError("Unable to construct a split-disjoint graph topology")

    chain = fresh_topology({(actor, middle), (middle, target_id)}, "chain")
    fork = fresh_topology({(middle, actor), (middle, target_id)}, "fork")
    add("graph_topology", entities, chain, binding, topology_case="directed_chain")
    add("graph_topology", entities, fork, binding, topology_case="common_parent_fork")

    relational = family in {"two_hop_composition", "existential_aggregation"}
    base_entities, base_world, base_binding, extended_entities, extended_world, extended_binding, added = extension_pair(
        config, target, f"{episode_id}|entity_count", require_change=relational,
    )
    group = f"{episode_id}:entity_extension"
    add(
        "entity_count_extrapolation", base_entities, base_world, base_binding,
        metamorphic_group=group, metamorphic_role="base",
        entity_count_semantics="new_witness_sensitive" if relational else "irrelevant_extension_invariant",
    )
    add(
        "entity_count_extrapolation", extended_entities, extended_world, extended_binding,
        metamorphic_group=group, metamorphic_role="extended", added_entity=added,
        entity_count_semantics="new_witness_sensitive" if relational else "irrelevant_extension_invariant",
    )

    for entities, world, binding, unknown, effect in partial_queries(
        config, target, episode_id, forbidden
    ):
        add("partial_observation", entities, world, binding, unknown, requested_effect=effect)

    base_entities, base_world, base_binding, extended_entities, extended_world, _, added = extension_pair(
        config, target, f"{episode_id}|distractor", require_change=False,
    )
    group = f"{episode_id}:distractor"
    add(
        "distractor_invariance", base_entities, base_world, base_binding,
        metamorphic_group=group, metamorphic_role="base",
    )
    add(
        "distractor_invariance", extended_entities, extended_world, base_binding,
        metamorphic_group=group, metamorphic_role="with_inert_entity", added_entity=added,
    )

    entities, world, binding = unused_world(
        config, {"units": 3, "hubs": 1}, f"{episode_id}|permutation", forbidden
    )
    mapping = {}
    for entity_type in ("unit", "hub"):
        identifiers = sorted(value["id"] for value in entities if value["entity_type"] == entity_type)
        mapping.update(dict(zip(identifiers, reversed(identifiers), strict=True)))
    renamed_entities, renamed_world, renamed_binding = rename_state(entities, world, binding, mapping)
    renamed_entities = list(reversed(renamed_entities))
    group = f"{episode_id}:permutation"
    add(
        "permutation_equivariance", entities, world, binding,
        metamorphic_group=group, metamorphic_role="original", entity_mapping=mapping,
    )
    add(
        "permutation_equivariance", renamed_entities, renamed_world, renamed_binding,
        metamorphic_group=group, metamorphic_role="renamed_and_reordered",
    )
    return public_queries, oracle_queries


def build_episode(
    config: dict[str, Any], family: str, ordinal: int, target: ProgramHypothesis,
    structural_hashes: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    output_bits = len(target.program["output_bits"])
    hypotheses = enumerate_program_hypotheses(output_bits)
    support, search = greedy_identifying_support(
        target, hypotheses, config, support_layouts(config),
        config["limits"]["maximumSupportTraces"],
    )
    key = sha256_text(f"{config['seed']}|{family}|{ordinal}|{target.key}")[:20]
    episode_id = f"v22:{key}"
    public_support = []
    oracle_support = []
    support_hashes = set()
    for index, trace in enumerate(support):
        trace_id = f"{episode_id}:support:{index:02d}"
        public, oracle = serialize_state_item(
            config, trace_id, trace["entities"], trace["world"], trace["binding"], (),
            f"{episode_id}|support|{index}",
        )
        public["observed_transition_code"] = trace["transition_code"]
        oracle["transition_code"] = trace["transition_code"]
        oracle["entity_count"] = len(trace["entities"])
        public_support.append(public)
        oracle_support.append(oracle)
        support_hashes.add(oracle["canonical_state_binding_hash"])
    split = (
        "development_fit" if ordinal < config["fitEpisodesPerFamily"]
        else "development_evaluation"
    )
    public_queries, oracle_queries = build_queries(
        config, episode_id, family, ordinal, target, support_hashes, split,
        structural_hashes,
    )
    return {
        "id": episode_id,
        "schema_version": 22,
        "split": split,
        "construction_family": family,
        "generalization_axis": family,
        "agent_input": {
            "task": "induce_typed_relational_action_schema_and_answer_queries",
            "entity_types": config["entityTypes"],
            "action_schema": config["action"],
            "dsl_contract": {
                "value_type": "boolean", "operators": config["dsl"]["operators"],
                "maximum_bound_variables": 1, "outcome_bits": output_bits,
            },
            "support_traces": public_support,
            "queries": public_queries,
            "output_instruction": (
                "Infer one lifted rule for the parameterized action and return every possible "
                "visible transition code for each epistemic query."
            ),
        },
        "oracle_grounding": {"support": oracle_support, "queries": oracle_queries},
        "target": {
            "program": target.program,
            "program_key": target.key,
            "program_key_sha256": sha256_text(target.key),
            "component_families": list(target.component_families),
            "component_names": list(target.component_names),
            "behavior_identifiable": True,
        },
        "search": search,
        "source": {
            "kind": "v22_open_relational_development_generator",
            "v21_final_records_read": 0,
            "v21_final_model_results_read": 0,
        },
    }


def generate(config: dict[str, Any]) -> list[dict[str, Any]]:
    structural_hashes = {
        axis: {"development_fit": set(), "development_evaluation": set()}
        for axis in ("graph_topology", "entity_count_extrapolation")
    }
    return [
        build_episode(config, family, ordinal, target, structural_hashes)
        for family, ordinal, target in selected_targets(config)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v22.json")
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text())
    records = generate(config)
    output_dir = PROJECT_ROOT / config["outputDir"]
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {}
    dataset_parts = []
    for split in ("development_fit", "development_evaluation"):
        values = [value for value in records if value["split"] == split]
        content = "".join(canonical_json(value) + "\n" for value in values)
        relative = f"records/{split}.jsonl"
        (output_dir / relative).write_text(content)
        artifact_hashes[relative] = sha256_text(content)
        dataset_parts.append(f"{relative}\n{content}")
    implementation = (
        "python/v22_relational.py",
        "python/generate_v22_relational_development.py",
    )
    manifest = {
        "schema_version": 22,
        "experiment": config["experiment"],
        "config": args.config,
        "config_sha256": file_sha256(config_path),
        "records": len(records),
        "split_counts": dict(Counter(value["split"] for value in records)),
        "family_counts": dict(Counter(value["construction_family"] for value in records)),
        "query_axis_counts": dict(Counter(
            query["query_axis"]
            for value in records for query in value["oracle_grounding"]["queries"]
        )),
        "implementation_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in implementation
        },
        "artifact_sha256": artifact_hashes,
        "dataset_sha256": sha256_text("".join(dataset_parts)),
        "data_access": {
            "v21_final_records_read": 0,
            "v21_final_model_results_read": 0,
            "new_model_forward_passes": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "final_v22_records_created": 0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output_dir": config["outputDir"], **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
