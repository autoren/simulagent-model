"""Audit V22 typed-relational development structure, semantics, and metamorphic controls."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from v10_protocol import file_sha256
from v22_relational import (
    atom_universe,
    canonical_json,
    canonical_state_hash,
    epistemic_from_world,
    enumerate_program_hypotheses,
    execute_partial,
    expression_catalog,
    find_expression_counterexample,
    parse_atom,
    program_key,
    rows_to_epistemic,
    sha256_text,
    trace_consistent_hypotheses,
    validate_binding,
    validate_complete_world,
    validate_epistemic_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PUBLIC_KEYS = {
    "behavioral_signature", "epistemic_state", "oracle_grounding", "possible_transition_codes",
    "program", "program_key", "reference_complete_world", "semantic_signatures",
}


def read_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for split in ("development_fit", "development_evaluation"):
        path = root / "records" / f"{split}.jsonl"
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return records


def recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(recursive_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(recursive_keys(item) for item in value), set())
    return set()


def reference_world(rows: Sequence[dict[str, Any]]) -> dict[str, bool]:
    result = {}
    for row in rows:
        if row["atom"] in result:
            raise ValueError(f"Repeated reference atom {row['atom']}")
        result[row["atom"]] = row["value"]
    return result


def trace_for_induction(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "entities": value["entities"],
        "binding": value["action_binding"],
        "world": reference_world(value["reference_complete_world"]),
        "transition_code": value["transition_code"],
    }


def audit(
    records: Sequence[dict[str, Any]], config: dict[str, Any],
    manifest: dict[str, Any] | None = None, root: Path | None = None,
) -> dict[str, Any]:
    errors = []
    family_counts = Counter(value["construction_family"] for value in records)
    split_counts = Counter(value["split"] for value in records)
    bit_counts = Counter(value["agent_input"]["dsl_contract"]["outcome_bits"] for value in records)
    program_keys = [value["target"]["program_key"] for value in records]
    public_leaks = {
        value["id"]: sorted(recursive_keys(value["agent_input"]) & FORBIDDEN_PUBLIC_KEYS)
        for value in records
        if recursive_keys(value["agent_input"]) & FORBIDDEN_PUBLIC_KEYS
    }
    if len(records) != len(config["constructionFamilies"]) * config["episodesPerFamily"]:
        errors.append("Episode count differs from the development protocol")
    expected_family = {value: config["episodesPerFamily"] for value in config["constructionFamilies"]}
    if dict(family_counts) != expected_family:
        errors.append("Construction-family quotas differ from the protocol")
    expected_split = len(config["constructionFamilies"]) * config["fitEpisodesPerFamily"]
    if split_counts != {"development_fit": expected_split, "development_evaluation": expected_split}:
        errors.append("Fit/evaluation split is not balanced")
    if bit_counts != {1: len(records) // 2, 2: len(records) // 2}:
        errors.append("One-/two-bit mechanics are not balanced")
    if len(program_keys) != len(set(program_keys)):
        errors.append("Target programs are repeated")
    fit_programs = {
        value["target"]["program_key"] for value in records if value["split"] == "development_fit"
    }
    evaluation_programs = {
        value["target"]["program_key"]
        for value in records if value["split"] == "development_evaluation"
    }
    if fit_programs & evaluation_programs:
        errors.append("Fit and evaluation program structures overlap")
    if public_leaks:
        errors.append("Agent inputs contain target or oracle fields")

    catalog = expression_catalog()
    equivalent_pairs = []
    for index, left in enumerate(catalog):
        for right in catalog[index + 1:]:
            if find_expression_counterexample(
                left.expression, right.expression, config, config["entityLayouts"],
                config["limits"]["maximumTruthTableAtomsPerEquivalenceCheck"],
            ) is None:
                equivalent_pairs.append((left.name, right.name))
    if equivalent_pairs:
        errors.append("Expression catalog contains bounded-behavior duplicates")

    query_axis_counts = Counter()
    entity_count_counts = Counter()
    unknown_effect_counts = Counter()
    semantic_operator_counts = Counter()
    orientation_counts = Counter()
    topology_counts = Counter()
    support_lengths = []
    remaining_sizes = []
    oracle_answer_mismatches = 0
    state_validation_errors = []
    semantic_signature_errors = 0
    support_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    query_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    structural_axis_hashes: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    all_support_hashes = set()
    all_query_hashes = set()
    metamorphic: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    orientation_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relational_count_pairs = 0

    for record in records:
        target = record["target"]["program"]
        if program_key(target) != record["target"]["program_key"]:
            errors.append(f"Program key mismatch in {record['id']}")
        bits = record["agent_input"]["dsl_contract"]["outcome_bits"]
        hypotheses = enumerate_program_hypotheses(bits)
        support = record["oracle_grounding"]["support"]
        episode_support_hashes = set()
        episode_query_hashes = set()
        public_support = {value["id"]: value for value in record["agent_input"]["support_traces"]}
        support_lengths.append(len(support))
        traces = []
        for value in support:
            try:
                entities = value["entities"]
                world = reference_world(value["reference_complete_world"])
                state = rows_to_epistemic(value["epistemic_state"])
                validate_complete_world(config, entities, world)
                validate_epistemic_state(config, entities, state)
                validate_binding(config, entities, value["action_binding"])
                if state != epistemic_from_world(config, entities, world):
                    state_validation_errors.append(f"support_state:{value['id']}")
                if len(entities) not in config["supportEntityCounts"]:
                    state_validation_errors.append(f"support_entity_count:{value['id']}")
                expected_hash = canonical_state_hash(config, entities, world, value["action_binding"])
                if expected_hash != value["canonical_state_binding_hash"]:
                    state_validation_errors.append(f"support_hash:{value['id']}")
                if public_support[value["id"]]["observed_transition_code"] != value["transition_code"]:
                    state_validation_errors.append(f"support_outcome:{value['id']}")
                traces.append(trace_for_induction(value))
                all_support_hashes.add(expected_hash)
                episode_support_hashes.add(expected_hash)
                support_hashes_by_split[record["split"]].add(expected_hash)
            except (KeyError, ValueError) as error:
                state_validation_errors.append(f"{value.get('id')}:{error}")
            for signature in value["semantic_signatures"]:
                semantic_operator_counts[signature["semantic_operator"]] += 1
                if signature["relation_orientation"] is not None:
                    orientation_counts[signature["relation_orientation"]] += 1
                atom_values = parse_atom(signature["atom"])
                if signature["arguments"] != list(atom_values[2:]):
                    semantic_signature_errors += 1
        remaining = trace_consistent_hypotheses(hypotheses, traces, config)
        remaining_sizes.append(len(remaining))
        if len(remaining) != 1 or remaining[0].key != record["target"]["program_key"]:
            errors.append(f"Support does not uniquely identify target in {record['id']}")

        public_queries = {value["id"]: value for value in record["agent_input"]["queries"]}
        for query in record["oracle_grounding"]["queries"]:
            query_axis_counts[query["query_axis"]] += 1
            entity_count_counts[query["entity_count"]] += 1
            unknown_effect_counts[query["unknown_effect"]] += 1
            topology_counts[query["graph_topology"]] += 1
            try:
                entities = query["entities"]
                world = reference_world(query["reference_complete_world"])
                state = rows_to_epistemic(query["epistemic_state"])
                validate_complete_world(config, entities, world)
                validate_epistemic_state(config, entities, state)
                validate_binding(config, entities, query["action_binding"])
                if set(state) != set(atom_universe(config, entities)):
                    state_validation_errors.append(f"query_atoms:{query['id']}")
                if len(query["unknown_atoms"]) > config["limits"]["maximumUnknownAtomsPerQuery"]:
                    state_validation_errors.append(f"query_unknown_limit:{query['id']}")
                expected = execute_partial(
                    [target], config, entities, state, query["action_binding"],
                    config["limits"]["maximumUnknownAtomsPerQuery"],
                )
                if expected["possible_transition_codes"] != query["possible_transition_codes"] or expected["identifiable"] != query["identifiable"]:
                    oracle_answer_mismatches += 1
                expected_hash = canonical_state_hash(config, entities, world, query["action_binding"])
                if expected_hash != query["canonical_state_binding_hash"]:
                    state_validation_errors.append(f"query_hash:{query['id']}")
                all_query_hashes.add(expected_hash)
                episode_query_hashes.add(expected_hash)
                query_hashes_by_split[record["split"]].add(expected_hash)
                structural_axis_hashes[query["query_axis"]][record["split"]].add(expected_hash)
                if public_queries[query["id"]]["action"]["binding"] != query["action_binding"]:
                    state_validation_errors.append(f"public_binding:{query['id']}")
            except (KeyError, ValueError) as error:
                state_validation_errors.append(f"{query.get('id')}:{error}")
            for signature in query["semantic_signatures"]:
                semantic_operator_counts[signature["semantic_operator"]] += 1
                if signature["relation_orientation"] is not None:
                    orientation_counts[signature["relation_orientation"]] += 1
                atom_values = parse_atom(signature["atom"])
                if signature["arguments"] != list(atom_values[2:]):
                    semantic_signature_errors += 1
            group = query.get("metamorphic_group")
            if group:
                metamorphic[group].append((record, query))
            if query["query_axis"] == "relation_orientation":
                orientation_cases[record["id"]].append(query)
        if episode_support_hashes & episode_query_hashes:
            errors.append(f"Support/query graph overlap within {record['id']}")

    if state_validation_errors:
        errors.append("Relational world or epistemic-state validation failed")
    if oracle_answer_mismatches:
        errors.append("Stored oracle query answers differ from execution")
    if semantic_signature_errors:
        errors.append("Language semantic signatures do not preserve ordered atom arguments")
    structural_split_overlap = {
        axis: len(
            structural_axis_hashes[axis]["development_fit"]
            & structural_axis_hashes[axis]["development_evaluation"]
        )
        for axis in ("graph_topology", "entity_count_extrapolation")
    }
    if any(structural_split_overlap.values()):
        errors.append("Fit/evaluation topology or entity-count graphs overlap")
    if set(query_axis_counts) != set(config["queryAxes"]):
        errors.append("A registered query axis is absent")
    if set(entity_count_counts) != set(config["queryEntityCounts"]):
        errors.append("Queries do not cover all registered entity counts")
    if not {"fully_observed", "outcome_sensitive", "outcome_invariant"} <= set(unknown_effect_counts):
        errors.append("Queries do not cover required epistemic effects")
    if set(semantic_operator_counts) != set(config["language"]["supportedOperators"]):
        errors.append("Registered relational language operators are missing")
    if set(orientation_counts) != set(config["language"]["relationOrientations"]):
        errors.append("Direct or inverse relation language is missing")

    permutation_checks = []
    distractor_checks = []
    entity_count_checks = []
    for group, values in metamorphic.items():
        if len(values) != 2:
            errors.append(f"Metamorphic group {group} does not have exactly two members")
            continue
        queries = [value[1] for value in values]
        axis = queries[0]["query_axis"]
        same_answer = all(
            value["possible_transition_codes"] == queries[0]["possible_transition_codes"]
            for value in queries
        )
        if axis == "permutation_equivariance":
            same_hash = queries[0]["canonical_state_binding_hash"] == queries[1]["canonical_state_binding_hash"]
            permutation_checks.append(same_answer and same_hash)
        elif axis == "distractor_invariance":
            distractor_checks.append(same_answer and queries[0]["entity_count"] != queries[1]["entity_count"])
        elif axis == "entity_count_extrapolation":
            expected_sensitive = queries[0]["entity_count_semantics"] == "new_witness_sensitive"
            entity_count_checks.append((not same_answer) if expected_sensitive else same_answer)
            relational_count_pairs += int(expected_sensitive)
    if not permutation_checks or not all(permutation_checks):
        errors.append("Permutation equivariance metamorphic control failed")
    if not distractor_checks or not all(distractor_checks):
        errors.append("Distractor invariance metamorphic control failed")
    if not entity_count_checks or not all(entity_count_checks):
        errors.append("Entity-count semantic pairs failed")

    orientation_checks = []
    for episode_id, queries in orientation_cases.items():
        if len(queries) != 2:
            orientation_checks.append(False)
            continue
        valid = []
        for query in queries:
            binding = query["action_binding"]
            state = rows_to_epistemic(query["epistemic_state"])
            direct = state[f"r:linked:{binding['actor']}:{binding['target']}"]
            reverse = state[f"r:linked:{binding['target']}:{binding['actor']}"]
            valid.append(direct != reverse)
        orientation_checks.append(all(valid))
    if not orientation_checks or not all(orientation_checks):
        errors.append("Relation-orientation control failed")

    if manifest is not None and root is not None:
        for path, expected in manifest["implementation_sha256"].items():
            if file_sha256(PROJECT_ROOT / path) != expected:
                errors.append(f"Implementation hash mismatch: {path}")
        dataset_parts_by_name = {}
        for relative, expected in manifest["artifact_sha256"].items():
            content = (root / relative).read_text()
            if sha256_text(content) != expected:
                errors.append(f"Artifact hash mismatch: {relative}")
            dataset_parts_by_name[relative] = f"{relative}\n{content}"
        dataset_order = (
            "records/development_fit.jsonl",
            "records/development_evaluation.jsonl",
        )
        if sha256_text("".join(dataset_parts_by_name[value] for value in dataset_order)) != manifest["dataset_sha256"]:
            errors.append("Dataset hash mismatch")
        expected_access = {
            "v21_final_records_read": 0,
            "v21_final_model_results_read": 0,
            "new_model_forward_passes": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "final_v22_records_created": 0,
        }
        if manifest["data_access"] != expected_access:
            errors.append("Manifest violates the V22 development firewall")

    checks = {
        "episode_and_stratum_counts": len(records) == 24 and dict(family_counts) == expected_family,
        "program_structures_unique_and_split_disjoint": len(program_keys) == len(set(program_keys)) and not (fit_programs & evaluation_programs),
        "expression_catalog_bounded_unique": not equivalent_pairs,
        "support_identifies_target": remaining_sizes and max(remaining_sizes) == 1,
        "oracle_query_execution_exact": oracle_answer_mismatches == 0,
        "explicit_epistemic_semantics_valid": not state_validation_errors,
        "support_query_graphs_disjoint_within_episode": not any(
            "Support/query graph overlap within" in value for value in errors
        ),
        "registered_structural_axes_split_disjoint": not any(structural_split_overlap.values()),
        "permutation_equivariance": bool(permutation_checks) and all(permutation_checks),
        "distractor_invariance": bool(distractor_checks) and all(distractor_checks),
        "entity_count_semantics": bool(entity_count_checks) and all(entity_count_checks),
        "relation_orientation": bool(orientation_checks) and all(orientation_checks),
        "language_signature_coverage": set(semantic_operator_counts) == set(config["language"]["supportedOperators"]) and set(orientation_counts) == set(config["language"]["relationOrientations"]),
        "zero_public_target_leaks": not public_leaks,
        "development_firewall": (
            config["firewall"]["developmentOnly"]
            and config["limits"]["newModelForwardPassesPermitted"] == 0
            and config["limits"]["adapterTrainingRunsPermitted"] == 0
            and not (PROJECT_ROOT / "data/v22-final").exists()
            and not (PROJECT_ROOT / "outputs/v22-final").exists()
        ),
    }
    return {
        "schema_version": 22,
        "experiment": "v22_typed_relational_development_audit",
        "passed": not errors and all(checks.values()),
        "errors": errors,
        "checks": checks,
        "episodes": len(records),
        "family_counts": dict(family_counts),
        "split_counts": dict(split_counts),
        "outcome_bit_counts": dict(bit_counts),
        "support_traces": {
            "minimum": min(support_lengths), "maximum": max(support_lengths),
            "mean": sum(support_lengths) / len(support_lengths),
        },
        "maximum_remaining_hypotheses": max(remaining_sizes),
        "query_axis_counts": dict(query_axis_counts),
        "entity_count_counts": dict(entity_count_counts),
        "unknown_effect_counts": dict(unknown_effect_counts),
        "semantic_operator_counts": dict(semantic_operator_counts),
        "relation_orientation_counts": dict(orientation_counts),
        "graph_topology_counts": dict(topology_counts),
        "structural_axis_split_overlap": structural_split_overlap,
        "relational_new_witness_pairs": relational_count_pairs,
        "expression_equivalent_pairs": equivalent_pairs,
        "public_leaks": public_leaks,
        "state_validation_errors": state_validation_errors,
        "data_access": {
            "v21_final_records_read": 0,
            "v21_final_model_results_read": 0,
            "new_model_forward_passes": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "final_v22_records_created": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v22.json")
    parser.add_argument("--dataset", default="data/v22-relational-development")
    parser.add_argument("--output", default="outputs/v22-relational-development/audit.json")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    root = PROJECT_ROOT / args.dataset
    manifest = json.loads((root / "manifest.json").read_text())
    result = audit(read_records(root), config, manifest, root)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
