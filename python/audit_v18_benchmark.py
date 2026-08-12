"""Audit the V18 development benchmark without reading any protected evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from v18_schema import (
    enumerate_program_hypotheses,
    evaluate_program,
    execute_query,
    program_signature,
    trace_consistent_hypotheses,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_AGENT_KEYS = frozenset({
    "action_dependency_schema",
    "allowed_values",
    "assignment",
    "behavioral_signature",
    "executable_schema",
    "output_bits",
    "relevant_determinants",
    "transition_cases",
})


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def recursively_present_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(recursively_present_keys(entry) for entry in value.values()))
    if isinstance(value, list):
        return set().union(*(recursively_present_keys(entry) for entry in value)) if value else set()
    return set()


def read_records(dataset_dir: Path) -> list[dict[str, Any]]:
    records = []
    for split in ("train", "calibration", "development"):
        path = dataset_dir / "records" / f"{split}.jsonl"
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return records


def audit(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_splits = {
        "train": config["trainingEpisodes"],
        "calibration": config["calibrationEpisodes"],
        "development": len(config["developmentAxes"]) * config["episodesPerDevelopmentAxis"],
    }
    split_counts = Counter(value["split"] for value in records)
    if dict(split_counts) != expected_splits:
        errors.append(f"Split counts differ: {dict(split_counts)} != {expected_splits}")
    identifiers = [value["id"] for value in records]
    if len(identifiers) != len(set(identifiers)):
        errors.append("Episode ids are not unique")
    actions = [value["agent_input"]["candidate_action"] for value in records]
    if len(actions) != len(set(actions)):
        errors.append("Candidate actions cross episode boundaries")

    leaked_keys: Counter[str] = Counter()
    symbolic_mismatches = 0
    underidentified_schemas = 0
    query_ids: list[str] = []
    query_effects: dict[str, Counter[str]] = defaultdict(Counter)
    query_labels: dict[str, Counter[bool]] = defaultdict(Counter)
    support_lengths: Counter[int] = Counter()
    program_signatures: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    family_pairs: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    maximum_depth: dict[str, set[int]] = defaultdict(set)
    relevant_counts: dict[str, Counter[int]] = defaultdict(Counter)

    hypothesis_cache: dict[tuple[tuple[str, ...], int], Any] = {}
    for record in records:
        axis = record["generalization_axis"]
        leaked_keys.update(recursively_present_keys(record["agent_input"]) & FORBIDDEN_AGENT_KEYS)
        if record["schema_version"] != 18:
            errors.append(f"{record['id']} has the wrong schema version")
        if record["source"] != {
            "kind": "procedural_v18_development_simulator",
            "v17_records_read": 0,
            "v17_model_results_read": 0,
        }:
            errors.append(f"{record['id']} violates the V17 firewall")

        schema = record["target"]["executable_schema"]
        determinant_ids = tuple(value["id"] for value in record["agent_input"]["determinant_ontology"])
        outcome_bits = record["agent_input"]["dsl_contract"]["outcome_bits"]
        signature = program_signature(schema)
        if list(signature) != record["target"]["behavioral_signature"]:
            symbolic_mismatches += 1
        program_signatures[axis].append(signature)
        family_pairs[axis].add(tuple(record["program_metadata"]["component_families"]))
        maximum_depth[axis].add(record["program_metadata"]["maximum_depth"])
        relevant_counts[axis][len(record["target"]["relevant_determinants"])] += 1
        support_lengths[record["program_metadata"]["support_traces"]] += 1

        agent_support = {
            value["trace_id"]: value for value in record["agent_input"]["support_traces"]
        }
        grounded_support = []
        for grounding in record["oracle_grounding"]["support"]:
            trace = agent_support.get(grounding["trace_id"])
            if trace is None or evaluate_program(schema, grounding["assignment"]) != trace["observed_transition_code"]:
                symbolic_mismatches += 1
                continue
            grounded_support.append({
                "assignment": grounding["assignment"],
                "transition_code": trace["observed_transition_code"],
            })
        cache_key = (determinant_ids, outcome_bits)
        hypotheses = hypothesis_cache.setdefault(
            cache_key, enumerate_program_hypotheses(determinant_ids, outcome_bits)
        )
        consistent = trace_consistent_hypotheses(hypotheses, grounded_support, determinant_ids)
        if len(consistent) != 1 or consistent[0].signature != signature:
            underidentified_schemas += 1

        agent_query_ids = {value["query_id"] for value in record["agent_input"]["queries"]}
        oracle_query_ids = {value["query_id"] for value in record["oracle_grounding"]["queries"]}
        if agent_query_ids != oracle_query_ids:
            errors.append(f"{record['id']} agent and oracle query ids differ")
        for query in record["oracle_grounding"]["queries"]:
            query_ids.append(query["query_id"])
            answer = execute_query(schema, query["allowed_values"])
            for key in ("compatible_assignments", "possible_transition_codes", "identifiable"):
                if answer[key] != query[key]:
                    symbolic_mismatches += 1
            query_effects[axis][query["unknown_effect"]] += 1
            query_labels[axis][query["identifiable"]] += 1

    if leaked_keys:
        errors.append(f"Agent inputs leak target-side schema fields: {dict(leaked_keys)}")
    if symbolic_mismatches:
        errors.append(f"Found {symbolic_mismatches} symbolic target mismatches")
    if underidentified_schemas:
        errors.append(f"Found {underidentified_schemas} schemas not identified by their full support")
    if len(query_ids) != len(set(query_ids)):
        errors.append("Query ids are not globally unique")

    for axis in config["developmentAxes"]:
        if len(program_signatures[axis]) != config["episodesPerDevelopmentAxis"]:
            errors.append(f"Development axis {axis} has the wrong episode count")
        if not {True, False}.issubset(query_labels[axis]):
            errors.append(f"Development axis {axis} lacks an identifiability class")
        if not {"outcome_sensitive", "outcome_invariant"}.issubset(query_effects[axis]):
            errors.append(f"Development axis {axis} lacks a required unresolved-query effect")

    train_signatures = set(program_signatures["training_components"])
    calibration_signatures = set(program_signatures["known_component_calibration"])
    if train_signatures & calibration_signatures:
        errors.append("Training and calibration program behaviors overlap")
    signature_overlaps = {}
    for axis in config["developmentAxes"]:
        overlap = set(program_signatures[axis]) & train_signatures
        signature_overlaps[axis] = len(overlap)
        if axis == "determinant_vocabulary":
            if len(overlap) != config["episodesPerDevelopmentAxis"]:
                errors.append("Vocabulary axis does not exactly reuse training behaviors")
        elif overlap:
            errors.append(f"Development axis {axis} overlaps training behaviors")

    train_family_pairs = family_pairs["training_components"]
    recombination_pairs = family_pairs["known_primitive_recombination"]
    primitive_families = set().union(*train_family_pairs)
    if any(family not in primitive_families for pair in recombination_pairs for family in pair):
        errors.append("Primitive-recombination axis contains an unseen primitive family")
    if recombination_pairs & train_family_pairs:
        errors.append("Primitive-recombination axis repeats a training component pair")
    if maximum_depth["composition_depth"] != {3}:
        errors.append("Composition-depth episodes are not uniformly deeper than training")
    if maximum_depth["structural_composition"] != {2}:
        errors.append("Structural-composition axis is confounded with greater depth")
    if any(value > 3 for value in relevant_counts["outcome_invariance"]):
        errors.append("Outcome-invariance mechanics use all four determinants")
    if any(record["lexicon_family"] != config["heldOutLexicon"] for record in records if record["generalization_axis"] == "determinant_vocabulary"):
        errors.append("Vocabulary axis does not exclusively use the held-out lexicon")
    if any(record["lexicon_family"] == config["heldOutLexicon"] for record in records if record["generalization_axis"] != "determinant_vocabulary"):
        errors.append("Held-out vocabulary appears outside its isolated axis")

    artifact_mismatches = 0
    if manifest is not None and dataset_dir is not None:
        if manifest["data_access"] != {
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "fresh_final_mechanic_created": False,
            "adapter_training_runs": 0,
        }:
            errors.append("Manifest violates the V18 development firewall")
        for relative, expected in manifest["artifact_sha256"].items():
            if sha256_text((dataset_dir / relative).read_text()) != expected:
                artifact_mismatches += 1
        if artifact_mismatches:
            errors.append(f"Found {artifact_mismatches} artifact hash mismatches")

    return {
        "passed": not errors,
        "errors": errors,
        "episodes": len(records),
        "split_counts": dict(split_counts),
        "queries": len(query_ids),
        "support_length_distribution": dict(sorted(support_lengths.items())),
        "agent_input_forbidden_keys": dict(leaked_keys),
        "symbolic_target_mismatches": symbolic_mismatches,
        "underidentified_schemas": underidentified_schemas,
        "development_query_effects": {
            axis: dict(query_effects[axis]) for axis in config["developmentAxes"]
        },
        "development_identifiability_labels": {
            axis: {str(key).lower(): value for key, value in query_labels[axis].items()}
            for axis in config["developmentAxes"]
        },
        "training_signature_overlap": signature_overlaps,
        "family_pairs": {axis: sorted(list(values)) for axis, values in family_pairs.items()},
        "maximum_depth": {axis: sorted(values) for axis, values in maximum_depth.items()},
        "relevant_determinant_counts": {
            axis: dict(sorted(values.items())) for axis, values in relevant_counts.items()
        },
        "artifact_hash_mismatches": artifact_mismatches,
        "data_access": {
            "v17_records_read": 0,
            "v17_model_results_read": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v18.json")
    parser.add_argument("--dataset", default="data/v18")
    parser.add_argument("--output", default="outputs/v18-schema-induction/audit.json")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    dataset_dir = (PROJECT_ROOT / args.dataset).resolve()
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    result = audit(read_records(dataset_dir), config, manifest, dataset_dir)
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
