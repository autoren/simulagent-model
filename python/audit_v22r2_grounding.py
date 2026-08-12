"""Pre-model structural, semantic, and firewall audit for V22r2."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from audit_v22_relational import audit as audit_v22
from audit_v22_relational import read_records as read_v22_records
from v10_protocol import file_sha256
from v22_relational import execute_partial, parse_atom, rows_to_epistemic
from v22r2_grounding import (
    ENTITY_ALIASES,
    PROJECT_ROOT,
    canonical_scene_graph,
    scene_prompt_text,
)


FORBIDDEN_PUBLIC_KEYS = {
    "query_axis", "generalization_axis", "construction_family", "source_item_id",
    "program", "program_key", "program_key_sha256", "epistemic_state",
    "reference_complete_world", "semantic_signatures", "atom_groundings",
    "allowed_values", "truth_label", "possible_transition_codes", "identifiable",
    "counterfactual_group", "counterfactual_role", "metamorphic_group",
    "metamorphic_role", "canonical_state_binding_hash",
}
SEQUENTIAL_ID = re.compile(r"(?:query|support|trace)[:_-]?\d+$", re.IGNORECASE)
OPAQUE_ID = re.compile(r"^(?:episode|item|atom|ev)_[0-9a-f]{16,20}$")


def read_jsonl_directory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def state_world(row: dict[str, Any]) -> dict[str, bool]:
    return {value["atom"]: bool(value["value"]) for value in row["reference_complete_world"]}


def signature_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["predicate_kind"], row["truth_label"], row["semantic_operator"],
        row["relation_orientation"],
    )


def surface_and_prompt_metrics(
    scenes: Sequence[dict[str, Any]], config: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    banks: dict[str, set[str]] = defaultdict(set)
    cells: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    cell_counts: dict[str, Counter[tuple[Any, ...]]] = defaultdict(Counter)
    prompts: dict[str, set[str]] = defaultdict(set)
    maximum_words = 0
    for scene in scenes:
        split = scene["split"]
        prompt = scene_prompt_text(scene)
        prompts[split].add(prompt)
        maximum_words = max(maximum_words, len(prompt.split()))
        for row in scene["target"]["atom_groundings"]:
            banks[split].add(row["surface_bank"])
            cells[split].add(signature_key(row))
            cell_counts[split][signature_key(row)] += 1
    fit_banks = banks["grounding_fit"] | banks["grounding_calibration"]
    evaluation_banks = banks["grounding_evaluation"]
    if fit_banks != set(config["language"]["fitSurfaceBanks"]):
        errors.append(f"Fit surface-bank coverage differs from registration: {fit_banks}")
    if evaluation_banks != set(config["language"]["evaluationSurfaceBanks"]):
        errors.append(f"Evaluation surface-bank coverage differs from registration: {evaluation_banks}")
    if fit_banks & evaluation_banks:
        errors.append("Fit and evaluation surface banks overlap")
    missing = cells["grounding_evaluation"] - cells["grounding_fit"]
    if missing:
        errors.append(f"Evaluation contains unsupported semantic cells: {sorted(missing, key=str)}")
    exact_overlap = prompts["grounding_fit"] & prompts["grounding_evaluation"]
    if exact_overlap:
        errors.append("Exact full-scene prompts overlap between fit and evaluation")
    return {
        "surface_banks": {key: sorted(value) for key, value in sorted(banks.items())},
        "semantic_cell_counts": {
            split: {str(key): count for key, count in sorted(counts.items(), key=lambda row: str(row[0]))}
            for split, counts in sorted(cell_counts.items())
        },
        "evaluation_cells_missing_from_fit": len(missing),
        "exact_fit_evaluation_prompt_overlap": len(exact_overlap),
        "unique_full_scene_prompts": {key: len(value) for key, value in sorted(prompts.items())},
        "new_model_forward_passes": len(scenes),
        "maximum_prompt_whitespace_words": maximum_words,
    }


def public_interface_metrics(
    records: Sequence[dict[str, Any]], scenes: Sequence[dict[str, Any]], errors: list[str]
) -> dict[str, Any]:
    public_axis_fields = 0
    sequential_ids = []
    invalid_opaque_ids = []
    entity_ids = set()
    candidate_sorted = 0
    evidence_sorted = 0
    entity_sorted = 0
    item_count = 0
    all_ids = []
    for record in records:
        keys = set(recursive_keys(record["agent_input"]))
        leaked = keys & FORBIDDEN_PUBLIC_KEYS
        public_axis_fields += len(keys & {"query_axis", "generalization_axis"})
        if leaked:
            errors.append(f"Forbidden public fields in {record['id']}: {sorted(leaked)}")
        all_ids.append(record["id"])
    for scene in scenes:
        public = scene["agent_input"]
        all_ids.append(scene["id"])
        item_count += 1
        candidates = [row["id"] for row in public["atom_candidates"]]
        evidence = [row["id"] for row in public["evidence"]]
        entities = [row["id"] for row in public["entities"]]
        all_ids.extend(candidates + evidence)
        entity_ids.update(entities)
        candidate_sorted += candidates == sorted(candidates)
        evidence_sorted += evidence == sorted(evidence)
        entity_sorted += entities == sorted(entities)
        if len(candidates) != len(evidence) or len(candidates) != len(scene["target"]["atom_groundings"]):
            errors.append(f"Scene inventory cardinality mismatch in {scene['id']}")
        if len(set(candidates)) != len(candidates) or len(set(evidence)) != len(evidence):
            errors.append(f"Repeated public IDs in {scene['id']}")
    for identifier in all_ids:
        if SEQUENTIAL_ID.search(identifier):
            sequential_ids.append(identifier)
        if not OPAQUE_ID.match(identifier):
            invalid_opaque_ids.append(identifier)
    if public_axis_fields:
        errors.append(f"Found {public_axis_fields} public axis fields")
    if sequential_ids:
        errors.append(f"Sequential public IDs remain: {sequential_ids[:3]}")
    if invalid_opaque_ids:
        errors.append(f"Non-opaque public IDs remain: {invalid_opaque_ids[:3]}")
    if not entity_ids <= set(ENTITY_ALIASES):
        errors.append("Public entity identifiers leave the registered opaque alias inventory")
    for name, count in (
        ("candidate", candidate_sorted), ("evidence", evidence_sorted), ("entity", entity_sorted)
    ):
        if count / item_count > 0.6:
            errors.append(f"Too many {name} inventories remain lexically sorted: {count}/{item_count}")
    return {
        "public_axis_fields": public_axis_fields,
        "sequential_public_ids": len(sequential_ids),
        "invalid_opaque_public_ids": len(invalid_opaque_ids),
        "opaque_entity_aliases_used": len(entity_ids),
        "lexically_sorted_fraction": {
            "candidate": candidate_sorted / item_count,
            "evidence": evidence_sorted / item_count,
            "entity": entity_sorted / item_count,
        },
    }


def counterfactual_metrics(records: Sequence[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    orientation_pairs = 0
    topology_pairs = 0
    orientation_exact = 0
    topology_exact = 0
    for record in records:
        queries = record["oracle_grounding"]["queries"]
        orientation = [row for row in queries if row["query_axis"] == "relation_orientation"]
        topology = [row for row in queries if row["query_axis"] == "graph_topology"]
        if len(orientation) != 2 or len(topology) != 2:
            errors.append(f"Missing registered counterfactual pair in {record['id']}")
            continue
        orientation_pairs += 1
        left, right = orientation
        if left["entities"] != right["entities"] or left["action_binding"] != right["action_binding"]:
            errors.append(f"Orientation pair context differs in {record['id']}")
        lw, rw = state_world(left), state_world(right)
        changed = {atom for atom in lw if lw[atom] != rw[atom]}
        binding = left["action_binding"]
        expected = {
            f"r:linked:{binding['actor']}:{binding['target']}",
            f"r:linked:{binding['target']}:{binding['actor']}",
        }
        if changed == expected:
            orientation_exact += 1
        else:
            errors.append(f"Orientation pair changes {sorted(changed)} in {record['id']}")
        topology_pairs += 1
        left, right = topology
        if left["entities"] != right["entities"] or left["action_binding"] != right["action_binding"]:
            errors.append(f"Topology pair context differs in {record['id']}")
        lw, rw = state_world(left), state_world(right)
        changed = {atom for atom in lw if lw[atom] != rw[atom]}
        if changed and all(parse_atom(atom)[:2] == ("r", "linked") for atom in changed):
            topology_exact += 1
        else:
            errors.append(f"Topology pair has non-isolated changes in {record['id']}: {sorted(changed)}")
    return {
        "orientation_pairs": orientation_pairs,
        "orientation_exact_background_matches": orientation_exact,
        "topology_pairs": topology_pairs,
        "topology_linked_only_background_matches": topology_exact,
    }


def oracle_metrics(
    records: Sequence[dict[str, Any]], v22_config: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    queries = 0
    exact = 0
    structural_graphs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    within_episode_overlap = 0
    for record in records:
        program = record["target"]["program"]
        outcome_bits = record["agent_input"]["dsl_contract"]["outcome_bits"]
        support_hashes = {
            row["canonical_state_binding_hash"] for row in record["oracle_grounding"]["support"]
        }
        query_hashes = {
            row["canonical_state_binding_hash"] for row in record["oracle_grounding"]["queries"]
        }
        overlap = support_hashes & query_hashes
        within_episode_overlap += len(overlap)
        if overlap:
            errors.append(f"Support/query graphs overlap within {record['id']}")
        for role in ("support", "queries"):
            key = "support" if role == "support" else "queries"
            for row in record["oracle_grounding"][key]:
                if role == "support":
                    continue
                if row["query_axis"] in {"graph_topology", "entity_count_extrapolation"}:
                    structural_graphs[row["query_axis"]][record["split"]].add(
                        row["canonical_state_binding_hash"]
                    )
                queries += 1
                state = rows_to_epistemic(row["epistemic_state"])
                answer = execute_partial(
                    [program], v22_config, row["entities"], state, row["action_binding"],
                    v22_config["limits"]["maximumUnknownAtomsPerQuery"],
                )
                if answer["possible_transition_codes"] == row["possible_transition_codes"]:
                    exact += 1
                else:
                    errors.append(f"Recomputed oracle answer differs in {row['id']}")
                if any(len(code.rsplit("_", 1)[-1]) != outcome_bits for code in row["possible_transition_codes"]):
                    errors.append(f"Outcome width mismatch in {row['id']}")
    overlaps = {}
    unique = {}
    for axis, by_split in structural_graphs.items():
        development = by_split["grounding_fit"] | by_split["grounding_calibration"]
        overlap = development & by_split["grounding_evaluation"]
        overlaps[axis] = len(overlap)
        unique[axis] = {key: len(value) for key, value in sorted(by_split.items())}
        if overlap:
            errors.append(f"{axis} graphs overlap across development/evaluation: {len(overlap)}")
    return {
        "queries": queries,
        "recomputed_answer_exact": exact / queries,
        "support_query_graph_overlap_within_episode": within_episode_overlap,
        "registered_structural_axis_split_overlap": overlaps,
        "registered_structural_axis_unique_graphs": unique,
    }


def audit(
    records: Sequence[dict[str, Any]], scenes: Sequence[dict[str, Any]],
    config: dict[str, Any], v22_config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    split_counts = Counter(row["split"] for row in records)
    scene_split_counts = Counter(row["split"] for row in scenes)
    if split_counts != Counter({
        "grounding_fit": 8, "grounding_calibration": 4, "grounding_evaluation": 12,
    }):
        errors.append(f"Episode split counts differ from registration: {split_counts}")
    if len(scenes) != 384 or Counter(row["role"] for row in scenes) != Counter({"support": 72, "query": 312}):
        errors.append("Scene population differs from the registered 72 support / 312 query items")
    scene_ids = {row["id"] for row in scenes}
    record_item_ids = {
        row["id"] for record in records
        for role in ("support_traces", "queries")
        for row in record["agent_input"][role]
    }
    if scene_ids != record_item_ids:
        errors.append("Scene IDs do not exactly match public episode item IDs")
    public = public_interface_metrics(records, scenes, errors)
    counterfactual = counterfactual_metrics(records, errors)
    surface = surface_and_prompt_metrics(scenes, config, errors)
    oracle = oracle_metrics(records, v22_config, errors)
    if surface["new_model_forward_passes"] > config["gates"]["preExtraction"]["maximumNewModelForwardPasses"]:
        errors.append("Registered forward-pass budget is exceeded")
    return {
        "schema_version": "22r2",
        "passed": not errors,
        "decision": (
            "authorize_v22r2_protocol_lock" if not errors else
            "repair_v22r2_before_model_access"
        ),
        "errors": errors,
        "population": {
            "episodes": len(records), "scenes": len(scenes),
            "episode_split_counts": dict(sorted(split_counts.items())),
            "scene_split_counts": dict(sorted(scene_split_counts.items())),
        },
        "public_interface": public,
        "controlled_counterfactuals": counterfactual,
        "surface_and_prompts": surface,
        "oracle_reproduction": oracle,
        "firewall": {
            "new_model_forward_passes": 0,
            "model_predictions_read": 0,
            "adapter_training_runs": 0,
            "final_suite_records_read": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v22r2.json")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default="outputs/v22r2-relational-grounding/pre-extraction-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    dataset = (PROJECT_ROOT / (args.dataset or config["outputDir"])).resolve()
    records = read_jsonl_directory(dataset / "records")
    scenes = read_jsonl_directory(dataset / "scenes")
    v22_config = json.loads((PROJECT_ROOT / config["sourceV22Config"]).read_text())
    result = audit(records, scenes, config, v22_config)
    manifest = json.loads((dataset / "manifest.json").read_text())
    source_checks = {
        "v22_config": file_sha256(PROJECT_ROOT / config["sourceV22Config"]),
        "v22_manifest": file_sha256(PROJECT_ROOT / config["sourceV22Manifest"]),
        "v22_audit": file_sha256(PROJECT_ROOT / config["sourceV22Audit"]),
        "v22_result": file_sha256(PROJECT_ROOT / config["sourceV22Result"]),
    }
    manifest_checks = {
        "v22_config": manifest["source"]["v22_config_sha256"],
        "v22_manifest": manifest["source"]["v22_manifest_sha256"],
        "v22_audit": manifest["source"]["v22_audit_sha256"],
        "v22_result": manifest["source"]["v22_result_sha256"],
    }
    result["source_hashes"] = source_checks
    if source_checks != manifest_checks:
        result["errors"].append("V22 source hashes differ from the V22r2 manifest")
    source_records = read_v22_records(PROJECT_ROOT / config["sourceV22Dataset"])
    source_audit = audit_v22(source_records, v22_config)
    source_result = json.loads((PROJECT_ROOT / config["sourceV22Result"]).read_text())
    result["source_v22_reproduction"] = {
        "structural_audit_passed": source_audit["passed"],
        "oracle_result_passed": source_result["passed"],
        "oracle_decision": source_result["decision"],
    }
    if not source_audit["passed"] or not source_result["passed"]:
        result["errors"].append("Immutable V22 source no longer passes its registered audits")
    feature_dir = PROJECT_ROOT / "outputs/v22r2-relational-grounding/features"
    evaluation_dir = PROJECT_ROOT / "outputs/v22r2-relational-grounding/evaluation"
    if feature_dir.exists() or evaluation_dir.exists():
        result["errors"].append("V22r2 model artifacts exist before protocol lock")
    result["passed"] = not result["errors"]
    result["decision"] = (
        "authorize_v22r2_protocol_lock" if result["passed"] else
        "repair_v22r2_before_model_access"
    )
    output = (PROJECT_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
