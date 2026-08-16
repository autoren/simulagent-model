#!/usr/bin/env python3
"""Build the paired V45 declared-language corpus from sealed V44 cases."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from typing import Any, Sequence

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v43_language import operator_ontology, predicate_ontology, public_entities, render_state
from v44_delayed import execute_partial
from v45_language import action_ontology, episode_aliases, render_action_sequence, safety_challenges


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_records(seal_path) -> list[dict[str, Any]]:
    seal = json.loads(seal_path.read_text())
    rows = []
    for artifact in seal["corpora"].values():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"V44 source corpus changed: {artifact['path']}")
        rows.extend(read(path))
    return sorted(rows, key=lambda row: row["id"])


def transform_record(source: dict[str, Any]) -> dict[str, Any]:
    aliases = episode_aliases(source)
    predicate = predicate_ontology(f"v45|{source['id']}")
    operator, operator_cues = operator_ontology(f"v45|{source['id']}")
    action, action_cues = action_ontology(source["id"])
    public_support, reference_support = [], []
    for support in source["agent_input"]["support_sequences"]:
        entities = public_entities(support["entities"], aliases)
        initial, initial_reference = render_state(
            support["initial_state"], aliases, predicate, operator_cues, f"v45|{support['id']}|initial",
        )
        action_text, action_reference = render_action_sequence(support["actions"], aliases, action_cues, support["id"])
        steps, step_references = [], []
        for index, state_rows in enumerate(support["observed_step_states"], start=1):
            rendered, reference = render_state(
                state_rows, aliases, predicate, operator_cues, f"v45|{support['id']}|step|{index}",
            )
            steps.append(rendered)
            step_references.append(reference)
        language_key = sha256_text(canonical_json({"initial": initial, "actions": action_text}))
        public_support.append({
            "id": support["id"],
            "entities": entities,
            "initial_state_language": initial,
            "action_language": action_text,
            "observed_step_state_language": steps,
            "language_structural_key": language_key,
        })
        reference_support.append({
            "id": support["id"],
            "initial_state": initial_reference,
            "actions": action_reference,
            "observed_step_states": step_references,
        })
    public_queries, reference_queries, oracle_queries = [], [], []
    target_program = source["target"]["program"]
    for query in source["agent_input"]["queries"]:
        entities = public_entities(query["entities"], aliases)
        initial, initial_reference = render_state(
            query["initial_state"], aliases, predicate, operator_cues, f"v45|{query['id']}|initial",
        )
        action_text, action_reference = render_action_sequence(query["actions"], aliases, action_cues, query["id"])
        language_key = sha256_text(canonical_json({"initial": initial, "actions": action_text}))
        public_queries.append({
            "id": query["id"],
            "entities": entities,
            "initial_state_language": initial,
            "action_language": action_text,
            "language_structural_key": language_key,
            **{key: query[key] for key in (
                "sequence_length", "entity_count", "partial_initial_state",
                "wait_counterfactual_group", "wait_counterfactual_role", "wait_placement_effect",
            )},
        })
        reference_queries.append({
            "id": query["id"],
            "initial_state": initial_reference,
            "actions": action_reference,
        })
        oracle_queries.append({
            "id": query["id"],
            "target": execute_partial(
                [target_program], entities, initial_reference["epistemic_state"], action_reference["actions"]
            ),
        })
    catalog = [{"id": alias, "entity_type": "unit"} for alias in sorted(aliases.values())]
    challenges = safety_challenges(catalog[:2], predicate, operator, operator_cues, action, action_cues)
    state_clauses = sum(
        len(sequence["initial_state"]["clauses"])
        + sum(len(step["clauses"]) for step in sequence["observed_step_states"])
        for sequence in reference_support
    ) + sum(len(query["initial_state"]["clauses"]) for query in reference_queries)
    actions = [
        action_row
        for sequence in (*reference_support, *reference_queries)
        for action_row in sequence["actions"]["actions"]
    ]
    return {
        "id": source["id"],
        "schema_version": 45,
        "split": source["split"],
        "construction_family": source["construction_family"],
        "agent_input": {
            "task": "compile_declared_states_and_ordered_actions_then_infer_and_execute_the_delayed_effect_mechanic",
            "entity_catalog": catalog,
            "predicate_ontology": predicate,
            "operator_ontology": operator,
            "action_ontology": action,
            "tick_semantics": source["agent_input"]["tick_semantics"],
            "support_sequences": public_support,
            "queries": public_queries,
            "safety_challenges": challenges,
        },
        "target": source["target"],
        "oracle_queries": oracle_queries,
        "reference": {
            "source_v44_record": source["id"],
            "entity_aliases": aliases,
            "support_sequences": reference_support,
            "queries": reference_queries,
        },
        "oracle_metadata": {
            **source["oracle_metadata"],
            "state_clauses": state_clauses,
            "action_commands": len(actions),
            "bound_action_commands": sum(row["id"] != "wait" for row in actions),
            "wait_commands": sum(row["id"] == "wait" for row in actions),
            "safety_challenges": len(challenges),
        },
    }


def build_population(seal_path) -> list[dict[str, Any]]:
    rows = [transform_record(row) for row in source_records(seal_path)]
    if len(rows) != 40 or len({row["target"]["program_key"] for row in rows}) != 40:
        raise RuntimeError("V45 must pair all 40 unique V44 mechanics")
    return rows


def corpus_hash(rows: Sequence[dict[str, Any]]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def population_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mechanics": len(rows),
        "support_sequences": sum(len(row["agent_input"]["support_sequences"]) for row in rows),
        "query_sequences": sum(len(row["agent_input"]["queries"]) for row in rows),
        "state_clauses": sum(row["oracle_metadata"]["state_clauses"] for row in rows),
        "action_commands": sum(row["oracle_metadata"]["action_commands"] for row in rows),
        "bound_action_commands": sum(row["oracle_metadata"]["bound_action_commands"] for row in rows),
        "wait_commands": sum(row["oracle_metadata"]["wait_commands"] for row in rows),
        "safety_challenges": sum(row["oracle_metadata"]["safety_challenges"] for row in rows),
        "wait_counterfactual_pairs": sum(row["oracle_metadata"]["wait_counterfactual_pairs"] for row in rows),
        "families": dict(Counter(row["construction_family"] for row in rows)),
        "splits": dict(Counter(row["split"] for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v45-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_paired_language_corpus"]:
        raise RuntimeError("V45 implementation lock does not authorize corpus construction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V45 locked implementation changed: {path}")
    output = PROJECT_ROOT / "data/v45-delayed-language-grounding"
    if output.exists():
        raise RuntimeError("V45 paired language corpus already exists")
    source_seal_path = PROJECT_ROOT / lock["source_v44_corpus_seal"]
    rows = build_population(source_seal_path)
    if corpus_hash(rows) != lock["expected_corpus_sha256"]:
        raise RuntimeError("V45 corpus differs from implementation lock")
    output.mkdir(parents=True)
    artifacts = {}
    for split in ("development_fit", "development_evaluation"):
        selected = [row for row in rows if row["split"] == split]
        path = output / f"{split}.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in selected))
        artifacts[split] = {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(selected), "sha256": file_sha256(path)}
    manifest = {
        "schema_version": 45,
        "experiment": lock["config_payload"]["experiment"],
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "source_v44_corpus_seal": str(source_seal_path.relative_to(PROJECT_ROOT)),
        "source_v44_corpus_seal_sha256": file_sha256(source_seal_path),
        "artifacts": artifacts,
        "counts": population_counts(rows),
        "data_access": {
            "v44_records_read": 40,
            "paired_development_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
