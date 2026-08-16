#!/usr/bin/env python3
"""Construct fresh V41 mechanics and their declared-language views."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json, sha256_text, target_hypotheses
from generate_v22_relational_development import build_episode
from v41_interface import language_scene


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def old_program_keys(config: dict[str, Any]) -> set[str]:
    root = PROJECT_ROOT / config["sourceV22DevelopmentCorpus"] / "records"
    return {row["target"]["program_key"] for path in sorted(root.glob("*.jsonl")) for row in read_jsonl(path)}


def select_fresh_targets(config: dict[str, Any], v22_config: dict[str, Any]):
    old = old_program_keys(config)
    selected = []
    for family in config["population"]["families"]:
        ordinal = 0
        for bits, count in ((1, config["population"]["oneBitMechanicsPerFamily"]), (2, config["population"]["twoBitMechanicsPerFamily"])):
            candidates = [row for row in target_hypotheses(family, bits) if row.key not in old]
            candidates.sort(key=lambda row: sha256_text(f"{config['population']['generatorSeed']}|{family}|{bits}|{row.key}"))
            if len(candidates) < count:
                raise RuntimeError(f"Insufficient unseen V41 targets for {family}/{bits}")
            for target in candidates[:count]:
                selected.append((family, ordinal, target))
                ordinal += 1
    if len(selected) != config["population"]["mechanics"] or len({row.key for _, _, row in selected}) != len(selected):
        raise RuntimeError("V41 target selection is not exactly 40 unique mechanics")
    return selected


def build_population(config: dict[str, Any], v22_config: dict[str, Any], v32_config: dict[str, Any]) -> list[dict[str, Any]]:
    generation_config = copy.deepcopy(v22_config)
    generation_config["seed"] = config["population"]["generatorSeed"]
    generation_config["episodesPerFamily"] = config["population"]["mechanicsPerFamily"]
    generation_config["fitEpisodesPerFamily"] = 0
    generation_config["limits"]["maximumSupportTraces"] = config["population"]["maximumSupportTraces"]
    structural_hashes = {
        axis: {"development_fit": set(), "development_evaluation": set()}
        for axis in ("graph_topology", "entity_count_extrapolation")
    }
    episodes = [
        build_episode(generation_config, family, ordinal, target, structural_hashes)
        for family, ordinal, target in select_fresh_targets(config, generation_config)
    ]
    cell_counts: Counter = Counter()
    records = []
    for episode_index, episode in enumerate(episodes):
        references = {"support": [], "queries": []}
        public_support = []
        public_queries = []
        for role, oracle_scenes, public_scenes in (
            ("support", episode["oracle_grounding"]["support"], public_support),
            ("queries", episode["oracle_grounding"]["queries"], public_queries),
        ):
            for scene_index, scene in enumerate(oracle_scenes):
                public, reference = language_scene(scene, v32_config, episode["id"], cell_counts, episode_index * 1000 + scene_index)
                if role == "support":
                    public["observed_transition_code"] = scene["transition_code"]
                public_scenes.append(public)
                references[role].append(reference)
        record = {
            "id": episode["id"].replace("v22:", "v41:"),
            "schema_version": 41,
            "split": "relational_confirmation",
            "construction_family": episode["construction_family"],
            "agent_input": {
                "task": "induce_typed_relational_action_schema_and_answer_queries",
                "entity_types": generation_config["entityTypes"],
                "action_schema": generation_config["action"],
                "dsl_contract": episode["agent_input"]["dsl_contract"],
                "support_traces": public_support,
                "queries": public_queries,
                "output_instruction": episode["agent_input"]["output_instruction"],
            },
            "oracle_grounding": episode["oracle_grounding"],
            "language_reference": references,
            "target": episode["target"],
            "search": episode["search"],
            "oracle_metadata": {
                "source_episode_id": episode["id"],
                "construction_family": episode["construction_family"],
                "outcome_bits": len(episode["target"]["program"]["output_bits"]),
                "target_was_absent_from_v22": episode["target"]["program_key"] not in old_program_keys(config),
            },
        }
        records.append(record)
    if len(records) != 40 or not all(row["oracle_metadata"]["target_was_absent_from_v22"] for row in records):
        raise RuntimeError("V41 population violates fresh-mechanic requirement")
    return sorted(records, key=lambda row: row["id"])


def corpus_hash(rows):
    return sha256_text("".join(canonical_json(row) + "\n" for row in sorted(rows, key=lambda row: row["id"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v41-implementation-lock.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_confirmation"]:
        raise RuntimeError("V41 implementation lock does not authorize construction")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V41 locked implementation changed: {path}")
    output = PROJECT_ROOT / "data/v41-relational-mechanic-confirmation"
    if output.exists():
        raise RuntimeError("V41 confirmation corpus already exists")
    rows = build_population(lock["config_payload"], lock["v22_config_payload"], lock["v32_config_payload"])
    if corpus_hash(rows) != lock["expected_corpus_sha256"]:
        raise RuntimeError("V41 confirmation corpus differs from implementation lock")
    output.mkdir(parents=True)
    path = output / "relational_confirmation.jsonl"
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))
    manifest = {
        "schema_version": 41,
        "experiment": lock["config_payload"]["experiment"],
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "artifact": {"path": str(path.relative_to(PROJECT_ROOT)), "records": len(rows), "sha256": file_sha256(path)},
        "counts": {
            "families": dict(Counter(row["construction_family"] for row in rows)),
            "outcome_bits": dict(Counter(str(row["oracle_metadata"]["outcome_bits"]) for row in rows)),
            "support_scenes": sum(len(row["agent_input"]["support_traces"]) for row in rows),
            "query_scenes": sum(len(row["agent_input"]["queries"]) for row in rows),
            "language_clauses": sum(len(scene["evidence_packets"]) for row in rows for scene in row["agent_input"]["support_traces"] + row["agent_input"]["queries"]),
        },
        "data_access": {"confirmation_scoring_runs": 0, "model_forward_passes": 0, "v22r2_evaluation_records_read": 0, "v28_runs": 0},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
