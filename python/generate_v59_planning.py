#!/usr/bin/env python3
"""Construct the independently split V59 planning population."""
from __future__ import annotations

import argparse
import copy
import json
import random

from generate_v55r1_planning import (
    make_case,
    prior_observation_design_keys,
    public_case,
    simulate_case,
)
from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text, unary_atom
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import entities
from v46_stochastic import _configuration_key
from v53_smc2 import scaled_beta_sample, stream_seed
from v55r1_planning import planning_registry
from v59_planning import assert_search_payload_is_public


def history_class_for_record(record: int) -> str:
    return "prior_like_all_wait" if record % 2 == 0 else "mixed_informative"


def target_assignments(config: dict, registry_size: int = 8) -> list[int]:
    count = config["population"]["tasks"]
    copies = config["population"]["tasksPerGeneratingTemplate"]
    values = [index for index in range(registry_size) for _ in range(copies)]
    if len(values) != count:
        raise RuntimeError("V59 truth allocation does not match task count")
    random.Random(config["population"]["truthAssignmentSeed"]).shuffle(values)
    return values


def goal_assignments(config: dict) -> list[dict]:
    count = config["population"]["tasks"]
    entity_rows = entities(config["planningModel"]["entityCount"])
    copies = count // (2 * len(entity_rows))
    values = [
        {"atom": unary_atom("active", entity["id"]), "value": value}
        for entity in entity_rows for value in (False, True)
        for _ in range(copies)
    ]
    if len(values) != count:
        raise RuntimeError("V59 goal allocation does not match task count")
    random.Random(config["population"]["goalSeed"]).shuffle(values)
    return values


def horizon_assignments(config: dict) -> list[int]:
    values = [
        int(horizon)
        for horizon, count in config["population"]["tasksPerHorizon"].items()
        for _ in range(count)
    ]
    if len(values) != config["population"]["tasks"]:
        raise RuntimeError("V59 horizon allocation does not match task count")
    random.Random(config["population"]["horizonSeed"]).shuffle(values)
    return values


def record_theta(record: int, config: dict) -> float:
    return scaled_beta_sample(stream_seed(
        config["population"]["thetaPriorSeed"],
        "v59", record, "theta-prior",
    ))


def build_record(
    record: int,
    target_index: int,
    goal: dict,
    horizon: int,
    registry: list[dict],
    config: dict,
    used: set[str],
    prior: set[str],
) -> tuple[dict, dict]:
    case_config = copy.deepcopy(config)
    case_config["population"]["entityCount"] = config["planningModel"]["entityCount"]
    history_class = history_class_for_record(record)
    target = registry[target_index]
    theta = record_theta(record, config)
    supports = []
    lengths = config["population"]["supportSequenceLengths"]
    for ordinal in range(config["population"]["supportEpisodesPerTask"]):
        length = lengths[(record + ordinal) % len(lengths)]
        case = make_case(
            record, ordinal, length, history_class, case_config,
            "v59-support", used, prior,
        )
        sampled = simulate_case(
            target, theta, case, case_config,
            f"v59|{record}|support|{ordinal}",
        )
        supports.append(public_case(case, sampled))
    prefixes = config["population"]["ongoingHistoryPrefixLengths"]
    prefix = prefixes[record % len(prefixes)]
    case = make_case(
        record, 0, prefix, history_class, case_config,
        "v59-query", used, prior, query_goal=goal,
    )
    sampled = simulate_case(
        target, theta, case, case_config, f"v59|{record}|query"
    )
    query = public_case(case, sampled)
    query["prefix_length"] = prefix
    public_payload = {
        "supports": supports,
        "query": query,
        "goal": dict(goal),
        "planning_horizon": horizon,
    }
    public_row = {
        "id": f"planning_v59_{record:05d}",
        "schema_version": 59,
        "population": "budgeted_root_sampled_planning",
        "record": record,
        "history_class": history_class,
        "horizon": horizon,
        "public": public_payload,
    }
    assert_search_payload_is_public(public_row)
    audit_row = {
        "id": public_row["id"],
        "schema_version": 59,
        "record": record,
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
    return public_row, audit_row


def build_population(config: dict) -> tuple[list[dict], list[dict]]:
    registry = planning_registry({
        "planningSpecificRegistry": json.loads(
            (PROJECT_ROOT / "configs/v55r1-design-lock.json").read_text()
        )["config_payload"]["planningSpecificRegistry"]
    })
    used, prior = set(), prior_observation_design_keys()
    assignments = target_assignments(config, len(registry))
    goals = goal_assignments(config)
    horizons = horizon_assignments(config)
    pairs = [
        build_record(
            record, assignments[record], goals[record], horizons[record],
            registry, config, used, prior,
        )
        for record in range(config["population"]["tasks"])
    ]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def rows_hash(rows: list[dict]) -> str:
    return sha256_text("".join(canonical_json(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v59-implementation-lock.json"
    )
    parser.add_argument(
        "--output-dir", default="data/v59-budgeted-root-sampled-planning"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["construct_v59_population"]:
        raise RuntimeError("V59 implementation lock does not authorize population construction")
    for section in ("implementation_files_sha256", "base_dependencies_sha256"):
        for relative, digest in lock[section].items():
            if file_sha256(PROJECT_ROOT / relative) != digest:
                raise RuntimeError(f"V59 frozen implementation changed: {relative}")
    design = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())
    public_rows, audit_rows = build_population(design["config_payload"])
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    public_path = output_dir / "public.jsonl"
    audit_path = output_dir / "audit-truth.jsonl"
    public_path.write_text(
        "".join(canonical_json(row) + "\n" for row in public_rows)
    )
    audit_path.write_text(
        "".join(canonical_json(row) + "\n" for row in audit_rows)
    )
    pairing_hash = sha256_text(canonical_json([
        {"public": public["id"], "audit": audit["id"]}
        for public, audit in zip(public_rows, audit_rows, strict=True)
    ]))
    manifest = {
        "schema_version": 59,
        "experiment": "v59_population_manifest",
        "count": len(public_rows),
        "public_file": {
            "path": str(public_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(public_path),
            "rows_sha256": rows_hash(public_rows),
        },
        "audit_truth_file": {
            "path": str(audit_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(audit_path),
            "rows_sha256": rows_hash(audit_rows),
            "candidate_access": "forbidden",
        },
        "public_audit_pairing_sha256": pairing_hash,
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
