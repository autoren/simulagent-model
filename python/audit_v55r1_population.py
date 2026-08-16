#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generate_v55r1_planning import (
    goal_assignments,
    observation_design_key,
    population_hash,
    target_assignments,
)
from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text, unary_atom
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _visit_designs(value, result: set[str]) -> None:
    if isinstance(value, dict):
        if {"entities", "initial_state", "actions", "masks"} <= set(value):
            result.add(observation_design_key(value))
        for child in value.values():
            _visit_designs(child, result)
    elif isinstance(value, list):
        for child in value:
            _visit_designs(child, result)


def prior_designs(candidate_path: Path) -> set[str]:
    result: set[str] = set()
    for version in range(46, 56):
        for path in sorted((PROJECT_ROOT / "data").glob(f"v{version}*/*.jsonl")):
            if path.resolve() == candidate_path.resolve():
                continue
            for line in path.read_text().splitlines():
                if line.strip():
                    _visit_designs(json.loads(line), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population",
        default="data/v55r1-delayed-consequence-adequacy-confirmation/planning.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="data/v55r1-delayed-consequence-adequacy-confirmation/manifest.json",
    )
    parser.add_argument(
        "--implementation-lock", default="configs/v55r1-implementation-lock.json"
    )
    parser.add_argument(
        "--output",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/population-audit.json",
    )
    args = parser.parse_args()
    population_path, manifest_path, lock_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.population, args.manifest, args.implementation_lock, args.output)
    )
    rows = read_jsonl(population_path)
    manifest = json.loads(manifest_path.read_text())
    lock = json.loads(lock_path.read_text())
    design = json.loads((PROJECT_ROOT / lock["design_lock"]).read_text())
    config = design["config_payload"]
    errors: list[str] = []

    lock_bound = (
        lock["authorization"]["construct_v55r1_population"]
        and not lock["authorization"]["run_v55r1_evaluation"]
        and file_sha256(PROJECT_ROOT / lock["design_lock"])
        == lock["design_lock_sha256"]
        and file_sha256(PROJECT_ROOT / lock["implementation_audit"])
        == lock["implementation_audit_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for section in ("implementation_files_sha256", "base_dependencies_sha256")
            for path, digest in lock[section].items()
        )
    )
    if not lock_bound:
        errors.append("V55r1 implementation lock or frozen dependencies changed")

    manifest_ok = (
        manifest["count"] == len(rows) == 16
        and manifest["file"]["path"] == str(population_path.relative_to(PROJECT_ROOT))
        and manifest["file"]["sha256"] == file_sha256(population_path)
        and manifest["population_hash"] == population_hash(rows)
        and manifest["implementation_lock"] == str(lock_path.relative_to(PROJECT_ROOT))
        and manifest["implementation_lock_sha256"] == file_sha256(lock_path)
    )
    if not manifest_ok:
        errors.append("V55r1 manifest is not bound to the candidate population")

    ids_ok = (
        [row["record"] for row in rows] == list(range(16))
        and len({row["id"] for row in rows}) == 16
        and all(row["revision"] == "r1" for row in rows)
    )
    if not ids_ok:
        errors.append("V55r1 record identifiers or order are invalid")

    expected_targets = target_assignments(config)
    expected_goals = goal_assignments(config)
    allocation_ok = (
        [row["truth"]["target_program_index"] for row in rows] == expected_targets
        and [row["public"]["goal"] for row in rows] == expected_goals
        and Counter(expected_targets) == Counter({index: 2 for index in range(8)})
        and Counter(row["history_class"] for row in rows)
        == Counter(config["population"]["historyClasses"])
        and Counter(
            (row["public"]["goal"]["atom"], row["public"]["goal"]["value"])
            for row in rows
        ) == Counter({
            (unary_atom("active", "unit_0"), False): 4,
            (unary_atom("active", "unit_0"), True): 4,
            (unary_atom("active", "unit_1"), False): 4,
            (unary_atom("active", "unit_1"), True): 4,
        })
        and config["population"]["truthAssignmentSeed"]
        != config["population"]["goalSeed"]
        and not config["population"]["goalDependsOnGeneratingTruth"]
    )
    if not allocation_ok:
        errors.append("V55r1 truth, history, or truth-independent goal quotas failed")

    initial_goal_state_ok = True
    for row in rows:
        query, goal = row["public"]["query"], row["public"]["goal"]
        initial = {
            item["atom"]: item["allowed_values"][0]
            for item in query["initial_state"]
        }
        initial_goal_state_ok &= all(
            initial[unary_atom("active", entity["id"])] is not goal["value"]
            for entity in query["entities"]
        )
    if not initial_goal_state_ok:
        errors.append("V55r1 query active atoms are not opposite the public goal")

    designs = []
    for row in rows:
        for episode in [*row["public"]["supports"], row["public"]["query"]]:
            designs.append(observation_design_key(episode))
    prior = prior_designs(population_path)
    freshness_ok = (
        len(designs) == 80
        and len(set(designs)) == 80
        and not (set(designs) & prior)
        and all(
            episode["observation_design_key"] == observation_design_key(episode)
            for row in rows
            for episode in [*row["public"]["supports"], row["public"]["query"]]
        )
    )
    if not freshness_ok:
        errors.append("V55r1 public observation designs are duplicated or not fresh")

    public_history_hashes = {
        sha256_text(canonical_json(row["public"])) for row in rows
    }
    public_ok = (
        len(public_history_hashes) == 16
        and all(len(row["public"]["supports"]) == 4 for row in rows)
        and all("truth" in row and "truth" not in row["public"] for row in rows)
    )
    if not public_ok:
        errors.append("V55r1 public histories are duplicated or expose truth")

    evaluation_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v55r1-population-seal.json",
            "configs/v55r1-evaluation-implementation-lock.json",
            "configs/v55r1-outcome-lock.json",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation-attempt.json",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation",
        )
    )
    if not evaluation_absent:
        errors.append("V55r1 evaluation or downstream lock exists before population seal")

    audit = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_population_audit",
        "passed": not errors,
        "decision": (
            "authorize_v55r1_population_seal" if not errors
            else "invalidate_v55r1_population"
        ),
        "errors": errors,
        "population": str(population_path.relative_to(PROJECT_ROOT)),
        "population_sha256": file_sha256(population_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(lock_path),
        "checks": {
            "implementation_lock_and_dependencies_bound": lock_bound,
            "manifest_and_population_hash": manifest_ok,
            "record_ids_and_order": ids_ok,
            "truth_history_and_truth_independent_goal_quotas": allocation_ok,
            "query_initial_goal_state": initial_goal_state_ok,
            "fresh_unique_observation_designs": freshness_ok,
            "unique_public_histories_and_truth_separation": public_ok,
            "evaluation_absent": evaluation_absent,
        },
        "metrics": {
            "tasks": len(rows),
            "unique_observation_designs": len(set(designs)),
            "unique_public_histories": len(public_history_hashes),
            "history_class_counts": dict(Counter(
                row["history_class"] for row in rows
            )),
            "truth_program_counts": {
                str(key): value for key, value in sorted(Counter(
                    row["truth"]["target_program_index"] for row in rows
                ).items())
            },
            "goal_counts": {
                f"{atom}|{value}": count
                for (atom, value), count in sorted(Counter(
                    (row["public"]["goal"]["atom"], row["public"]["goal"]["value"])
                    for row in rows
                ).items())
            },
            "prior_design_overlap": len(set(designs) & prior),
        },
        "data_access": {
            "candidate_records_audited": len(rows),
            "planning_evaluation_runs": 0,
            "additional_v55_planning_evaluation_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
