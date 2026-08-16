#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter

from generate_v55_planning import (
    goal_for_record,
    goal_values,
    history_class_for_record,
    observation_design_key,
    population_hash,
    prior_observation_design_keys,
    target_assignments,
)
from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v46_stochastic import _configuration_key
from v53_smc2 import (
    continuous_sequential_filter,
    instantiate_program,
    mechanic_registry,
)
from v55_planning import assert_planning_payload_is_public


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/v55-short-horizon-bayes-adaptive-planning/manifest.json",
    )
    parser.add_argument(
        "--implementation-lock", default="configs/v55-implementation-lock.json"
    )
    parser.add_argument(
        "--evaluation-implementation-lock",
        default="configs/v55-evaluation-implementation-lock.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/population-audit.json",
    )
    args = parser.parse_args()
    manifest_path, implementation_path, evaluation_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (
            args.manifest, args.implementation_lock,
            args.evaluation_implementation_lock, args.output,
        )
    )
    manifest = json.loads(manifest_path.read_text())
    implementation = json.loads(implementation_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    errors = []

    locks_ok = (
        implementation["authorization"]["construct_v55_planning_population"]
        and not implementation["authorization"]["run_v55_planning_evaluation"]
        and evaluation["authorization"]["construct_v55_planning_population"]
        and not evaluation["authorization"]["run_v55_planning_evaluation"]
        and evaluation["implementation_lock_sha256"] == file_sha256(implementation_path)
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for lock in (implementation, evaluation)
            for section in ("implementation_files_sha256", "base_dependencies_sha256")
            for path, digest in lock.get(section, {}).items()
        )
    )
    if not locks_ok:
        errors.append("V55 implementation or evaluation lock is not intact")

    path = PROJECT_ROOT / manifest["file"]["path"]
    rows = read_jsonl(path)
    files_ok = (
        file_sha256(path) == manifest["file"]["sha256"]
        and population_hash(rows) == manifest["population_hash"]
        and manifest["implementation_lock_sha256"] == file_sha256(implementation_path)
    )
    if not files_ok:
        errors.append("V55 population file or manifest binding failed")

    count_ok = len(rows) == manifest["count"] == config["population"]["planningTasks"]
    ids = [row["id"] for row in rows]
    ids_ok = len(ids) == len(set(ids))
    if not count_ok or not ids_ok:
        errors.append("V55 task count or identifier uniqueness failed")

    public_ok = True
    truth_nested = True
    schema_ok = True
    design_keys = []
    history_keys = []
    classes = Counter()
    programs = Counter()
    booleans = Counter()
    assignments = target_assignments(config)
    values = goal_values(config)
    deterministic_assignment_ok = True
    for row in rows:
        try:
            assert_planning_payload_is_public(row["public"])
        except PermissionError:
            public_ok = False
        truth_nested &= "truth" in row and not any(
            key.startswith("target_") or key.startswith("query_")
            for key in row if key != "truth"
        )
        record = row["record"]
        classes[row["history_class"]] += 1
        programs[row["truth"]["target_program_index"]] += 1
        booleans[row["public"]["goal"]["value"]] += 1
        query = row["public"]["query"]
        expected_goal = goal_for_record(
            record, query["entities"], config, values[record]
        )
        deterministic_assignment_ok &= (
            row["history_class"] == history_class_for_record(record)
            and row["truth"]["target_program_index"] == assignments[record]
            and row["public"]["goal"] == expected_goal
        )
        history_keys.append(sha256_text(canonical_json(row["public"])))
        for episode in [*row["public"]["supports"], query]:
            design_keys.append(observation_design_key(episode))
            expected_length = episode["sequence_length"]
            schema_ok &= (
                len(episode["actions"]) == expected_length
                and len(episode["masks"]) == expected_length
                and len(episode["observations"]) == expected_length
                and episode["entity_count"] == config["population"]["entityCount"]
                and all(
                    set(value) == {"atom", "value"}
                    for step in episode["observations"] for value in step
                )
            )
    allocation_ok = (
        classes == Counter(config["population"]["historyClasses"])
        and programs == Counter({index: 4 for index in range(8)})
        and booleans == Counter({False: 16, True: 16})
        and deterministic_assignment_ok
    )
    if not public_ok or not truth_nested or not schema_ok or not allocation_ok:
        errors.append("V55 public boundary, schema, or frozen allocation failed")

    prior = prior_observation_design_keys()
    freshness_ok = (
        len(design_keys) == len(set(design_keys))
        and not (set(design_keys) & prior)
        and len(history_keys) == len(set(history_keys))
    )
    if not freshness_ok:
        errors.append("V55 public histories or observation designs are reused")

    registry = mechanic_registry(5303)
    truth_ok = True
    for row in rows:
        truth = row["truth"]
        program = instantiate_program(
            registry[truth["target_program_index"]]["template"],
            truth["target_theta"],
        )
        for episode in row["public"]["supports"]:
            world = {
                item["atom"]: item["allowed_values"][0]
                for item in episode["initial_state"]
            }
            likelihood, _ = continuous_sequential_filter(
                program, episode["entities"], world,
                episode["actions"], episode["observations"],
            )
            truth_ok &= bool(likelihood)
        query = row["public"]["query"]
        world = {
            item["atom"]: item["allowed_values"][0]
            for item in query["initial_state"]
        }
        likelihood, configurations = continuous_sequential_filter(
            program, query["entities"], world,
            query["actions"], query["observations"],
        )
        possible = {
            _configuration_key(value["world"], value["queue"])
            for value in configurations.values()
        }
        truth_ok &= (
            bool(likelihood) and truth["query_configuration_key"] in possible
        )
    if not truth_ok:
        errors.append("V55 generating truth has zero likelihood or lost configuration")

    stream_roots = [
        config["population"][key]
        for key in (
            "generatorSeed", "historySeed", "truthAssignmentSeed", "goalSeed",
            "thetaPriorSeed", "trajectorySeed", "tieBreakAuditSeed",
            "policyEvaluationSeed",
        )
    ]
    streams_ok = len(stream_roots) == len(set(stream_roots))
    if not streams_ok:
        errors.append("V55 population stream roots collide")

    evaluation_absent = not any(
        (PROJECT_ROOT / item).exists()
        for item in (
            "configs/v55-population-seal.json",
            "configs/v55-outcome-lock.json",
            "outputs/v55-short-horizon-bayes-adaptive-planning/evaluation-attempt.json",
            "outputs/v55-short-horizon-bayes-adaptive-planning/evaluation",
        )
    )
    if not evaluation_absent:
        errors.append("V55 evaluation or downstream lock exists before population seal")

    audit = {
        "schema_version": 55,
        "experiment": "v55_population_audit",
        "passed": not errors,
        "decision": (
            "authorize_v55_population_seal" if not errors
            else "reject_v55_planning_population"
        ),
        "errors": errors,
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "evaluation_implementation_lock": str(evaluation_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_path),
        "checks": {
            "locks_intact_and_pre_evaluation": locks_ok,
            "manifest_and_population_hashes": files_ok,
            "task_count_and_unique_ids": count_ok and ids_ok,
            "public_boundary_schema_and_allocation": (
                public_ok and truth_nested and schema_ok and allocation_ok
            ),
            "fresh_unique_histories_and_designs": freshness_ok,
            "generating_truth_likelihood_and_configuration_retention": truth_ok,
            "independent_stream_roots": streams_ok,
            "evaluation_absent": evaluation_absent,
        },
        "counts": {
            "planning_tasks": len(rows),
            "history_classes": dict(classes),
            "generating_programs": dict(sorted(programs.items())),
            "goal_booleans": {str(key).lower(): value for key, value in booleans.items()},
            "unique_public_histories": len(set(history_keys)),
            "unique_observation_designs": len(set(design_keys)),
        },
        "data_access": {
            "planning_evaluation_runs": 0,
            "planning_metrics_computed": 0,
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
