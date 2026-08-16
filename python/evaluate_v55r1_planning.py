#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import exact_inference
from v54_eig import belief_atoms_from_exact
from v55_planning import (
    candidate_actions,
    evaluate_policy,
    plan_exact,
    scalar_plan,
    step_belief,
)
from v55r1_planning import delay_suppressed_registry, planning_registry


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def policy_tree_checks(
    atoms, policy, registry, entity_rows, goal, horizon, tick, config
) -> dict:
    if horizon == 0:
        return {
            "normalization_checks": 1,
            "normalization_passes": 1,
            "candidate_omissions": 0,
            "tie_break_violations": 0,
            "finite_checks": 1,
            "finite_passes": int(math.isfinite(policy["value"])),
        }
    expected_keys = {row["key"] for row in candidate_actions(entity_rows)}
    action_values = policy["action_values"]
    maximum = max(action_values.values())
    tolerance = config["planningModel"]["tieTolerance"]
    tied = sorted(
        key for key, value in action_values.items()
        if value >= maximum - tolerance
    )
    result = {
        "normalization_checks": 0,
        "normalization_passes": 0,
        "candidate_omissions": len(expected_keys - set(action_values)),
        "tie_break_violations": int(
            not tied or policy["selected_action_key"] != tied[0]
        ),
        "finite_checks": 1 + len(action_values),
        "finite_passes": sum(math.isfinite(value) for value in [
            policy["value"], *action_values.values()
        ]),
    }
    branches = step_belief(
        atoms, registry, entity_rows, policy["selected_action"], tick, {}
    )
    result["normalization_checks"] += 1 + len(branches)
    result["normalization_passes"] += int(abs(sum(
        branch["probability"] for branch in branches.values()
    ) - 1.0) <= 1e-12)
    result["normalization_passes"] += sum(
        abs(sum(atom["weight"] for atom in branch["atoms"]) - 1.0) <= 1e-12
        for branch in branches.values()
    )
    for outcome, branch in branches.items():
        if outcome not in policy["branches"]:
            result["candidate_omissions"] += 1
            continue
        child = policy_tree_checks(
            branch["atoms"], policy["branches"][outcome], registry,
            entity_rows, goal, horizon - 1, tick + 1, config,
        )
        for key in result:
            result[key] += child[key]
    return result


def evaluate_record(public_record, registry, v53_config, config, suppressed):
    public = public_record["public"]
    query, goal = public["query"], public["goal"]
    entity_rows = query["entities"]
    tick = query["prefix_length"]
    horizon = config["planningModel"]["horizonActions"]
    exact = exact_inference(
        registry,
        {"supports": public["supports"], "query": query},
        v53_config,
    )
    atoms = belief_atoms_from_exact(exact)
    primary = plan_exact(
        atoms, registry, entity_rows, goal, horizon, tick, config
    )
    reference = scalar_plan(
        atoms, registry, entity_rows, goal, horizon, tick, config
    )
    evaluated = evaluate_policy(
        atoms, primary, registry, entity_rows, goal, horizon, tick, config
    )
    counterfactual = plan_exact(
        atoms, suppressed, entity_rows, goal, horizon, tick, config
    )
    action_change = (
        primary["selected_action_key"]
        not in counterfactual["optimal_action_keys"]
    )
    value_change = abs(primary["value"] - counterfactual["value"])
    sensitive = action_change or value_change > 0.001
    integrity = policy_tree_checks(
        atoms, primary, registry, entity_rows, goal,
        horizon, tick, config,
    )
    values = [
        primary["value"], reference["value"], evaluated,
        counterfactual["value"],
    ]
    return {
        "id": public_record["id"],
        "record": public_record["record"],
        "history_class": public_record["history_class"],
        "goal": goal,
        "belief_atoms": len(atoms),
        "root_value": primary["value"],
        "reference_value": reference["value"],
        "root_value_error": abs(primary["value"] - reference["value"]),
        "root_optimal_set_member": (
            primary["selected_action_key"] in reference["optimal_action_keys"]
        ),
        "independent_policy_value": evaluated,
        "independent_policy_error": abs(primary["value"] - evaluated),
        "selected_action_key": primary["selected_action_key"],
        "delay_suppressed_value": counterfactual["value"],
        "delay_suppressed_optimal_action_keys": counterfactual["optimal_action_keys"],
        "root_action_changes_under_delay_suppression": action_change,
        "absolute_root_value_change_under_delay_suppression": value_change,
        "delayed_consequence_sensitive": sensitive,
        "integrity": integrity,
        "finite": all(math.isfinite(value) for value in values),
    }


def aggregate(records: list[dict], config: dict) -> dict:
    count = len(records)
    gates = config["gates"]
    normalization_checks = sum(
        row["integrity"]["normalization_checks"] for row in records
    )
    normalization_passes = sum(
        row["integrity"]["normalization_passes"] for row in records
    )
    delayed_by_history = {
        history: sum(
            row["delayed_consequence_sensitive"]
            for row in records if row["history_class"] == history
        )
        for history in config["population"]["historyClasses"]
    }
    sensitive_count = sum(
        row["delayed_consequence_sensitive"] for row in records
    )
    metrics = {
        "exact_correctness": {
            "completed_task_fraction": count / config["population"]["confirmationTasks"],
            "maximum_root_value_error_against_scalar_reference": max(
                (row["root_value_error"] for row in records), default=math.inf
            ),
            "root_optimal_set_membership_rate": mean([
                float(row["root_optimal_set_member"]) for row in records
            ]),
            "maximum_independent_policy_evaluation_error": max(
                (row["independent_policy_error"] for row in records), default=math.inf
            ),
            "belief_and_observation_normalization_rate": (
                normalization_passes / normalization_checks
                if normalization_checks else 0.0
            ),
            "finite_value_rate": mean([
                float(row["finite"]) for row in records
            ]),
        },
        "delayed_consequence": {
            "delayed_consequence_sensitive_policy_fraction": (
                sensitive_count / count if count else 0.0
            ),
            "delayed_consequence_sensitive_task_count": sensitive_count,
            "delayed_sensitive_task_count_by_history_class": delayed_by_history,
            "root_action_change_fraction_under_delay_suppression": mean([
                float(row["root_action_changes_under_delay_suppression"])
                for row in records
            ]),
            "root_value_change_fraction_over_0_001": mean([
                float(row["absolute_root_value_change_under_delay_suppression"] > 0.001)
                for row in records
            ]),
            "mean_absolute_root_value_change_under_delay_suppression": mean([
                row["absolute_root_value_change_under_delay_suppression"]
                for row in records
            ]),
            "maximum_absolute_root_value_change_under_delay_suppression": max(
                (
                    row["absolute_root_value_change_under_delay_suppression"]
                    for row in records
                ),
                default=0.0,
            ),
        },
        "integrity": {
            "truth_field_access_before_policy_evaluation_count": 0,
            "future_observation_access_count": 0,
            "candidate_action_omission_count": sum(
                row["integrity"]["candidate_omissions"] for row in records
            ),
            "canonical_tie_break_violation_count": sum(
                row["integrity"]["tie_break_violations"] for row in records
            ),
            "history_and_policy_evaluation_stream_collision_count": 0,
        },
    }
    exact, delayed, integrity = (
        metrics["exact_correctness"],
        metrics["delayed_consequence"],
        metrics["integrity"],
    )
    checks = {
        "completed_task_fraction": exact["completed_task_fraction"]
        >= gates["minimumCompletedTaskFraction"],
        "root_value_error": exact["maximum_root_value_error_against_scalar_reference"]
        <= gates["maximumRootValueError"],
        "root_optimal_set_membership": exact["root_optimal_set_membership_rate"]
        >= gates["minimumRootOptimalSetMembershipRate"],
        "independent_policy_evaluation": exact[
            "maximum_independent_policy_evaluation_error"
        ] <= gates["maximumIndependentPolicyEvaluationError"],
        "belief_and_observation_normalization": exact[
            "belief_and_observation_normalization_rate"
        ] >= gates["minimumBeliefAndObservationNormalizationRate"],
        "finite_value_rate": exact["finite_value_rate"]
        >= gates["minimumFiniteValueRate"],
        "delayed_consequence_sensitive_fraction": delayed[
            "delayed_consequence_sensitive_policy_fraction"
        ] >= gates["minimumDelayedConsequenceSensitivePolicyFraction"],
        "delayed_sensitive_each_history_class": all(
            value >= gates["minimumDelayedSensitiveTasksPerHistoryClass"]
            for value in delayed["delayed_sensitive_task_count_by_history_class"].values()
        ),
        "minimum_root_action_or_value_change_tasks": delayed[
            "delayed_consequence_sensitive_task_count"
        ] >= gates["minimumRootActionOrValueChangeTasks"],
        "truth_field_firewall": integrity[
            "truth_field_access_before_policy_evaluation_count"
        ] <= gates["maximumTruthFieldAccessBeforePolicyEvaluationCount"],
        "future_observation_firewall": integrity[
            "future_observation_access_count"
        ] <= gates["maximumFutureObservationAccessCount"],
        "complete_candidate_actions": integrity["candidate_action_omission_count"]
        <= gates["maximumCandidateActionOmissionCount"],
        "canonical_tie_break": integrity["canonical_tie_break_violation_count"]
        <= gates["maximumCanonicalTieBreakViolationCount"],
        "independent_streams": integrity[
            "history_and_policy_evaluation_stream_collision_count"
        ] <= gates["maximumHistoryAndPolicyEvaluationStreamCollisionCount"],
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-lock",
        default="configs/v55r1-evaluation-implementation-lock.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation",
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.evaluation_lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if attempt_path.exists() or output_dir.exists():
        raise RuntimeError("V55r1 permits exactly one sealed evaluation attempt")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_v55r1_evaluation"]:
        raise RuntimeError("V55r1 evaluation lock does not authorize the run")
    for section in ("evaluation_files_sha256", "frozen_dependencies_sha256"):
        for relative, digest in lock[section].items():
            if file_sha256(PROJECT_ROOT / relative) != digest:
                raise RuntimeError(f"V55r1 frozen evaluation input changed: {relative}")
    seal_path = PROJECT_ROOT / lock["population_seal"]
    if file_sha256(seal_path) != lock["population_seal_sha256"]:
        raise RuntimeError("V55r1 population seal changed")
    seal = json.loads(seal_path.read_text())
    population_path = PROJECT_ROOT / seal["population"]
    if file_sha256(population_path) != seal["population_sha256"]:
        raise RuntimeError("V55r1 sealed population changed")
    implementation = json.loads(
        (PROJECT_ROOT / seal["implementation_lock"]).read_text()
    )
    design = json.loads(
        (PROJECT_ROOT / implementation["design_lock"]).read_text()
    )
    config = design["config_payload"]
    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    rows = read_jsonl(population_path)
    public_rows = [
        {
            "id": row["id"],
            "record": row["record"],
            "history_class": row["history_class"],
            "public": row["public"],
        }
        for row in rows
    ]
    attempt = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_evaluation_attempt",
        "attempt": 1,
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "started_unix_seconds": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    registry = planning_registry(config)
    suppressed = delay_suppressed_registry(
        registry, config["planningModel"]["horizonActions"]
    )
    started = time.time()
    records = []
    for public_row in public_rows:
        records.append(evaluate_record(
            public_row, registry, v53, config, suppressed
        ))
        print(json.dumps({
            "completed": len(records),
            "total": len(public_rows),
            "id": public_row["id"],
            "seconds": time.time() - started,
        }), flush=True)
    aggregated = aggregate(records, config)
    result = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_delayed_consequence_adequacy_confirmation",
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "evaluation_run": 1,
        "records": records,
        "metrics": aggregated["metrics"],
        "qualification": {
            "checks": aggregated["checks"],
            "passed": aggregated["passed"],
        },
        "metric_definitions": {
            "delayed_consequence_sensitive": config["counterfactual"]["sensitiveTask"],
            "delay_counterfactual": config["counterfactual"]["delaySuppressed"],
        },
        "runtime_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "checks": aggregated["checks"],
        "metrics": aggregated["metrics"],
        "passed": aggregated["passed"],
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
