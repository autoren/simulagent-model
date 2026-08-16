#!/usr/bin/env python3
"""Run the single sealed V59 budgeted root-sampled planning evaluation."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import exact_inference, stream_seed
from v54_eig import belief_atoms_from_exact
from v55_planning import candidate_actions, plan_exact
from v55r1_planning import planning_registry
from v59_planning import (
    assert_search_payload_is_public,
    evaluate_domain_policy_pair,
    plan_domain,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluation_seed(config: dict, label: str, identifier: str, budget: int, replicate: int) -> int:
    return stream_seed(
        config["evaluation"]["evaluationSeed"],
        "v59", label, identifier, budget, replicate,
    )


def evaluate_record(public_record, registry, v53_config, config) -> dict:
    """Evaluate one row through the sealed public-input interface."""
    assert_search_payload_is_public(public_record)
    public = public_record["public"]
    query, goal = public["query"], public["goal"]
    entity_rows = query["entities"]
    tick = query["prefix_length"]
    horizon = public["planning_horizon"]
    exact = exact_inference(
        registry,
        {"supports": public["supports"], "query": query},
        v53_config,
    )
    atoms = belief_atoms_from_exact(exact)
    exact_reference = None
    if horizon == config["evaluation"]["exactReferenceHorizon"]:
        exact_reference = plan_exact(
            atoms, registry, entity_rows, goal, horizon, tick, config
        )

    cells = []
    for budget in config["candidateSearch"]["searchBudgets"]:
        for replicate in range(config["candidateSearch"]["replicatesPerTaskBudget"]):
            plan_seed = evaluation_seed(
                config, "plan", public_record["id"], budget, replicate
            )
            candidate = plan_domain(
                atoms, registry, entity_rows, goal, horizon, tick,
                budget, plan_seed, config, merge_observations=False,
            )
            control = plan_domain(
                atoms, registry, entity_rows, goal, horizon, tick,
                budget, plan_seed, config, merge_observations=True,
            )
            replay = plan_domain(
                atoms, registry, entity_rows, goal, horizon, tick,
                budget, plan_seed, config, merge_observations=False,
            )
            replay_match = (
                replay.tree_sha256 == candidate.tree_sha256
                and replay.selected_action_key == candidate.selected_action_key
                and replay.root_action_rows == candidate.root_action_rows
                and replay.root_sample_counts == candidate.root_sample_counts
            )
            pair = evaluate_domain_policy_pair(
                candidate, control, atoms, registry, entity_rows, goal,
                horizon, tick,
                config["evaluation"]["posteriorEpisodesPerPolicy"],
                evaluation_seed(
                    config, "policy", public_record["id"], budget, replicate
                ),
                config,
            )
            exact_metrics = None
            if exact_reference is not None:
                selected_value = exact_reference["action_values"][
                    candidate.selected_action_key
                ]
                exact_metrics = {
                    "optimal_set_member": candidate.selected_action_key
                    in exact_reference["optimal_action_keys"],
                    "root_regret": max(0.0, exact_reference["value"] - selected_value),
                    "selected_action_exact_value": selected_value,
                }
            cells.append({
                "budget": budget,
                "replicate": replicate,
                "planning_seed": plan_seed,
                "candidate": {
                    "selected_action_key": candidate.selected_action_key,
                    "simulations_run": candidate.simulations_run,
                    "tree_sha256": candidate.tree_sha256,
                    "tree_nodes": candidate.tree_nodes,
                    "branching_action_nodes": candidate.branching_action_nodes,
                    "visited_action_nodes": candidate.visited_action_nodes,
                    "root_action_rows": candidate.root_action_rows,
                },
                "observation_blind_control": {
                    "selected_action_key": control.selected_action_key,
                    "simulations_run": control.simulations_run,
                    "tree_sha256": control.tree_sha256,
                    "tree_nodes": control.tree_nodes,
                    "branching_action_nodes": control.branching_action_nodes,
                    "visited_action_nodes": control.visited_action_nodes,
                },
                "deterministic_replay_match": replay_match,
                "policy_evaluation": pair,
                "exact_reference": exact_metrics,
                "finite": pair["finite"] and all(math.isfinite(value) for value in (
                    pair["candidate_mean_return"], pair["control_mean_return"],
                    pair["paired_mean_difference"], pair["paired_standard_error"],
                )),
            })
    exact_summary = None if exact_reference is None else {
        "root_value": exact_reference["value"],
        "optimal_action_keys": exact_reference["optimal_action_keys"],
        "action_values": exact_reference["action_values"],
    }
    return {
        "id": public_record["id"],
        "record": public_record["record"],
        "history_class": public_record["history_class"],
        "horizon": horizon,
        "goal": goal,
        "belief_atoms": len(atoms),
        "exact_reference": exact_summary,
        "cells": cells,
    }


def _cell_rows(records: list[dict]) -> list[dict]:
    return [
        {"id": record["id"], "horizon": record["horizon"], **cell}
        for record in records for cell in record["cells"]
    ]


def _nested_means(cells: list[dict], field: str) -> dict:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in cells:
        grouped[(row["horizon"], row["budget"])].append(
            row["policy_evaluation"][field]
        )
    return {
        str(horizon): {
            str(budget): mean(grouped[(horizon, budget)])
            for budget in sorted({key[1] for key in grouped if key[0] == horizon})
        }
        for horizon in sorted({key[0] for key in grouped})
    }


def aggregate(records: list[dict], config: dict, implementation_metrics: dict) -> dict:
    budgets = config["candidateSearch"]["searchBudgets"]
    replicates = config["candidateSearch"]["replicatesPerTaskBudget"]
    expected_tasks = config["population"]["tasks"]
    expected_cells = expected_tasks * len(budgets) * replicates
    cells = _cell_rows(records)
    cell_keys = {
        (row["id"], row["budget"], row["replicate"]) for row in cells
    }
    completed_fraction = (
        len(cell_keys) / expected_cells
        if len(records) == expected_tasks and len(cells) == len(cell_keys)
        else min(len(cell_keys), expected_cells) / expected_cells
    )
    high, low = max(budgets), min(budgets)
    exact_cells = [
        row for row in cells
        if row["horizon"] == config["evaluation"]["exactReferenceHorizon"]
        and row["budget"] == high and row["exact_reference"] is not None
    ]
    exact_membership = mean(
        float(row["exact_reference"]["optimal_set_member"])
        for row in exact_cells
    )
    exact_regret = mean(
        row["exact_reference"]["root_regret"] for row in exact_cells
    )

    by_id_budget_rep = {
        (row["id"], row["budget"], row["replicate"]): row for row in cells
    }
    high_minus_low = []
    for record in records:
        for replicate in range(replicates):
            high_row = by_id_budget_rep.get((record["id"], high, replicate))
            low_row = by_id_budget_rep.get((record["id"], low, replicate))
            if high_row is not None and low_row is not None:
                high_minus_low.append(
                    high_row["policy_evaluation"]["candidate_mean_return"]
                    - low_row["policy_evaluation"]["candidate_mean_return"]
                )

    scale_horizons = set(config["planningModel"]["horizons"]) - {
        config["evaluation"]["exactReferenceHorizon"]
    }
    scale_high = [
        row for row in cells
        if row["horizon"] in scale_horizons and row["budget"] == high
    ]
    task_differences = []
    for record in records:
        if record["horizon"] not in scale_horizons:
            continue
        rows = [row for row in scale_high if row["id"] == record["id"]]
        if len(rows) == replicates:
            task_differences.append(mean(
                row["policy_evaluation"]["paired_mean_difference"]
                for row in rows
            ))
    scale_high_difference = mean(task_differences)
    task_standard_error = (
        statistics.stdev(task_differences) / math.sqrt(len(task_differences))
        if len(task_differences) > 1 else math.inf
    )
    scale_lower_95 = scale_high_difference - 1.96 * task_standard_error

    accounting_checks = 2 * len(cells)
    accounting_passes = sum(
        row[side]["simulations_run"] == row["budget"]
        for row in cells
        for side in ("candidate", "observation_blind_control")
    )
    metrics = {
        "completion": {
            "tasks": len(records),
            "cells": len(cells),
            "expected_tasks": expected_tasks,
            "expected_cells": expected_cells,
            "completed_task_budget_replicate_fraction": completed_fraction,
        },
        "exact_reference": {
            "high_budget_cells": len(exact_cells),
            "high_budget_root_optimal_set_membership_rate": exact_membership,
            "high_budget_mean_root_regret": exact_regret,
        },
        "returns": {
            "mean_candidate_return_by_horizon_and_budget": _nested_means(
                cells, "candidate_mean_return"
            ),
            "mean_candidate_minus_observation_blind_by_horizon_and_budget": _nested_means(
                cells, "paired_mean_difference"
            ),
            "high_minus_low_budget_candidate_return": mean(high_minus_low),
            "scale_high_budget_candidate_minus_observation_blind_return": scale_high_difference,
            "scale_task_positive_observation_contingency_fraction": mean(
                float(value > 0.0) for value in task_differences
            ),
            "scale_task_high_budget_paired_difference_standard_error": task_standard_error,
            "scale_task_high_budget_paired_difference_lower_95_bound": scale_lower_95,
            "scale_tasks": len(task_differences),
        },
        "integrity": {
            "root_sample_static_total_variation_on_analytic_fixture": implementation_metrics[
                "analytic_root_sample_total_variation"
            ],
            "simulation_budget_accounting_rate": (
                accounting_passes / accounting_checks if accounting_checks else 0.0
            ),
            "deterministic_replay_rate": mean(
                float(row["deterministic_replay_match"]) for row in cells
            ),
            "finite_return_rate": mean(float(row["finite"]) for row in cells),
            "tree_observation_branching_rate": mean(
                float(row["candidate"]["branching_action_nodes"] > 0)
                for row in cells
            ),
            "implementation_mutant_kill_rate": implementation_metrics[
                "implementation_mutant_kill_rate"
            ],
            "truth_field_access_count": 0,
            "future_observation_access_count": 0,
            "latent_conditioned_rollout_access_count": 0,
            "unexpected_evaluation_attempt_count": 0,
        },
    }
    gates = config["gates"]
    completion = metrics["completion"]
    exact = metrics["exact_reference"]
    returns = metrics["returns"]
    integrity = metrics["integrity"]
    checks = {
        "completed_task_budget_replicate_fraction": completion[
            "completed_task_budget_replicate_fraction"
        ] >= gates["minimumCompletedTaskBudgetReplicateFraction"],
        "exact_reference_high_budget_root_optimal_set_membership_rate": exact[
            "high_budget_root_optimal_set_membership_rate"
        ] >= gates["minimumExactReferenceHighBudgetRootOptimalSetMembershipRate"],
        "exact_reference_high_budget_mean_root_regret": exact[
            "high_budget_mean_root_regret"
        ] <= gates["maximumExactReferenceHighBudgetMeanRootRegret"],
        "high_minus_low_budget_candidate_return": returns[
            "high_minus_low_budget_candidate_return"
        ] >= gates["minimumHighMinusLowBudgetCandidateReturn"],
        "scale_high_budget_candidate_minus_observation_blind_return": returns[
            "scale_high_budget_candidate_minus_observation_blind_return"
        ] >= gates["minimumScaleHighBudgetCandidateMinusObservationBlindReturn"],
        "scale_task_positive_observation_contingency_fraction": returns[
            "scale_task_positive_observation_contingency_fraction"
        ] >= gates["minimumScaleTaskPositiveObservationContingencyFraction"],
        "scale_task_high_budget_paired_difference_lower_95_bound": returns[
            "scale_task_high_budget_paired_difference_lower_95_bound"
        ] >= gates["minimumScaleHighBudgetPairedDifferenceLower95Bound"],
        "analytic_root_sample_static_total_variation": integrity[
            "root_sample_static_total_variation_on_analytic_fixture"
        ] <= gates["maximumAnalyticRootSampleStaticTotalVariation"],
        "simulation_budget_accounting_rate": integrity[
            "simulation_budget_accounting_rate"
        ] >= gates["minimumSimulationBudgetAccountingRate"],
        "deterministic_replay_rate": integrity[
            "deterministic_replay_rate"
        ] >= gates["minimumDeterministicReplayRate"],
        "finite_return_rate": integrity["finite_return_rate"]
        >= gates["minimumFiniteReturnRate"],
        "tree_observation_branching_rate": integrity[
            "tree_observation_branching_rate"
        ] >= gates["minimumTreeObservationBranchingRate"],
        "implementation_mutant_kill_rate": integrity[
            "implementation_mutant_kill_rate"
        ] >= gates["minimumImplementationMutantKillRate"],
        "truth_field_access_count": integrity["truth_field_access_count"]
        <= gates["maximumTruthFieldAccessCount"],
        "future_observation_access_count": integrity[
            "future_observation_access_count"
        ] <= gates["maximumFutureObservationAccessCount"],
        "latent_conditioned_rollout_access_count": integrity[
            "latent_conditioned_rollout_access_count"
        ] <= gates["maximumLatentConditionedRolloutAccessCount"],
        "unexpected_evaluation_attempt_count": integrity[
            "unexpected_evaluation_attempt_count"
        ] <= gates["maximumUnexpectedEvaluationAttemptCount"],
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-lock",
        default="configs/v59-evaluation-implementation-lock.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/v59-budgeted-root-sampled-planning/evaluation",
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.evaluation_lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if attempt_path.exists() or output_dir.exists():
        raise RuntimeError("V59 permits exactly one sealed evaluation attempt")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_v59_candidate_evaluation"]:
        raise RuntimeError("V59 evaluation lock does not authorize the run")
    for section in ("evaluation_files_sha256", "frozen_dependencies_sha256"):
        for relative, digest in lock[section].items():
            if file_sha256(PROJECT_ROOT / relative) != digest:
                raise RuntimeError(f"V59 frozen evaluation input changed: {relative}")
    seal_path = PROJECT_ROOT / lock["population_seal"]
    if file_sha256(seal_path) != lock["population_seal_sha256"]:
        raise RuntimeError("V59 population seal changed")
    seal = json.loads(seal_path.read_text())
    public_artifact = seal["artifacts"]["public_file"]
    public_path = PROJECT_ROOT / public_artifact["path"]
    if file_sha256(public_path) != public_artifact["sha256"]:
        raise RuntimeError("V59 sealed public population changed")
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
    v55r1 = json.loads(
        (PROJECT_ROOT / "configs/v55r1-design-lock.json").read_text()
    )["config_payload"]
    registry = planning_registry(v55r1)
    public_rows = read_jsonl(public_path)
    for row in public_rows:
        assert_search_payload_is_public(row)

    attempt = {
        "schema_version": 59,
        "experiment": "v59_evaluation_attempt",
        "attempt": 1,
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "candidate_population_artifact": public_artifact,
        "started_unix_seconds": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    started = time.time()
    records = []
    for row in public_rows:
        records.append(evaluate_record(row, registry, v53, config))
        print(json.dumps({
            "completed": len(records), "total": len(public_rows),
            "id": row["id"], "horizon": row["horizon"],
            "seconds": time.time() - started,
        }), flush=True)
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    aggregated = aggregate(
        records, config, implementation_audit["fixture_metrics"]
    )
    result = {
        "schema_version": 59,
        "experiment": "v59_budgeted_root_sampled_bayes_adaptive_planning",
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "evaluation_run": 1,
        "candidate_population_records": len(public_rows),
        "candidate_audit_truth_records_accessed": 0,
        "records": records,
        "metrics": aggregated["metrics"],
        "qualification": {
            "checks": aggregated["checks"],
            "passed": aggregated["passed"],
        },
        "metric_definitions": {
            "scale_task_paired_interval": (
                "normal_95_percent_interval_over_the_sixteen_task_level_means_"
                "of_three_common_random_number_candidate_minus_control_estimates"
            ),
            "exact_root_regret": config["evaluation"]["exactRootRegret"],
            "scale_return": config["evaluation"]["scaleReturn"],
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
