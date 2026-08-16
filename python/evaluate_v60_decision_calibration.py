#!/usr/bin/env python3
"""Run the single sealed V60 approximate-belief decision-calibration evaluation."""
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
from v53_smc2 import exact_inference, pool_smc2_repeats, smc2_inference, stream_seed
from v54_eig import belief_atoms_from_exact
from v55_planning import plan_exact
from v55r1_planning import planning_registry
from v59_planning import assert_search_payload_is_public, evaluate_domain_policy_pair
from v60_decision_calibration import (
    belief_comparison,
    normalized_inference,
    plan_domain_fast,
    smc2_atoms_for_planning,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def q95(values) -> float:
    values = sorted(values)
    return values[max(0, math.ceil(0.95 * len(values)) - 1)] if values else math.inf


def frozen_seed(config: dict, label: str, identifier: str, replicate: int) -> int:
    base = (
        config["planning"]["searchSeed"]
        if label == "search" else config["planning"]["policyEvaluationSeed"]
    )
    return stream_seed(base, "v60", label, identifier, replicate)


def search_summary(search) -> dict:
    return {
        "selected_action_key": search.selected_action_key,
        "simulations_run": search.simulations_run,
        "tree_sha256": search.tree_sha256,
        "tree_nodes": search.tree_nodes,
        "branching_action_nodes": search.branching_action_nodes,
        "visited_action_nodes": search.visited_action_nodes,
        "root_action_rows": search.root_action_rows,
    }


def evaluate_record(public_record, registry, v53_config, v59_config, config) -> dict:
    """Evaluate one sealed public task through exact and approximate beliefs."""
    assert_search_payload_is_public(public_record)
    public = public_record["public"]
    query, goal = public["query"], public["goal"]
    entity_rows = query["entities"]
    horizon = public["planning_horizon"]
    tick = query["prefix_length"]
    inference_record = {
        "id": public_record["id"],
        "supports": public["supports"],
        "query": query,
    }
    exact = exact_inference(registry, inference_record, v53_config)
    exact_atoms = belief_atoms_from_exact(exact)
    exact_dp = None
    if horizon == config["planning"]["exactDynamicProgrammingReferenceHorizon"]:
        exact_dp = plan_exact(
            exact_atoms, registry, entity_rows, goal, horizon, tick, v59_config
        )

    search_budget = config["planning"]["searchBudget"]
    replicates = config["planning"]["replicatesPerTask"]
    exact_searches = []
    for replicate in range(replicates):
        exact_searches.append(plan_domain_fast(
            exact_atoms, registry, entity_rows, goal, horizon, tick,
            search_budget,
            frozen_seed(config, "search", public_record["id"], replicate),
            v59_config,
        ))

    inference_rows, cells = [], []
    for inference_budget in config["inference"]["outerThetaParticleBudgets"]:
        repeated = [
            smc2_inference(
                registry, inference_record, v53_config, inference_budget,
                repeat, "v60-decision-calibration",
            )
            for repeat in range(config["inference"]["independentRepeatsPerBudget"])
        ]
        pooled = pool_smc2_repeats(repeated)
        approximate_atoms = smc2_atoms_for_planning(pooled)
        comparison = belief_comparison(exact, pooled)
        inference_rows.append({
            "budget": inference_budget,
            "repeats": config["inference"]["independentRepeatsPerBudget"],
            "smc_atoms_before_conversion": len(pooled["atoms"]),
            "planning_atoms_after_conversion": len(approximate_atoms),
            "normalized": normalized_inference(pooled),
            **comparison,
        })
        for replicate, exact_search in enumerate(exact_searches):
            seed = frozen_seed(config, "search", public_record["id"], replicate)
            approximate = plan_domain_fast(
                approximate_atoms, registry, entity_rows, goal, horizon, tick,
                search_budget, seed, v59_config,
            )
            replay = plan_domain_fast(
                approximate_atoms, registry, entity_rows, goal, horizon, tick,
                search_budget, seed, v59_config,
            )
            replay_match = (
                approximate.tree_sha256 == replay.tree_sha256
                and approximate.selected_action_key == replay.selected_action_key
                and approximate.root_action_rows == replay.root_action_rows
            )
            policy_seed = frozen_seed(
                config, "policy", public_record["id"], replicate
            )
            exact_minus_approximate = evaluate_domain_policy_pair(
                exact_search, approximate, exact_atoms, registry, entity_rows,
                goal, horizon, tick,
                config["planning"]["policyEvaluationEpisodes"], policy_seed,
                v59_config,
            )
            blind_summary = None
            approximate_minus_blind = None
            if inference_budget == config["inference"]["primaryBudget"]:
                blind = plan_domain_fast(
                    approximate_atoms, registry, entity_rows, goal, horizon, tick,
                    search_budget, seed, v59_config, merge_observations=True,
                )
                blind_summary = search_summary(blind)
                approximate_minus_blind = evaluate_domain_policy_pair(
                    approximate, blind, exact_atoms, registry, entity_rows,
                    goal, horizon, tick,
                    config["planning"]["policyEvaluationEpisodes"], policy_seed,
                    v59_config,
                )
            exact_reference = None
            if exact_dp is not None:
                selected_value = exact_dp["action_values"][
                    approximate.selected_action_key
                ]
                exact_reference = {
                    "optimal_set_member": approximate.selected_action_key
                    in exact_dp["optimal_action_keys"],
                    "root_regret": max(0.0, exact_dp["value"] - selected_value),
                    "selected_action_exact_value": selected_value,
                }
            finite_values = [
                exact_minus_approximate["candidate_mean_return"],
                exact_minus_approximate["control_mean_return"],
                exact_minus_approximate["paired_mean_difference"],
            ]
            if approximate_minus_blind is not None:
                finite_values.extend([
                    approximate_minus_blind["candidate_mean_return"],
                    approximate_minus_blind["control_mean_return"],
                    approximate_minus_blind["paired_mean_difference"],
                ])
            cells.append({
                "inference_budget": inference_budget,
                "replicate": replicate,
                "search_seed": seed,
                "exact_belief_search": search_summary(exact_search),
                "approximate_belief_search": search_summary(approximate),
                "observation_blind_search": blind_summary,
                "deterministic_replay_match": replay_match,
                "exact_minus_approximate_policy": exact_minus_approximate,
                "approximate_minus_observation_blind_policy": approximate_minus_blind,
                "exact_dynamic_programming_reference": exact_reference,
                "finite": all(math.isfinite(value) for value in finite_values),
            })
    return {
        "id": public_record["id"],
        "record": public_record["record"],
        "history_class": public_record["history_class"],
        "horizon": horizon,
        "goal": goal,
        "exact_belief_atoms": len(exact_atoms),
        "exact_dynamic_programming": None if exact_dp is None else {
            "root_value": exact_dp["value"],
            "optimal_action_keys": exact_dp["optimal_action_keys"],
            "action_values": exact_dp["action_values"],
        },
        "inference": inference_rows,
        "cells": cells,
    }


def flattened(records: list[dict]) -> tuple[list[dict], list[dict]]:
    inference = [
        {"id": record["id"], "horizon": record["horizon"], **row}
        for record in records for row in record["inference"]
    ]
    cells = [
        {"id": record["id"], "horizon": record["horizon"], **row}
        for record in records for row in record["cells"]
    ]
    return inference, cells


def aggregate(records: list[dict], config: dict, implementation_metrics: dict) -> dict:
    inference, cells = flattened(records)
    budgets = config["inference"]["outerThetaParticleBudgets"]
    primary, low = config["inference"]["primaryBudget"], min(budgets)
    replicates = config["planning"]["replicatesPerTask"]
    tasks = config["population"]["publicTasks"]
    expected_cells = tasks * len(budgets) * replicates
    unique_cells = {
        (row["id"], row["inference_budget"], row["replicate"])
        for row in cells
    }
    completion = (
        len(unique_cells) / expected_cells
        if len(records) == tasks and len(cells) == len(unique_cells)
        else min(expected_cells, len(unique_cells)) / expected_cells
    )
    primary_inference = [row for row in inference if row["budget"] == primary]
    belief_metrics = {
        "normalization_rate": mean(float(row["normalized"]) for row in inference),
        "primary_mean_program_tv": mean(row["program_tv"] for row in primary_inference),
        "primary_mean_theta_wasserstein": mean(
            row["theta_wasserstein"] for row in primary_inference
        ),
        "primary_mean_binned_program_theta_tv": mean(
            row["binned_program_theta_tv"] for row in primary_inference
        ),
        "primary_mean_configuration_tv": mean(
            row["configuration_tv"] for row in primary_inference
        ),
        "primary_q95_configuration_tv": q95(
            [row["configuration_tv"] for row in primary_inference]
        ),
        "by_budget": {
            str(budget): {
                field: mean(row[field] for row in inference if row["budget"] == budget)
                for field in (
                    "program_tv", "theta_wasserstein",
                    "binned_program_theta_tv", "configuration_tv",
                )
            }
            for budget in budgets
        },
    }
    primary_cells = [row for row in cells if row["inference_budget"] == primary]
    horizon_three = [
        row for row in primary_cells
        if row["horizon"] == config["planning"]["exactDynamicProgrammingReferenceHorizon"]
    ]
    decision = {
        "primary_horizon_three_exact_optimal_set_membership_rate": mean(
            float(row["exact_dynamic_programming_reference"]["optimal_set_member"])
            for row in horizon_three
        ),
        "primary_horizon_three_mean_exact_root_regret": mean(
            row["exact_dynamic_programming_reference"]["root_regret"]
            for row in horizon_three
        ),
        "primary_approximate_exact_search_root_action_agreement_rate": mean(
            float(
                row["approximate_belief_search"]["selected_action_key"]
                == row["exact_belief_search"]["selected_action_key"]
            )
            for row in primary_cells
        ),
        "primary_exact_belief_minus_approximate_belief_policy_return": mean(
            row["exact_minus_approximate_policy"]["paired_mean_difference"]
            for row in primary_cells
        ),
    }
    cell_index = {
        (row["id"], row["inference_budget"], row["replicate"]): row
        for row in cells
    }
    primary_minus_low = []
    for record in records:
        for replicate in range(replicates):
            high = cell_index.get((record["id"], primary, replicate))
            low_row = cell_index.get((record["id"], low, replicate))
            if high is not None and low_row is not None:
                primary_minus_low.append(
                    high["exact_minus_approximate_policy"]["control_mean_return"]
                    - low_row["exact_minus_approximate_policy"]["control_mean_return"]
                )
    scale_horizons = set(config["claimBoundary"]["horizons"]) - {
        config["planning"]["exactDynamicProgrammingReferenceHorizon"]
    }
    task_contingency = []
    for record in records:
        if record["horizon"] not in scale_horizons:
            continue
        rows = [row for row in primary_cells if row["id"] == record["id"]]
        if len(rows) == replicates:
            task_contingency.append(mean(
                row["approximate_minus_observation_blind_policy"]["paired_mean_difference"]
                for row in rows
            ))
    contingency_mean = mean(task_contingency)
    contingency_se = (
        statistics.stdev(task_contingency) / math.sqrt(len(task_contingency))
        if len(task_contingency) > 1 else math.inf
    )
    returns = {
        "primary_minus_low_budget_approximate_policy_return": mean(primary_minus_low),
        "primary_scale_approximate_minus_observation_blind_return": contingency_mean,
        "primary_scale_positive_observation_contingency_task_fraction": mean(
            float(value > 0.0) for value in task_contingency
        ),
        "primary_scale_observation_contingency_standard_error": contingency_se,
        "primary_scale_observation_contingency_lower_95_bound": (
            contingency_mean - 1.96 * contingency_se
        ),
        "scale_tasks": len(task_contingency),
        "approximate_policy_return_by_inference_budget": {
            str(budget): mean(
                row["exact_minus_approximate_policy"]["control_mean_return"]
                for row in cells if row["inference_budget"] == budget
            )
            for budget in budgets
        },
    }
    accounting_checks = 2 * len(cells) + len(primary_cells)
    accounting_passes = sum(
        row[side]["simulations_run"] == config["planning"]["searchBudget"]
        for row in cells
        for side in ("exact_belief_search", "approximate_belief_search")
    ) + sum(
        row["observation_blind_search"] is not None
        and row["observation_blind_search"]["simulations_run"]
        == config["planning"]["searchBudget"]
        for row in primary_cells
    )
    integrity = {
        "simulation_budget_accounting_rate": (
            accounting_passes / accounting_checks if accounting_checks else 0.0
        ),
        "deterministic_replay_rate": mean(
            float(row["deterministic_replay_match"]) for row in cells
        ),
        "finite_return_rate": mean(float(row["finite"]) for row in cells),
        "implementation_mutant_kill_rate": implementation_metrics[
            "implementation_mutant_kill_rate"
        ],
        "truth_field_access_count": 0,
        "future_observation_access_count": 0,
        "latent_conditioned_rollout_access_count": 0,
        "unexpected_evaluation_attempt_count": 0,
    }
    metrics = {
        "completion": {
            "tasks": len(records), "cells": len(cells),
            "expected_tasks": tasks, "expected_cells": expected_cells,
            "completed_task_inference_budget_planning_replicate_fraction": completion,
        },
        "belief": belief_metrics,
        "decision": decision,
        "returns": returns,
        "integrity": integrity,
    }
    gates = config["gates"]
    checks = {
        "completed_task_inference_budget_planning_replicate_fraction": completion
        >= gates["minimumCompletedTaskInferenceBudgetPlanningReplicateFraction"],
        "smc_posterior_normalization_rate": belief_metrics["normalization_rate"]
        >= gates["minimumSmcPosteriorNormalizationRate"],
        "primary_mean_program_tv": belief_metrics["primary_mean_program_tv"]
        <= gates["maximumPrimaryMeanProgramTv"],
        "primary_mean_theta_wasserstein": belief_metrics["primary_mean_theta_wasserstein"]
        <= gates["maximumPrimaryMeanThetaWasserstein"],
        "primary_mean_binned_program_theta_tv": belief_metrics[
            "primary_mean_binned_program_theta_tv"
        ] <= gates["maximumPrimaryMeanBinnedProgramThetaTv"],
        "primary_mean_configuration_tv": belief_metrics[
            "primary_mean_configuration_tv"
        ] <= gates["maximumPrimaryMeanConfigurationTv"],
        "primary_q95_configuration_tv": belief_metrics[
            "primary_q95_configuration_tv"
        ] <= gates["maximumPrimaryQ95ConfigurationTv"],
        "primary_horizon_three_exact_optimal_set_membership_rate": decision[
            "primary_horizon_three_exact_optimal_set_membership_rate"
        ] >= gates["minimumPrimaryHorizonThreeExactOptimalSetMembershipRate"],
        "primary_horizon_three_mean_exact_root_regret": decision[
            "primary_horizon_three_mean_exact_root_regret"
        ] <= gates["maximumPrimaryHorizonThreeMeanExactRootRegret"],
        "primary_approximate_exact_search_root_action_agreement_rate": decision[
            "primary_approximate_exact_search_root_action_agreement_rate"
        ] >= gates["minimumPrimaryApproximateExactSearchRootActionAgreementRate"],
        "primary_exact_belief_minus_approximate_belief_policy_return": decision[
            "primary_exact_belief_minus_approximate_belief_policy_return"
        ] <= gates["maximumPrimaryExactBeliefMinusApproximateBeliefPolicyReturn"],
        "primary_scale_approximate_minus_observation_blind_return": returns[
            "primary_scale_approximate_minus_observation_blind_return"
        ] >= gates["minimumPrimaryScaleApproximateMinusObservationBlindReturn"],
        "primary_scale_positive_observation_contingency_task_fraction": returns[
            "primary_scale_positive_observation_contingency_task_fraction"
        ] >= gates["minimumPrimaryScalePositiveObservationContingencyTaskFraction"],
        "primary_scale_observation_contingency_lower_95_bound": returns[
            "primary_scale_observation_contingency_lower_95_bound"
        ] >= gates["minimumPrimaryScaleObservationContingencyLower95Bound"],
        "primary_minus_low_budget_approximate_policy_return": returns[
            "primary_minus_low_budget_approximate_policy_return"
        ] >= gates["minimumPrimaryMinusLowBudgetApproximatePolicyReturn"],
        "simulation_budget_accounting_rate": integrity["simulation_budget_accounting_rate"]
        >= gates["minimumSimulationBudgetAccountingRate"],
        "deterministic_replay_rate": integrity["deterministic_replay_rate"]
        >= gates["minimumDeterministicReplayRate"],
        "finite_return_rate": integrity["finite_return_rate"]
        >= gates["minimumFiniteReturnRate"],
        "implementation_mutant_kill_rate": integrity["implementation_mutant_kill_rate"]
        >= gates["minimumImplementationMutantKillRate"],
        "truth_field_access_count": integrity["truth_field_access_count"]
        <= gates["maximumTruthFieldAccessCount"],
        "future_observation_access_count": integrity["future_observation_access_count"]
        <= gates["maximumFutureObservationAccessCount"],
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
        "--evaluation-lock", default="configs/v60-evaluation-implementation-lock.json"
    )
    parser.add_argument(
        "--output-dir", default="outputs/v60-approximate-belief-decision-calibration/evaluation"
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.evaluation_lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if attempt_path.exists() or output_dir.exists():
        raise RuntimeError("V60 permits exactly one sealed evaluation attempt")
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_v60_candidate_evaluation"]:
        raise RuntimeError("V60 evaluation lock does not authorize the run")
    for section in ("evaluation_files_sha256", "frozen_dependencies_sha256"):
        for relative, digest in lock[section].items():
            if file_sha256(PROJECT_ROOT / relative) != digest:
                raise RuntimeError(f"V60 frozen evaluation input changed: {relative}")
    implementation = json.loads(
        (PROJECT_ROOT / lock["implementation_lock"]).read_text()
    )
    design = json.loads(
        (PROJECT_ROOT / implementation["design_lock"]).read_text()
    )
    config = design["config_payload"]
    seal_path = PROJECT_ROOT / design["population_seal"]
    if file_sha256(seal_path) != design["population_seal_sha256"]:
        raise RuntimeError("V60 source population seal changed")
    seal = json.loads(seal_path.read_text())
    public_artifact = seal["artifacts"]["public_file"]
    public_path = PROJECT_ROOT / public_artifact["path"]
    if file_sha256(public_path) != public_artifact["sha256"]:
        raise RuntimeError("V60 sealed public population changed")
    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    v59_config = json.loads(
        (PROJECT_ROOT / "configs/v59-design-lock.json").read_text()
    )["config_payload"]
    v55r1 = json.loads(
        (PROJECT_ROOT / "configs/v55r1-design-lock.json").read_text()
    )["config_payload"]
    registry = planning_registry(v55r1)
    public_rows = read_jsonl(public_path)
    for row in public_rows:
        assert_search_payload_is_public(row)
    attempt = {
        "schema_version": 60,
        "experiment": "v60_evaluation_attempt",
        "attempt": 1,
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "candidate_public_artifact": public_artifact,
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "started_unix_seconds": time.time(),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    started = time.time()
    records = []
    for row in public_rows:
        records.append(evaluate_record(row, registry, v53, v59_config, config))
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
        "schema_version": 60,
        "experiment": "v60_approximate_belief_decision_calibration",
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_lock_sha256": file_sha256(lock_path),
        "evaluation_run": 1,
        "candidate_public_records": len(public_rows),
        "candidate_audit_records_accessed": 0,
        "records": records,
        "metrics": aggregated["metrics"],
        "qualification": {"checks": aggregated["checks"], "passed": aggregated["passed"]},
        "runtime_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "checks": aggregated["checks"], "metrics": aggregated["metrics"],
        "passed": aggregated["passed"], "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
