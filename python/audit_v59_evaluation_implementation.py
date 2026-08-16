#!/usr/bin/env python3
"""Pre-run audit of the V59 one-shot evaluator using altered-seed data."""
from __future__ import annotations

import argparse
import copy
import inspect
import json

from evaluate_v59_planning import aggregate, evaluate_record, main as evaluation_main
from generate_v59_planning import build_record, prior_observation_design_keys
from v10_protocol import file_sha256
from v22_relational import unary_atom
from v22r2_grounding import PROJECT_ROOT
from v55r1_planning import planning_registry


EVALUATION_FILES = (
    "python/evaluate_v59_planning.py",
    "python/audit_and_summarize_v59.py",
    "python/freeze_v59_outcome.py",
    "python/audit_v59_evaluation_implementation.py",
)

FROZEN_DEPENDENCIES = (
    "python/v53_smc2.py",
    "python/v54_eig.py",
    "python/v55_planning.py",
    "python/v55r1_planning.py",
    "python/v59_planning.py",
    "python/generate_v59_planning.py",
    "configs/v53r2-design-lock.json",
    "configs/v55r1-design-lock.json",
    "configs/v59-design-lock.json",
    "configs/v59-implementation-lock.json",
    "configs/v59-population-seal.json",
)


def passing_records(config: dict) -> list[dict]:
    budgets = config["candidateSearch"]["searchBudgets"]
    records = []
    for index in range(config["population"]["tasks"]):
        horizon = config["planningModel"]["horizons"][index % 3]
        cells = []
        for budget in budgets:
            for replicate in range(config["candidateSearch"]["replicatesPerTaskBudget"]):
                candidate_return = 0.5 + 0.01 * (budget == max(budgets))
                exact = None
                if horizon == config["evaluation"]["exactReferenceHorizon"]:
                    exact = {
                        "optimal_set_member": True,
                        "root_regret": 0.0,
                        "selected_action_exact_value": 0.5,
                    }
                cells.append({
                    "budget": budget,
                    "replicate": replicate,
                    "candidate": {
                        "simulations_run": budget,
                        "branching_action_nodes": 1,
                    },
                    "observation_blind_control": {"simulations_run": budget},
                    "deterministic_replay_match": True,
                    "policy_evaluation": {
                        "candidate_mean_return": candidate_return,
                        "control_mean_return": candidate_return - 0.02,
                        "paired_mean_difference": 0.02,
                    },
                    "exact_reference": exact,
                    "finite": True,
                })
        records.append({
            "id": f"stub_{index:03d}", "horizon": horizon, "cells": cells,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population-seal", default="configs/v59-population-seal.json"
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/v59-budgeted-root-sampled-planning/"
            "evaluation-implementation-audit.json"
        ),
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.population_seal).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    v55r1 = json.loads(
        (PROJECT_ROOT / "configs/v55r1-design-lock.json").read_text()
    )["config_payload"]
    implementation_audit = json.loads(
        (PROJECT_ROOT / implementation["implementation_audit"]).read_text()
    )
    errors: list[str] = []

    manifest_path = PROJECT_ROOT / seal["manifest"]
    manifest = json.loads(manifest_path.read_text())
    seal_bound = (
        seal["authorization"]["write_and_audit_v59_candidate_runner"]
        and not seal["authorization"]["run_v59_candidate_evaluation"]
        and not seal["authorization"]["modify_v59_population"]
        and not seal["authorization"]["candidate_access_v59_audit_truth"]
        and file_sha256(manifest_path) == seal["manifest_sha256"]
        and file_sha256(PROJECT_ROOT / seal["population_audit"])
        == seal["population_audit_sha256"]
        and file_sha256(implementation_path) == seal["implementation_lock_sha256"]
        and all(
            file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"]
            for row in seal["artifacts"].values()
        )
        and manifest["count"] == config["population"]["tasks"]
    )
    if not seal_bound:
        errors.append("V59 population seal is not intact and pre-evaluation")

    evaluator_source = inspect.getsource(evaluate_record)
    main_source = inspect.getsource(evaluation_main)
    firewall_ok = (
        set(inspect.signature(evaluate_record).parameters)
        == {"public_record", "registry", "v53_config", "config"}
        and "truth" not in evaluator_source
        and "future_observation" not in evaluator_source
        and 'seal["artifacts"]["public_file"]' in main_source
        and 'seal["artifacts"]["audit_truth_file"]' not in main_source
        and "assert_search_payload_is_public(public_record)" in evaluator_source
    )
    if not firewall_ok:
        errors.append("V59 evaluator crosses the public-only candidate firewall")

    fixture_config = copy.deepcopy(config)
    for key, value in tuple(fixture_config["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            fixture_config["population"][key] = value + 8_000_000
    fixture_config["candidateSearch"]["searchBudgets"] = [16]
    fixture_config["candidateSearch"]["replicatesPerTaskBudget"] = 1
    fixture_config["evaluation"]["posteriorEpisodesPerPolicy"] = 64
    fixture_registry = planning_registry(v55r1)
    used, prior = set(), prior_observation_design_keys()
    public_fixture, _ = build_record(
        0, 0,
        {"atom": unary_atom("active", "unit_0"), "value": True},
        3, fixture_registry, fixture_config, used, prior,
    )
    exact_config = copy.deepcopy(v53)
    exact_config["exactBenchmark"]["quadratureNodes"] = 5
    fixture_result = evaluate_record(
        public_fixture, fixture_registry, exact_config, fixture_config
    )
    fixture_cell = fixture_result["cells"][0]
    evaluator_fixture_ok = (
        len(fixture_result["cells"]) == 1
        and fixture_cell["candidate"]["simulations_run"] == 16
        and fixture_cell["observation_blind_control"]["simulations_run"] == 16
        and fixture_cell["deterministic_replay_match"]
        and fixture_cell["finite"]
        and fixture_cell["exact_reference"] is not None
    )
    if not evaluator_fixture_ok:
        errors.append("V59 altered-seed evaluator fixture failed")

    stubs = passing_records(config)
    aggregated = aggregate(stubs, config, implementation_audit["fixture_metrics"])
    missing = aggregate(stubs[:-1], config, implementation_audit["fixture_metrics"])
    negative = copy.deepcopy(stubs)
    for record in negative:
        if record["horizon"] in (5, 7):
            for cell in record["cells"]:
                if cell["budget"] == max(config["candidateSearch"]["searchBudgets"]):
                    cell["policy_evaluation"]["paired_mean_difference"] = -0.02
    negative_result = aggregate(
        negative, config, implementation_audit["fixture_metrics"]
    )
    aggregation_ok = (
        aggregated["passed"] and len(aggregated["checks"]) == 17
        and not missing["passed"]
        and not negative_result["passed"]
        and not negative_result["checks"][
            "scale_high_budget_candidate_minus_observation_blind_return"
        ]
    )
    if not aggregation_ok:
        errors.append("V59 fixed-denominator or noncompensatory aggregation failed")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v59-evaluation-implementation-lock.json",
            "configs/v59-outcome-lock.json",
            "outputs/v59-budgeted-root-sampled-planning/evaluation-attempt.json",
            "outputs/v59-budgeted-root-sampled-planning/evaluation",
            "outputs/v59-budgeted-root-sampled-planning/post-result-audit.json",
            "docs/v59-results.md",
        )
    )
    if not downstream_absent:
        errors.append("V59 candidate evaluation or downstream artifact exists")

    audit = {
        "schema_version": 59,
        "experiment": "v59_evaluation_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v59_evaluation_implementation_lock" if not errors
            else "repair_v59_evaluation_implementation"
        ),
        "errors": errors,
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in EVALUATION_FILES
        },
        "frozen_dependencies_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in FROZEN_DEPENDENCIES
        },
        "checks": {
            "sealed_population_metadata_and_implementation": seal_bound,
            "public_only_candidate_firewall": firewall_ok,
            "altered_seed_exact_search_and_policy_evaluator": evaluator_fixture_ok,
            "fixed_denominator_and_seventeen_noncompensatory_gates": aggregation_ok,
            "single_attempt_and_downstream_absence": downstream_absent,
        },
        "fixture_metrics": {
            "belief_atoms": fixture_result["belief_atoms"],
            "candidate_simulations": fixture_cell["candidate"]["simulations_run"],
            "control_simulations": fixture_cell["observation_blind_control"]["simulations_run"],
            "deterministic_replay_match": fixture_cell["deterministic_replay_match"],
            "qualification_check_count": len(aggregated["checks"]),
        },
        "data_access": {
            "v59_candidate_population_records_accessed": 0,
            "v59_audit_truth_records_accessed": 0,
            "v59_candidate_evaluation_runs": 0,
            "altered_seed_evaluator_fixture_records": 1,
            "human_authored_v58_records": 0,
            "model_forward_passes": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
