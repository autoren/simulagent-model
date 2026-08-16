#!/usr/bin/env python3
"""Pre-run V60 evaluator audit using altered-seed public data only."""
from __future__ import annotations

import argparse
import copy
import inspect
import json

from evaluate_v60_decision_calibration import aggregate, evaluate_record, main as evaluation_main
from generate_v59_planning import build_record, prior_observation_design_keys
from v10_protocol import file_sha256
from v22_relational import unary_atom
from v22r2_grounding import PROJECT_ROOT
from v55r1_planning import planning_registry


EVALUATION_FILES = (
    "python/evaluate_v60_decision_calibration.py",
    "python/audit_and_summarize_v60.py",
    "python/freeze_v60_outcome.py",
    "python/audit_v60_evaluation_implementation.py",
)

FROZEN_DEPENDENCIES = (
    "python/v53_smc2.py", "python/v54_eig.py", "python/v55_planning.py",
    "python/v55r1_planning.py", "python/v59_planning.py",
    "python/v60_decision_calibration.py", "python/generate_v59_planning.py",
    "configs/v53r2-design-lock.json", "configs/v53r2-outcome-lock.json",
    "configs/v59-design-lock.json", "configs/v59-outcome-lock.json",
    "configs/v59-population-seal.json", "configs/v60-design-lock.json",
    "configs/v60-implementation-lock.json",
)


def passing_records(config: dict) -> list[dict]:
    budgets = config["inference"]["outerThetaParticleBudgets"]
    primary = config["inference"]["primaryBudget"]
    rows = []
    for index in range(config["population"]["publicTasks"]):
        horizon = config["claimBoundary"]["horizons"][index % 3]
        inference = [{
            "budget": budget, "normalized": True, "program_tv": 0.0,
            "theta_wasserstein": 0.0, "binned_program_theta_tv": 0.0,
            "configuration_tv": 0.0,
        } for budget in budgets]
        cells = []
        for budget in budgets:
            for replicate in range(config["planning"]["replicatesPerTask"]):
                exact_reference = None
                if horizon == config["planning"]["exactDynamicProgrammingReferenceHorizon"]:
                    exact_reference = {
                        "optimal_set_member": True, "root_regret": 0.0,
                        "selected_action_exact_value": 0.5,
                    }
                candidate_return = 0.5 + 0.01 * (budget == primary)
                cells.append({
                    "inference_budget": budget, "replicate": replicate,
                    "exact_belief_search": {"selected_action_key": "a", "simulations_run": 1024},
                    "approximate_belief_search": {"selected_action_key": "a", "simulations_run": 1024},
                    "observation_blind_search": (
                        {"selected_action_key": "b", "simulations_run": 1024}
                        if budget == primary else None
                    ),
                    "deterministic_replay_match": True,
                    "exact_minus_approximate_policy": {
                        "candidate_mean_return": candidate_return,
                        "control_mean_return": candidate_return,
                        "paired_mean_difference": 0.0,
                    },
                    "approximate_minus_observation_blind_policy": (
                        {"candidate_mean_return": candidate_return,
                         "control_mean_return": candidate_return - 0.02,
                         "paired_mean_difference": 0.02}
                        if budget == primary else None
                    ),
                    "exact_dynamic_programming_reference": exact_reference,
                    "finite": True,
                })
        rows.append({"id": f"stub_{index:03d}", "horizon": horizon, "inference": inference, "cells": cells})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v60-implementation-lock.json")
    parser.add_argument(
        "--output", default="outputs/v60-approximate-belief-decision-calibration/evaluation-implementation-audit.json"
    )
    args = parser.parse_args()
    implementation_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    implementation = json.loads(implementation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    v53 = json.loads((PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text())["config_payload"]
    v59 = json.loads((PROJECT_ROOT / "configs/v59-design-lock.json").read_text())["config_payload"]
    v55r1 = json.loads((PROJECT_ROOT / "configs/v55r1-design-lock.json").read_text())["config_payload"]
    registry = planning_registry(v55r1)
    implementation_audit = json.loads((PROJECT_ROOT / implementation["implementation_audit"]).read_text())
    errors: list[str] = []

    seal_path = PROJECT_ROOT / design["population_seal"]
    seal = json.loads(seal_path.read_text())
    source_bound = (
        implementation["authorization"]["write_and_audit_v60_evaluator"]
        and not implementation["authorization"]["run_v60_evaluation"]
        and not implementation["authorization"]["access_v59_audit_truth"]
        and file_sha256(seal_path) == design["population_seal_sha256"]
        and all(file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"] for row in seal["artifacts"].values())
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for section in ("implementation_files_sha256", "base_dependencies_sha256")
            for path, digest in implementation[section].items()
        )
    )
    if not source_bound:
        errors.append("V60 evaluator is not bound to its frozen implementation and source population")

    evaluator_source = inspect.getsource(evaluate_record)
    main_source = inspect.getsource(evaluation_main)
    firewall_ok = (
        set(inspect.signature(evaluate_record).parameters)
        == {"public_record", "registry", "v53_config", "v59_config", "config"}
        and "truth" not in evaluator_source
        and "future_observation" not in evaluator_source
        and 'seal["artifacts"]["public_file"]' in main_source
        and 'seal["artifacts"]["audit_truth_file"]' not in main_source
        and "assert_search_payload_is_public(public_record)" in evaluator_source
    )
    if not firewall_ok:
        errors.append("V60 evaluator crosses the sealed public-input firewall")

    fixture_v59 = copy.deepcopy(v59)
    for key, value in tuple(fixture_v59["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            fixture_v59["population"][key] = value + 11_000_000
    used, prior = set(), prior_observation_design_keys()
    public_fixture, _ = build_record(
        0, 0, {"atom": unary_atom("active", "unit_0"), "value": True},
        3, registry, fixture_v59, used, prior,
    )
    fixture_v53 = copy.deepcopy(v53)
    fixture_v53["exactBenchmark"]["quadratureNodes"] = 5
    fixture_v53["smcSquared"]["innerStateParticleBudget"] = 15
    fixture_config = copy.deepcopy(config)
    fixture_config["inference"]["outerThetaParticleBudgets"] = [7]
    fixture_config["inference"]["primaryBudget"] = 7
    fixture_config["inference"]["independentRepeatsPerBudget"] = 1
    fixture_config["planning"]["searchBudget"] = 32
    fixture_config["planning"]["replicatesPerTask"] = 1
    fixture_config["planning"]["policyEvaluationEpisodes"] = 64
    fixture_result = evaluate_record(
        public_fixture, registry, fixture_v53, fixture_v59, fixture_config
    )
    fixture_cell = fixture_result["cells"][0]
    fixture_ok = (
        len(fixture_result["inference"]) == len(fixture_result["cells"]) == 1
        and fixture_result["inference"][0]["normalized"]
        and fixture_cell["exact_belief_search"]["simulations_run"] == 32
        and fixture_cell["approximate_belief_search"]["simulations_run"] == 32
        and fixture_cell["observation_blind_search"]["simulations_run"] == 32
        and fixture_cell["deterministic_replay_match"] and fixture_cell["finite"]
        and fixture_cell["exact_dynamic_programming_reference"] is not None
    )
    if not fixture_ok:
        errors.append("V60 altered-seed end-to-end evaluator fixture failed")

    stubs = passing_records(config)
    aggregated = aggregate(stubs, config, implementation_audit["fixture_metrics"])
    missing = aggregate(stubs[:-1], config, implementation_audit["fixture_metrics"])
    biased = copy.deepcopy(stubs)
    for record in biased:
        for row in record["inference"]:
            if row["budget"] == config["inference"]["primaryBudget"]:
                row["configuration_tv"] = 0.5
    biased_result = aggregate(biased, config, implementation_audit["fixture_metrics"])
    aggregate_ok = (
        aggregated["passed"] and len(aggregated["checks"]) == 23
        and not missing["passed"] and not biased_result["passed"]
        and not biased_result["checks"]["primary_mean_configuration_tv"]
    )
    if not aggregate_ok:
        errors.append("V60 fixed-denominator or noncompensatory aggregation failed")

    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in (
        "configs/v60-evaluation-implementation-lock.json", "configs/v60-outcome-lock.json",
        "outputs/v60-approximate-belief-decision-calibration/evaluation-attempt.json",
        "outputs/v60-approximate-belief-decision-calibration/evaluation",
        "outputs/v60-approximate-belief-decision-calibration/post-result-audit.json",
        "docs/v60-results.md",
    ))
    if not downstream_absent:
        errors.append("V60 candidate evaluation or downstream artifact already exists")

    audit = {
        "schema_version": 60, "experiment": "v60_evaluation_implementation_audit",
        "passed": not errors,
        "decision": "freeze_v60_evaluation_implementation" if not errors else "repair_v60_evaluator",
        "errors": errors,
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_files_sha256": {path: file_sha256(PROJECT_ROOT / path) for path in EVALUATION_FILES},
        "frozen_dependencies_sha256": {path: file_sha256(PROJECT_ROOT / path) for path in FROZEN_DEPENDENCIES},
        "checks": {
            "frozen_implementation_and_source_population": source_bound,
            "sealed_public_only_candidate_firewall": firewall_ok,
            "altered_seed_end_to_end_evaluator": fixture_ok,
            "fixed_denominator_and_twenty_three_noncompensatory_gates": aggregate_ok,
            "single_attempt_and_downstream_absence": downstream_absent,
        },
        "fixture_metrics": {
            "exact_belief_atoms": fixture_result["exact_belief_atoms"],
            "approximate_planning_atoms": fixture_result["inference"][0]["planning_atoms_after_conversion"],
            "configuration_tv": fixture_result["inference"][0]["configuration_tv"],
            "deterministic_replay_match": fixture_cell["deterministic_replay_match"],
            "qualification_check_count": len(aggregated["checks"]),
        },
        "data_access": {
            "v59_candidate_public_records_accessed": 0, "v59_audit_records_accessed": 0,
            "altered_seed_fixture_records": 1, "v60_candidate_evaluation_runs": 0,
            "human_authored_v58_records": 0, "model_forward_passes": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
