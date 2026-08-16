#!/usr/bin/env python3
"""Audit V60's SMC²-to-planner interface before candidate evaluation."""
from __future__ import annotations

import argparse
import copy
import json

import v59_planning as v59
from generate_v59_planning import build_record, prior_observation_design_keys
from v10_protocol import file_sha256
from v22_relational import canonical_json, unary_atom
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import exact_inference, pool_smc2_repeats, smc2_inference
from v54_eig import belief_atoms_from_exact
from v55r1_planning import planning_registry
from v60_decision_calibration import (
    atom_marginals,
    belief_comparison,
    forbidden_truth_conditioned_belief,
    normalized_inference,
    plan_domain_fast,
    run_root_sampled_uct_fast,
    smc2_atoms_for_planning,
)


IMPLEMENTATION_FILES = (
    "python/v60_decision_calibration.py",
    "python/test_v60_decision_calibration.py",
    "python/audit_v60_implementation.py",
)

BASE_DEPENDENCIES = (
    "python/v53_smc2.py",
    "python/v54_eig.py",
    "python/v55_planning.py",
    "python/v55r1_planning.py",
    "python/v59_planning.py",
    "python/generate_v59_planning.py",
    "configs/v53r2-design-lock.json",
    "configs/v53r2-outcome-lock.json",
    "configs/v59-outcome-lock.json",
    "configs/v59-population-seal.json",
)


def analytic_search(rows, seed=6071):
    actions = [
        {"key": "guess_a", "action": {"id": "guess_a"}},
        {"key": "guess_b", "action": {"id": "guess_b"}},
    ]
    return run_root_sampled_uct_fast(
        rows, actions, 1, 0, 512, seed,
        lambda state, action, tick, draw: ({
            **state, "correct": action["id"][-1] == state["mode"],
        }, "terminal"),
        lambda state: float(state.get("correct", False)),
        lambda action: 0.0,
        lambda state: state["mode"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v60-design-lock.json")
    parser.add_argument(
        "--output",
        default="outputs/v60-approximate-belief-decision-calibration/implementation-audit.json",
    )
    args = parser.parse_args()
    design_path = (PROJECT_ROOT / args.design_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    design = json.loads(design_path.read_text())
    config = design["config_payload"]
    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    v55r1 = json.loads(
        (PROJECT_ROOT / "configs/v55r1-design-lock.json").read_text()
    )["config_payload"]
    registry = planning_registry(v55r1)
    errors: list[str] = []

    design_bound = (
        design["authorization"]["write_and_audit_v60_implementation"]
        and not design["authorization"]["run_v60_evaluation"]
        and not design["authorization"]["construct_v60_population"]
        and not design["authorization"]["access_v59_audit_truth"]
        and design["config_sha256"] == file_sha256(PROJECT_ROOT / design["config"])
        and design["population_seal_sha256"]
        == file_sha256(PROJECT_ROOT / design["population_seal"])
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for path, digest in design["source_outcome_locks_sha256"].items()
        )
    )
    if not design_bound:
        errors.append("V60 implementation is not bound to the frozen design")

    fixture_config = copy.deepcopy(config)
    v59_config = json.loads(
        (PROJECT_ROOT / "configs/v59-design-lock.json").read_text()
    )["config_payload"]
    for key, value in tuple(v59_config["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            v59_config["population"][key] = value + 10_000_000
    used, prior = set(), prior_observation_design_keys()
    public_fixture, _ = build_record(
        0, 1, {"atom": unary_atom("active", "unit_1"), "value": True},
        3, registry, v59_config, used, prior,
    )
    public = public_fixture["public"]
    record = {
        "id": public_fixture["id"],
        "supports": public["supports"], "query": public["query"],
    }
    exact_config = copy.deepcopy(v53)
    exact_config["exactBenchmark"]["quadratureNodes"] = 5
    exact = exact_inference(registry, record, exact_config)
    exact_atoms = belief_atoms_from_exact(exact)
    linear = v59.plan_domain(
        exact_atoms, registry, public["query"]["entities"], public["goal"],
        3, public["query"]["prefix_length"], 64, 6061, v59_config,
    )
    fast = plan_domain_fast(
        exact_atoms, registry, public["query"]["entities"], public["goal"],
        3, public["query"]["prefix_length"], 64, 6061, v59_config,
    )
    sampler_replay_ok = (
        linear.tree_sha256 == fast.tree_sha256
        and linear.selected_action_key == fast.selected_action_key
        and linear.root_action_rows == fast.root_action_rows
        and linear.root_sample_counts == fast.root_sample_counts
    )
    if not sampler_replay_ok:
        errors.append("V60 CDF sampler does not replay V59 semantics")

    repeats = [
        smc2_inference(registry, record, v53, 31, repeat, "v60-fixture")
        for repeat in range(2)
    ]
    pooled = pool_smc2_repeats(repeats)
    converted = smc2_atoms_for_planning(pooled)
    converted_marginals = atom_marginals(
        converted, len(registry), v53["exactBenchmark"]["thetaBins"],
        v53["parameterModel"],
    )
    pooled_marginals = {
        key: pooled[key]
        for key in (
            "program", "theta_values", "theta_weights", "joint_bins", "configuration"
        )
    }
    conversion = belief_comparison(pooled_marginals, converted_marginals)
    conversion_ok = (
        normalized_inference(pooled)
        and abs(sum(row["weight"] for row in converted) - 1.0) <= 1e-12
        and all(value <= 1e-12 for value in conversion.values())
        and all("world" in row and "queue" in row and "node_index" in row for row in converted)
    )
    if not conversion_ok:
        errors.append("V60 particle-to-planner conversion changes posterior marginals")

    configuration_keys = sorted(pooled["configuration"])
    synthetic_exact = {
        "program": [0.8, 0.2],
        "theta_values": [0.2, 0.8], "theta_weights": [0.8, 0.2],
        "joint_bins": {"0:1": 0.8, "1:8": 0.2},
        "configuration": {configuration_keys[0]: 0.8, configuration_keys[-1]: 0.2},
    }
    uniform_program = {**synthetic_exact, "program": [0.5, 0.5]}
    theta_point = {
        **synthetic_exact, "theta_values": [0.32], "theta_weights": [1.0]
    }
    configuration_map = {
        **synthetic_exact, "configuration": {configuration_keys[0]: 1.0}
    }
    exact_belief_rows = [
        {"key": "a", "mode": "a", "weight": 0.9},
        {"key": "b", "mode": "b", "weight": 0.1},
    ]
    approximate_belief_rows = [
        {"key": "a", "mode": "a", "weight": 0.1},
        {"key": "b", "mode": "b", "weight": 0.9},
    ]
    exact_search = analytic_search(exact_belief_rows)
    approximate_search = analytic_search(approximate_belief_rows)
    truth_rejected = False
    try:
        forbidden_truth_conditioned_belief({"target_program_index": 1})
    except PermissionError:
        truth_rejected = True
    mutant_results = {
        "uniform_program_weight": belief_comparison(
            synthetic_exact, uniform_program
        )["program_tv"] > 0.1,
        "theta_point_mass": belief_comparison(
            synthetic_exact, theta_point
        )["theta_wasserstein"] > 0.1,
        "configuration_map": belief_comparison(
            synthetic_exact, configuration_map
        )["configuration_tv"] > 0.1,
        "approximate_belief_ignored": (
            exact_search.selected_action_key != approximate_search.selected_action_key
        ),
        "truth_conditioned_belief": truth_rejected,
    }
    mutant_kill_rate = sum(mutant_results.values()) / len(mutant_results)
    mutants_ok = mutant_kill_rate == 1.0
    if not mutants_ok:
        errors.append("V60 implementation controls did not detect every mutant")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v60-implementation-lock.json",
            "configs/v60-evaluation-implementation-lock.json",
            "configs/v60-outcome-lock.json",
            "outputs/v60-approximate-belief-decision-calibration/evaluation-attempt.json",
            "outputs/v60-approximate-belief-decision-calibration/evaluation",
            "docs/v60-results.md",
        )
    )
    if not downstream_absent:
        errors.append("V60 evaluation or downstream artifact already exists")

    audit = {
        "schema_version": 60,
        "experiment": "v60_implementation_audit",
        "passed": not errors,
        "decision": (
            "freeze_v60_implementation" if not errors else "repair_v60_implementation"
        ),
        "errors": errors,
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION_FILES
        },
        "base_dependencies_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in BASE_DEPENDENCIES
        },
        "checks": {
            "frozen_design_and_source_bindings": design_bound,
            "cdf_sampler_exact_v59_domain_replay": sampler_replay_ok,
            "smc2_particle_to_planner_marginal_preservation": conversion_ok,
            "all_five_implementation_mutants_detected": mutants_ok,
            "downstream_absence": downstream_absent,
        },
        "fixture_metrics": {
            "linear_tree_sha256": linear.tree_sha256,
            "fast_tree_sha256": fast.tree_sha256,
            "converted_atoms": len(converted),
            "conversion_errors": conversion,
            "implementation_mutant_kill_rate": mutant_kill_rate,
            "mutant_results": mutant_results,
            "exact_belief_action": exact_search.selected_action_key,
            "approximate_belief_action": approximate_search.selected_action_key,
        },
        "data_access": {
            "v59_candidate_public_records_accessed": 0,
            "v59_audit_truth_records_accessed": 0,
            "altered_seed_fixture_records": 1,
            "v60_evaluation_runs": 0,
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
