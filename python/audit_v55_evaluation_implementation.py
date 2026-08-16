#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json

from evaluate_v55_planning import (
    evaluate_record,
    qualification,
    suppressed_delay_two_registry,
)
from generate_v55_planning import build_record, prior_observation_design_keys
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import mechanic_registry


EVALUATION_FILES = (
    "python/evaluate_v55_planning.py",
    "python/audit_v55_populations.py",
    "python/seal_v55_population.py",
    "python/audit_and_summarize_v55.py",
    "python/freeze_v55_outcome.py",
    "python/audit_v55_evaluation_implementation.py",
    "scripts/run-v55-short-horizon-bayes-adaptive-planning.sh",
)


def passing_fixture(config):
    gates = config["gates"]
    return {
        "exact_correctness": {
            "completed_task_fraction": 1.0,
            "maximum_root_value_error_against_scalar_reference": 0.0,
            "root_optimal_set_membership_rate": 1.0,
            "maximum_bellman_residual": 0.0,
            "maximum_independent_policy_evaluation_error": 0.0,
            "belief_and_observation_normalization_rate": 1.0,
            "finite_value_rate": 1.0,
        },
        "decision_quality": {
            "mean_bayes_adaptive_minus_open_loop_value": gates[
                "minimumMeanBayesAdaptiveMinusOpenLoopValue"
            ],
            "positive_value_of_adaptation_fraction": gates[
                "minimumPositiveValueOfAdaptationFraction"
            ],
            "maximum_bayes_adaptive_regret_against_any_registered_deployable_baseline": 0.0,
            "clairvoyant_upper_bound_violation_rate": 0.0,
        },
        "nonmyopic_behavior": {
            "nonmyopic_root_action_fraction": gates[
                "minimumNonmyopicRootActionFraction"
            ],
            "information_then_control_policy_fraction": gates[
                "minimumInformationThenControlPolicyFraction"
            ],
            "delayed_consequence_sensitive_policy_fraction": gates[
                "minimumDelayedConsequenceSensitivePolicyFraction"
            ],
        },
        "controls": {
            "detected_or_dominated": gates["minimumControlsDetectedOrDominated"]
        },
        "integrity": {
            "truth_field_access_before_policy_evaluation_count": 0,
            "future_observation_access_count": 0,
            "candidate_action_omission_count": 0,
            "canonical_tie_break_violation_count": 0,
            "history_and_policy_evaluation_stream_collision_count": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation-lock", default="configs/v55-implementation-lock.json"
    )
    parser.add_argument(
        "--output",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/evaluation-implementation-audit.json",
    )
    args = parser.parse_args()
    implementation_path = (PROJECT_ROOT / args.implementation_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    implementation = json.loads(implementation_path.read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    v53 = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    errors = []

    implementation_bound = (
        implementation["authorization"]["construct_v55_planning_population"]
        and not implementation["authorization"]["run_v55_planning_evaluation"]
        and all(
            file_sha256(PROJECT_ROOT / path) == digest
            for section in ("implementation_files_sha256", "base_dependencies_sha256")
            for path, digest in implementation[section].items()
        )
    )
    if not implementation_bound:
        errors.append("V55 core implementation lock is not intact")

    boundary = qualification(passing_fixture(config), config["gates"])
    qualification_ok = boundary["passed"] and len(boundary["checks"]) == 20
    if not qualification_ok:
        errors.append("V55 qualification rejects its boundary fixture or omits gates")

    fixture_config = copy.deepcopy(config)
    for key, value in tuple(fixture_config["population"].items()):
        if key.endswith("Seed") and isinstance(value, int):
            fixture_config["population"][key] = value + 7_000_000
    exact_config = copy.deepcopy(v53)
    exact_config["exactBenchmark"]["quadratureNodes"] = 5
    registry = mechanic_registry(5303)
    row = build_record(
        1, 1, True, registry, fixture_config, set(), prior_observation_design_keys()
    )
    evaluated = evaluate_record(
        row, registry, exact_config, fixture_config,
        suppressed_delay_two_registry(registry, 3),
    )
    altered_fixture_ok = (
        evaluated["root_value_error"] <= 1e-12
        and evaluated["independent_policy_error"] <= 1e-12
        and evaluated["root_optimal_set_member"]
        and not evaluated["clairvoyant_violation"]
        and set(evaluated["baseline_values"]) == {
            "open_loop", "greedy", "map_program", "posterior_mean_theta",
            "eig_only", "belief_update_disabled",
        }
        and evaluated["integrity"]["candidate_omissions"] == 0
        and evaluated["integrity"]["tie_break_violations"] == 0
    )
    if not altered_fixture_ok:
        errors.append("V55 altered-seed evaluation fixture failed")

    evaluator_text = (PROJECT_ROOT / "python/evaluate_v55_planning.py").read_text()
    single_run_firewall = all(
        token in evaluator_text
        for token in (
            "evaluation already attempted", "evaluation-attempt.json",
            '"evaluation_run": 1',
        )
    )
    metric_definitions_frozen = all(
        token in evaluator_text
        for token in (
            "selected_root_action_outside_the_exact_horizon_one_optimal_set",
            "positive_program_theta_eig",
            "all_delay_two_events_are_shifted_beyond_the_three_action_horizon",
        )
    )
    if not single_run_firewall or not metric_definitions_frozen:
        errors.append("V55 single-run firewall or nonmyopic definitions are incomplete")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v55-evaluation-implementation-lock.json",
            "configs/v55-population-seal.json",
            "configs/v55-outcome-lock.json",
            "data/v55-short-horizon-bayes-adaptive-planning",
            "outputs/v55-short-horizon-bayes-adaptive-planning/population-audit.json",
            "outputs/v55-short-horizon-bayes-adaptive-planning/evaluation-attempt.json",
            "outputs/v55-short-horizon-bayes-adaptive-planning/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V55 population or evaluation exists before evaluation lock")

    audit = {
        "schema_version": 55,
        "experiment": "v55_evaluation_implementation_audit",
        "passed": not errors,
        "decision": (
            "authorize_v55_evaluation_implementation_lock" if not errors
            else "repair_v55_evaluation_implementation"
        ),
        "errors": errors,
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "implementation_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in EVALUATION_FILES
        },
        "base_dependencies_sha256": {
            "configs/v55-design-lock.json": file_sha256(
                PROJECT_ROOT / "configs/v55-design-lock.json"
            ),
            "configs/v55-implementation-lock.json": file_sha256(implementation_path),
        },
        "checks": {
            "core_implementation_lock_intact": implementation_bound,
            "qualification_boundary_fixture": qualification_ok,
            "qualification_check_count": len(boundary["checks"]),
            "altered_seed_full_evaluation_fixture": altered_fixture_ok,
            "single_run_firewall": single_run_firewall,
            "nonmyopic_metric_definitions_frozen": metric_definitions_frozen,
            "downstream_absent": downstream_absent,
        },
        "fixture_metrics": {
            "root_value_error": evaluated["root_value_error"],
            "independent_policy_error": evaluated["independent_policy_error"],
            "adaptive_minus_open_loop": evaluated["adaptive_minus_open_loop"],
            "nonmyopic_root_action": evaluated["nonmyopic_root_action"],
            "information_then_control": evaluated["information_then_control"],
            "delayed_consequence_sensitive": evaluated[
                "delayed_consequence_sensitive"
            ],
        },
        "data_access": {
            "v55_candidate_population_records_accessed": 0,
            "v55_population_generator_executions": 0,
            "v55_planning_evaluation_runs": 0,
            "altered_seed_evaluation_fixture_records": 1,
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
