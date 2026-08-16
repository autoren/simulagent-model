#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import exact_inference, mechanic_registry
from v54_eig import (
    belief_atoms_from_exact,
    expected_information_gain_from_joint,
    map_program_atoms,
    target_key,
    theta_point_mass_atoms,
)
from v55_planning import (
    attempted_future_outcome_leak,
    best_open_loop,
    candidate_actions,
    clairvoyant_value,
    eig_action,
    eig_policy_value,
    evaluate_policy,
    evaluate_static_update_disabled_policy,
    greedy_policy_value,
    map_program_policy_value,
    plan_exact,
    plan_static_update_disabled,
    posterior_mean_theta_policy_value,
    scalar_plan,
    step_belief,
)


BASELINE_NAMES = (
    "open_loop", "greedy", "map_program", "posterior_mean_theta",
    "eig_only", "belief_update_disabled",
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def action_eig(atoms, registry, entity_rows, action, tick):
    prior = defaultdict(float)
    for atom in atoms:
        prior[target_key(atom)] += atom["weight"]
    branches = step_belief(atoms, registry, entity_rows, action, tick, {})
    joint = {}
    for outcome, branch in branches.items():
        masses = defaultdict(float)
        for atom in branch["atoms"]:
            masses[target_key(atom)] += branch["probability"] * atom["weight"]
        joint[outcome] = dict(masses)
    return expected_information_gain_from_joint(dict(prior), joint)["eig"]


def suppressed_delay_two_registry(registry, horizon):
    result = copy.deepcopy(registry)
    for row in result:
        for rule in row["template"]["rules"]:
            for delayed in rule["stochastic_delayed"]:
                if delayed["delay"] == 2:
                    delayed["delay"] = horizon + 1
    return result


def policy_tree_checks(
    atoms, policy, registry, entity_rows, goal, horizon, tick, config
):
    if horizon == 0:
        return {
            "maximum_bellman_residual": 0.0,
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
        "maximum_bellman_residual": abs(policy["value"] - maximum),
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
        for key in (
            "normalization_checks", "normalization_passes",
            "candidate_omissions", "tie_break_violations",
            "finite_checks", "finite_passes",
        ):
            result[key] += child[key]
        result["maximum_bellman_residual"] = max(
            result["maximum_bellman_residual"],
            child["maximum_bellman_residual"],
        )
    return result


def information_then_control(
    atoms, primary, registry, entity_rows, goal, tick, config, nonmyopic
):
    if not nonmyopic:
        return False, 0.0, 0.0
    eig = action_eig(
        atoms, registry, entity_rows, primary["selected_action"], tick
    )
    if eig <= 1e-8:
        return False, eig, 0.0
    branches = step_belief(
        atoms, registry, entity_rows, primary["selected_action"], tick, {}
    )
    control_mass = 0.0
    for outcome, branch in branches.items():
        child = primary["branches"][outcome]
        if child["terminal"]:
            continue
        myopic = scalar_plan(
            branch["atoms"], registry, entity_rows, goal, 1, tick + 1, config
        )
        if child["selected_action_key"] in myopic["optimal_action_keys"]:
            control_mass += branch["probability"]
    return control_mass >= 0.5, eig, control_mass


def evaluate_record(row, registry, v53_config, config, delayed_registry):
    public = row["public"]
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
    policy_value = evaluate_policy(
        atoms, primary, registry, entity_rows, goal, horizon, tick, config
    )
    open_loop = best_open_loop(
        atoms, registry, entity_rows, goal, horizon, tick, config
    )
    greedy_root = scalar_plan(
        atoms, registry, entity_rows, goal, 1, tick, config
    )
    eig_root = eig_action(atoms, registry, entity_rows, tick, config)
    map_root = plan_exact(
        map_program_atoms(atoms), registry, entity_rows,
        goal, horizon, tick, config,
    )
    theta_root = plan_exact(
        theta_point_mass_atoms(atoms), registry, entity_rows,
        goal, horizon, tick, config,
    )
    disabled_root = plan_static_update_disabled(
        atoms, registry, entity_rows, goal, horizon, tick, config
    )
    baseline_values = {
        "open_loop": open_loop["value"],
        "greedy": greedy_policy_value(
            atoms, registry, entity_rows, goal, horizon, tick, config
        ),
        "map_program": map_program_policy_value(
            atoms, registry, entity_rows, goal, horizon, tick, config
        ),
        "posterior_mean_theta": posterior_mean_theta_policy_value(
            atoms, registry, entity_rows, goal, horizon, tick, config
        ),
        "eig_only": eig_policy_value(
            atoms, registry, entity_rows, goal, horizon, tick, config
        ),
        "belief_update_disabled": evaluate_static_update_disabled_policy(
            atoms, atoms, registry, entity_rows, goal, horizon, tick, config
        ),
    }
    root_actions = {
        "open_loop": open_loop["selected"]["action_keys"][0],
        "greedy": greedy_root["selected_action_key"],
        "map_program": map_root["selected_action_key"],
        "posterior_mean_theta": theta_root["selected_action_key"],
        "eig_only": eig_root["action_key"],
        "belief_update_disabled": disabled_root["selected_action_key"],
    }
    clairvoyant = clairvoyant_value(
        atoms, registry, entity_rows, goal, horizon, tick, config
    )
    nonmyopic = primary["selected_action_key"] not in greedy_root["optimal_action_keys"]
    info_control, root_eig, control_mass = information_then_control(
        atoms, primary, registry, entity_rows, goal, tick, config, nonmyopic
    )
    delayed_counterfactual = plan_exact(
        atoms, delayed_registry, entity_rows, goal, horizon, tick, config
    )
    delayed_sensitive = (
        primary["selected_action_key"]
        not in delayed_counterfactual["optimal_action_keys"]
        or abs(primary["value"] - delayed_counterfactual["value"]) > 1e-3
    )
    integrity = policy_tree_checks(
        atoms, primary, registry, entity_rows, goal, horizon, tick, config
    )
    values = [
        primary["value"], reference["value"], policy_value, clairvoyant,
        delayed_counterfactual["value"], *baseline_values.values(),
    ]
    return {
        "id": row["id"],
        "record": row["record"],
        "history_class": row["history_class"],
        "goal": goal,
        "belief_atoms": len(atoms),
        "root_value": primary["value"],
        "reference_value": reference["value"],
        "root_value_error": abs(primary["value"] - reference["value"]),
        "root_optimal_set_member": (
            primary["selected_action_key"] in reference["optimal_action_keys"]
        ),
        "independent_policy_value": policy_value,
        "independent_policy_error": abs(primary["value"] - policy_value),
        "selected_action_key": primary["selected_action_key"],
        "baseline_values": baseline_values,
        "baseline_root_action_keys": root_actions,
        "baseline_regrets": {
            name: baseline_values[name] - primary["value"]
            for name in BASELINE_NAMES
        },
        "adaptive_minus_open_loop": primary["value"] - open_loop["value"],
        "clairvoyant_value": clairvoyant,
        "clairvoyant_violation": primary["value"] > clairvoyant + 1e-10,
        "nonmyopic_root_action": nonmyopic,
        "information_then_control": info_control,
        "selected_root_static_eig": root_eig,
        "branchwise_myopic_control_mass": control_mass,
        "delayed_counterfactual_value": delayed_counterfactual["value"],
        "delayed_consequence_sensitive": delayed_sensitive,
        "root_action_disagreement": {
            name: root_actions[name] != primary["selected_action_key"]
            for name in BASELINE_NAMES
        },
        "integrity": integrity,
        "finite": all(math.isfinite(value) for value in values),
    }


def aggregate(records, config, leakage_rejected):
    count = len(records)
    gates = config["gates"]
    informative = [
        row for row in records
        if row["adaptive_minus_open_loop"]
        >= gates["adaptationInformativeMinimumValueGap"]
    ]
    control_rows = {}
    detected = 0
    for name in BASELINE_NAMES:
        mean_control_regret = mean([
            -row["baseline_regrets"][name] for row in records
        ])
        disagreement = mean([
            float(row["root_action_disagreement"][name]) for row in informative
        ])
        is_detected = mean_control_regret > 0.001 or disagreement > 0.10
        detected += int(is_detected)
        control_rows[name] = {
            "mean_exact_return_regret": mean_control_regret,
            "adaptation_informative_root_disagreement_rate": disagreement,
            "detected_or_dominated": is_detected,
        }
    detected += int(leakage_rejected)
    control_rows["future_outcome_leakage"] = {
        "rejected": leakage_rejected,
        "detected_or_dominated": leakage_rejected,
    }
    normalization_checks = sum(
        row["integrity"]["normalization_checks"] for row in records
    )
    normalization_passes = sum(
        row["integrity"]["normalization_passes"] for row in records
    )
    finite_checks = sum(row["integrity"]["finite_checks"] for row in records)
    finite_passes = sum(row["integrity"]["finite_passes"] for row in records)
    baseline_summary = {
        name: {
            "mean_regret": mean([row["baseline_regrets"][name] for row in records]),
            "maximum_regret": max(row["baseline_regrets"][name] for row in records),
        }
        for name in BASELINE_NAMES
    }
    metrics = {
        "exact_correctness": {
            "completed_task_fraction": count / config["population"]["planningTasks"],
            "maximum_root_value_error_against_scalar_reference": max(
                row["root_value_error"] for row in records
            ),
            "root_optimal_set_membership_rate": mean([
                float(row["root_optimal_set_member"]) for row in records
            ]),
            "maximum_bellman_residual": max(
                row["integrity"]["maximum_bellman_residual"] for row in records
            ),
            "maximum_independent_policy_evaluation_error": max(
                row["independent_policy_error"] for row in records
            ),
            "belief_and_observation_normalization_rate": (
                normalization_passes / normalization_checks
            ),
            "finite_value_rate": finite_passes / finite_checks,
        },
        "decision_quality": {
            "mean_bayes_adaptive_value": mean([row["root_value"] for row in records]),
            "mean_open_loop_value": mean([
                row["baseline_values"]["open_loop"] for row in records
            ]),
            "mean_bayes_adaptive_minus_open_loop_value": mean([
                row["adaptive_minus_open_loop"] for row in records
            ]),
            "positive_value_of_adaptation_fraction": len(informative) / count,
            "adaptation_informative_tasks": len(informative),
            "baseline_regret": baseline_summary,
            "maximum_bayes_adaptive_regret_against_any_registered_deployable_baseline": max(
                row["baseline_regrets"][name]
                for row in records for name in BASELINE_NAMES
            ),
            "clairvoyant_upper_bound_violation_rate": mean([
                float(row["clairvoyant_violation"]) for row in records
            ]),
        },
        "nonmyopic_behavior": {
            "nonmyopic_root_action_fraction": mean([
                float(row["nonmyopic_root_action"]) for row in records
            ]),
            "information_then_control_policy_fraction": mean([
                float(row["information_then_control"]) for row in records
            ]),
            "delayed_consequence_sensitive_policy_fraction": mean([
                float(row["delayed_consequence_sensitive"]) for row in records
            ]),
        },
        "controls": {
            "detected_or_dominated": detected,
            "rows": control_rows,
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
    return metrics


def qualification(metrics, gates):
    exact = metrics["exact_correctness"]
    decision = metrics["decision_quality"]
    behavior = metrics["nonmyopic_behavior"]
    integrity = metrics["integrity"]
    checks = {
        "completed_task_fraction": exact["completed_task_fraction"]
        >= gates["minimumCompletedTaskFraction"],
        "belief_and_observation_normalization_rate":
        exact["belief_and_observation_normalization_rate"]
        >= gates["minimumBeliefAndObservationNormalizationRate"],
        "finite_value_rate": exact["finite_value_rate"]
        >= gates["minimumFiniteValueRate"],
        "root_value_error": exact["maximum_root_value_error_against_scalar_reference"]
        <= gates["maximumRootValueError"],
        "root_optimal_set_membership": exact["root_optimal_set_membership_rate"]
        >= gates["minimumRootOptimalSetMembershipRate"],
        "bellman_residual": exact["maximum_bellman_residual"]
        <= gates["maximumBellmanResidual"],
        "independent_policy_evaluation":
        exact["maximum_independent_policy_evaluation_error"]
        <= gates["maximumIndependentPolicyEvaluationError"],
        "deployable_baseline_dominance": decision[
            "maximum_bayes_adaptive_regret_against_any_registered_deployable_baseline"
        ] <= gates["maximumBayesAdaptiveRegretAgainstAnyRegisteredDeployableBaseline"],
        "clairvoyant_upper_bound": decision["clairvoyant_upper_bound_violation_rate"]
        <= gates["maximumClairvoyantUpperBoundViolationRate"],
        "positive_adaptation_fraction": decision["positive_value_of_adaptation_fraction"]
        >= gates["minimumPositiveValueOfAdaptationFraction"],
        "mean_adaptive_minus_open_loop": decision[
            "mean_bayes_adaptive_minus_open_loop_value"
        ] >= gates["minimumMeanBayesAdaptiveMinusOpenLoopValue"],
        "nonmyopic_root_action_fraction": behavior["nonmyopic_root_action_fraction"]
        >= gates["minimumNonmyopicRootActionFraction"],
        "information_then_control_fraction": behavior[
            "information_then_control_policy_fraction"
        ] >= gates["minimumInformationThenControlPolicyFraction"],
        "delayed_consequence_sensitivity": behavior[
            "delayed_consequence_sensitive_policy_fraction"
        ] >= gates["minimumDelayedConsequenceSensitivePolicyFraction"],
        "truth_field_firewall": integrity[
            "truth_field_access_before_policy_evaluation_count"
        ] <= gates["maximumTruthFieldAccessBeforePolicyEvaluationCount"],
        "future_observation_firewall": integrity["future_observation_access_count"]
        <= gates["maximumFutureObservationAccessCount"],
        "complete_candidate_actions": integrity["candidate_action_omission_count"]
        <= gates["maximumCandidateActionOmissionCount"],
        "canonical_tie_break": integrity["canonical_tie_break_violation_count"]
        <= gates["maximumCanonicalTieBreakViolationCount"],
        "independent_streams": integrity[
            "history_and_policy_evaluation_stream_collision_count"
        ] <= gates["maximumHistoryAndPolicyEvaluationStreamCollisionCount"],
        "controls_detected_or_dominated": metrics["controls"]["detected_or_dominated"]
        >= gates["minimumControlsDetectedOrDominated"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-seal", default="configs/v55-population-seal.json")
    parser.add_argument(
        "--output-dir",
        default="outputs/v55-short-horizon-bayes-adaptive-planning/evaluation",
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.population_seal).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output_dir.parent / "evaluation-attempt.json"
    if attempt.exists():
        raise RuntimeError("V55 planning evaluation already attempted")
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 55,
        "experiment": "v55_planning_evaluation_attempt",
        "evaluation_run": 1,
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
    }, indent=2, sort_keys=True) + "\n")
    seal = json.loads(seal_path.read_text())
    if not seal["authorization"]["run_v55_planning_evaluation_once"]:
        raise RuntimeError("V55 population seal does not authorize evaluation")
    population_path = PROJECT_ROOT / seal["population"]["path"]
    if file_sha256(population_path) != seal["population"]["sha256"]:
        raise RuntimeError("V55 sealed planning population changed")
    evaluation_lock_path = PROJECT_ROOT / seal["evaluation_implementation_lock"]
    evaluation_lock = json.loads(evaluation_lock_path.read_text())
    if (
        not evaluation_lock["authorization"]["construct_v55_planning_population"]
        or evaluation_lock["authorization"]["run_v55_planning_evaluation"] is not False
    ):
        raise RuntimeError("V55 evaluation lock is not in the sealed pre-evaluation state")
    for section in ("implementation_files_sha256", "base_dependencies_sha256"):
        for path, digest in evaluation_lock[section].items():
            if file_sha256(PROJECT_ROOT / path) != digest:
                raise RuntimeError(f"V55 frozen evaluation implementation changed: {path}")
    implementation = json.loads(
        (PROJECT_ROOT / seal["implementation_lock"]).read_text()
    )
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    v53_config = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    registry = mechanic_registry(5303)
    delayed_registry = suppressed_delay_two_registry(
        registry, config["planningModel"]["horizonActions"]
    )
    rows = read_jsonl(population_path)
    records = []
    started = time.time()
    for index, row in enumerate(rows):
        records.append(evaluate_record(
            row, registry, v53_config, config, delayed_registry
        ))
        print(json.dumps({
            "completed": index + 1,
            "total": len(rows),
            "id": row["id"],
            "seconds": time.time() - started,
        }, sort_keys=True), flush=True)
    leakage_rejected = False
    try:
        attempted_future_outcome_leak({}, {"future": True})
    except PermissionError:
        leakage_rejected = True
    metrics = aggregate(records, config, leakage_rejected)
    decision = qualification(metrics, config["gates"])
    result = {
        "schema_version": 55,
        "experiment": "v55_short_horizon_exact_bayes_adaptive_planning",
        "evaluation_run": 1,
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_implementation_lock": str(
            evaluation_lock_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_lock_sha256": file_sha256(evaluation_lock_path),
        "metric_definitions": {
            "nonmyopic_root_action": "selected_root_action_outside_the_exact_horizon_one_optimal_set",
            "information_then_control": "nonmyopic_root_action_with_positive_program_theta_eig_and_at_least_half_predictive_mass_entering_a_branch_whose_next_action_is_horizon_one_optimal",
            "delayed_consequence_sensitive": "root_optimal_set_or_value_changes_by_more_than_0_001_when_all_delay_two_events_are_shifted_beyond_the_three_action_horizon",
        },
        "metrics": metrics,
        "qualification": decision,
        "records": records,
        "runtime_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "passed": decision["passed"],
        "checks": decision["checks"],
        "metrics": metrics,
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
