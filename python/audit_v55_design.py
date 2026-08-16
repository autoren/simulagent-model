#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v55-short-horizon-bayes-adaptive-planning.json"
    )
    parser.add_argument(
        "--plan", default="docs/v55-short-horizon-bayes-adaptive-planning-plan.md"
    )
    parser.add_argument(
        "--output", default="outputs/v55-short-horizon-bayes-adaptive-planning/design-audit.json"
    )
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.output)
    )
    config = json.loads(config_path.read_text())
    source_path = (PROJECT_ROOT / config["sourceV54OutcomeLock"]).resolve()
    source = json.loads(source_path.read_text())
    errors = []

    source_bound = (
        source["qualification_passed"]
        and source["authorization"]["preregister_short_horizon_exact_bayes_adaptive_planning"]
        and not source["authorization"]["construct_planning_population"]
        and file_sha256(PROJECT_ROOT / source["result"]) == source["result_sha256"]
        and file_sha256(PROJECT_ROOT / source["post_result_audit"])
        == source["post_result_audit_sha256"]
        and file_sha256(PROJECT_ROOT / source["summary"]) == source["summary_sha256"]
    )
    if not source_bound:
        errors.append("V54 does not authorize or bind the V55 preregistration")

    boundary = config["claimBoundary"]
    boundary_ok = (
        boundary["shortHorizonPlanning"]
        and boundary["exactBayesAdaptiveBeliefPlanning"]
        and boundary["hiddenWorldQueueState"]
        and boundary["unknownProgramIdentity"]
        and boundary["unknownContinuousTheta"]
        and boundary["horizon"] == 3
        and not any(boundary[key] for key in (
            "learnedPlanner", "particlePlanning", "approximateSearch",
            "longHorizonClaim", "formalVerification", "languageGrounding",
            "openOntology", "modelAccess", "adapterTraining", "finalEvaluation",
        ))
    )
    if not boundary_ok:
        errors.append("V55 claim boundary is too broad or omits joint belief planning")

    planning = config["planningModel"]
    planning_ok = (
        planning["candidateCount"] == 5
        and planning["entityCount"] == 2
        and planning["horizonActions"] == 3
        and planning["withinHorizonAdaptation"]
        and planning["discountFactor"] == 1.0
        and planning["terminalReward"] == {
            "success": 1.0,
            "failure": 0.0,
            "goal": "one_preregistered_world_atom_equals_one_preregistered_boolean_at_the_terminal_tick",
        }
        and planning["actionCost"] == {"pulse": 0.01, "route": 0.01, "wait": 0.0}
        and planning["tieTolerance"] == 1e-12
    )
    if not planning_ok:
        errors.append("V55 finite-horizon action, observation, reward, or tie semantics are invalid")

    oracle = config["exactOracle"]
    oracle_ok = (
        oracle["programTemplates"] == 8
        and oracle["quadratureNodes"] == 257
        and oracle["primaryPlanner"] != oracle["independentReference"]
        and oracle["independentPolicyEvaluator"]
        == "exact_recursive_execution_of_the_frozen_selected_policy_under_the_root_belief"
        and len(oracle["analyticFixtures"]) == 5
    )
    if not oracle_ok:
        errors.append("V55 exact planner, scalar reference, or policy evaluator is underspecified")

    population = config["population"]
    population_ok = (
        population["planningTasks"] == 32
        and population["tasksPerGeneratingTemplate"] == 4
        and population["entityCount"] == 2
        and sum(population["historyClasses"].values()) == 32
        and population["goalBooleanBalance"] == "exact_16_false_16_true"
        and len(population["freshPublicHistoriesAgainst"]) == 6
    )
    if not population_ok:
        errors.append("V55 task quotas, balance, or freshness are inconsistent")

    baselines_ok = len(config["baselines"]) == 7 and "clairvoyant" in config["baselines"]
    controls = config["controls"]
    control_keys = [key for key in controls if key.endswith("Control")]
    controls_ok = (
        baselines_ok and len(control_keys) == 7
        and controls["minimumControlsDetectedOrDominated"]
        == config["gates"]["minimumControlsDetectedOrDominated"] == 5
    )
    if not controls_ok:
        errors.append("V55 baselines or controls are incomplete")

    gates = config["gates"]
    gates_ok = (
        gates["minimumCompletedTaskFraction"] == 1.0
        and gates["minimumBeliefAndObservationNormalizationRate"] == 1.0
        and gates["minimumRootOptimalSetMembershipRate"] == 1.0
        and gates["maximumRootValueError"] == 1e-10
        and gates["maximumBayesAdaptiveRegretAgainstAnyRegisteredDeployableBaseline"] == 1e-10
        and gates["minimumPositiveValueOfAdaptationFraction"] == 0.15
        and gates["minimumMeanBayesAdaptiveMinusOpenLoopValue"] == 0.002
        and all(gates[key] == 0 for key in (
            "maximumTruthFieldAccessBeforePolicyEvaluationCount",
            "maximumFutureObservationAccessCount",
            "maximumCandidateActionOmissionCount",
            "maximumCanonicalTieBreakViolationCount",
            "maximumHistoryAndPolicyEvaluationStreamCollisionCount",
        ))
    )
    if not gates_ok:
        errors.append("V55 correctness, adaptation, or integrity gates are invalid")

    firewall_ok = (
        set(config["firewall"].values()) == {"forbidden"}
        and config["stageAuthorization"] == {
            "writeAndAuditExactBayesAdaptivePlanner": True,
            "constructPlanningPopulation": False,
            "runPlanningEvaluation": False,
            "formalVerification": False,
            "languageGrounding": False,
            "modelAccess": False,
        }
    )
    if not firewall_ok:
        errors.append("V55 firewall or stage authorization is invalid")

    seeds = [
        value for key, value in population.items()
        if key.endswith("Seed") and isinstance(value, int)
    ]
    seeds_ok = len(seeds) == len(set(seeds)) == 8
    if not seeds_ok:
        errors.append("V55 random-stream root seeds are not distinct")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v55-design-lock.json",
            "configs/v55-implementation-lock.json",
            "configs/v55-population-seal.json",
            "configs/v55-outcome-lock.json",
            "data/v55-short-horizon-bayes-adaptive-planning",
            "outputs/v55-short-horizon-bayes-adaptive-planning/implementation-audit.json",
            "outputs/v55-short-horizon-bayes-adaptive-planning/evaluation-attempt.json",
            "outputs/v55-short-horizon-bayes-adaptive-planning/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V55 downstream artifact exists before design lock")

    audit = {
        "schema_version": 55,
        "experiment": "v55_design_audit",
        "passed": not errors,
        "decision": "authorize_v55_design_lock" if not errors else "repair_v55_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "checks": {
            "source_v54_authorization_and_binding": source_bound,
            "claim_boundary": boundary_ok,
            "finite_horizon_reward_and_observation_model": planning_ok,
            "exact_oracles": oracle_ok,
            "population_quotas_and_freshness": population_ok,
            "baselines_and_controls": controls_ok,
            "noncompensatory_gates": gates_ok,
            "firewall_and_stage_authorization": firewall_ok,
            "distinct_root_seeds": seeds_ok,
            "downstream_absent": downstream_absent,
        },
        "data_access": {
            "v55_candidate_population_records_accessed": 0,
            "v55_planning_evaluation_runs": 0,
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
