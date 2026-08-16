#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/v55r1-delayed-consequence-adequacy-confirmation.json",
    )
    parser.add_argument(
        "--plan",
        default="docs/v55r1-delayed-consequence-adequacy-confirmation-plan.md",
    )
    parser.add_argument(
        "--output",
        default="outputs/v55r1-delayed-consequence-adequacy-confirmation/design-audit.json",
    )
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.output)
    )
    config = json.loads(config_path.read_text())
    outcome_path = (PROJECT_ROOT / config["sourceV55OutcomeLock"]).resolve()
    diagnosis_path = (PROJECT_ROOT / config["sourceV55DelayLocalization"]).resolve()
    implementation_path = (
        PROJECT_ROOT / config["sourceV55ImplementationLock"]
    ).resolve()
    outcome = json.loads(outcome_path.read_text())
    diagnosis = json.loads(diagnosis_path.read_text())
    implementation = json.loads(implementation_path.read_text())
    errors: list[str] = []

    source_ok = (
        outcome["decision"] == "retain_v55_failure_and_localize_failed_gates"
        and not outcome["qualification"]["passed"]
        and sorted(
            name for name, passed in outcome["qualification"]["checks"].items()
            if not passed
        ) == ["delayed_consequence_sensitivity"]
        and diagnosis["decision"]
        == "authorize_preregistration_only_for_a_v55r1_delayed_consequence_adequacy_confirmation"
        and diagnosis["data_access"]["additional_planning_evaluation_runs"] == 0
        and diagnosis["data_access"]["exact_planner_calls"] == 0
        and diagnosis["localization"]["primary_cause"]
        == "control_task_adequacy_failure_from_sparse_goal_alignment_and_same_target_immediate_redundancy"
        and implementation["authorization"]["construct_v55_planning_population"]
        and file_sha256(PROJECT_ROOT / outcome["result"]) == outcome["result_sha256"]
        and file_sha256(PROJECT_ROOT / outcome["post_result_audit"])
        == outcome["post_result_audit_sha256"]
    )
    if not source_ok:
        errors.append("V55 failure, localization, or frozen implementation is not intact")

    relationship = config["relationshipToV55"]
    relationship_ok = (
        relationship["v55RemainsFailedAndFrozen"]
        and relationship["supplementaryConfirmationOnly"]
        and relationship["inheritNineteenPassingV55Gates"]
        and not relationship["rerunV55Population"]
        and not relationship["reinterpretV55Population"]
        and relationship["repairTarget"] == "delayed_consequence_sensitivity_only"
    )
    if not relationship_ok:
        errors.append("V55r1 improperly relabels or broadens the frozen V55 result")

    boundary = config["claimBoundary"]
    boundary_ok = (
        boundary["exactThreeActionBayesAdaptivePlanning"]
        and boundary["decisionRelevantDelayTwoConsequences"]
        and boundary["jointProgramThetaHiddenConfigurationBelief"]
        and boundary["planningSpecificLatentMechanicRegistry"]
        and boundary["supplementaryConfirmation"]
        and not any(boundary[key] for key in (
            "generalPlanningPopulationPass", "longHorizonClaim",
            "approximateSearch", "learnedPlanner", "formalVerification",
            "languageGrounding", "modelAccess", "adapterTraining",
            "finalEvaluation",
        ))
    )
    if not boundary_ok:
        errors.append("V55r1 claim boundary is too broad")

    planning = config["planningModel"]
    planning_ok = (
        planning["horizonActions"] == 3
        and planning["candidateCount"] == 5
        and planning["entityCount"] == 2
        and planning["actionCost"] == {"pulse": 0.01, "route": 0.01, "wait": 0.0}
        and planning["discountFactor"] == 1.0
        and planning["tieTolerance"] == 1e-12
    )
    if not planning_ok:
        errors.append("V55r1 changed the frozen planning semantics")

    registry = config["planningSpecificRegistry"]
    blueprints = registry["templateBlueprints"]
    registry_ok = (
        registry["templates"] == len(blueprints) == 8
        and registry["thetaBranchesPerTemplate"] == 1
        and registry["quadratureNodes"] == 257
        and registry["delayClassCounts"] == {
            "delay_two": 4, "delay_one": 2, "immediate": 2
        }
        and sum(row["timing"] == "delay_two" for row in blueprints) == 4
        and sum(row["timing"] == "delay_one" for row in blueprints) == 2
        and sum(row["timing"] == "immediate" for row in blueprints) == 2
        and all(row["trigger"] != row["distractorAction"] for row in blueprints)
        and len({json.dumps(row, sort_keys=True) for row in blueprints}) == 8
        and len(registry["requirements"]) == 9
    )
    if not registry_ok:
        errors.append("V55r1 planning-specific registry is incomplete or imbalanced")

    population = config["population"]
    seeds = [
        value for key, value in population.items()
        if key.endswith("Seed") and isinstance(value, int)
    ]
    population_ok = (
        population["confirmationTasks"] == 16
        and population["tasksPerGeneratingTemplate"] == 2
        and sum(population["historyClasses"].values()) == 16
        and population["supportEpisodesPerTask"] == 4
        and population["goalDependsOnGeneratingTruth"] is False
        and len(population["freshPublicHistoriesAgainst"]) == 7
        and len(seeds) == len(set(seeds)) == 8
    )
    if not population_ok:
        errors.append("V55r1 population balance, truth independence, or streams are invalid")

    counterfactual = config["counterfactual"]
    counterfactual_ok = (
        "all_delay_two_due_ticks" in counterfactual["delaySuppressed"]
        and "absolute_root_value_change_exceeds_0_001" in counterfactual["sensitiveTask"]
        and counterfactual["causalIsolation"]
        == "no_other_transition_reward_observation_action_or_inference_semantics_change"
    )
    if not counterfactual_ok:
        errors.append("V55r1 delay counterfactual is not causally isolated")

    exact = config["exactChecks"]
    exact_ok = (
        exact["primaryPlanner"] != exact["independentReference"]
        and "independent" in exact["independentPolicyEvaluator"]
        and len(exact["analyticFixtures"]) == 6
        and any("decision_relevance" in row for row in exact["analyticFixtures"])
    )
    if not exact_ok:
        errors.append("V55r1 exact references or decision-relevance fixtures are underspecified")

    gates = config["gates"]
    gates_ok = (
        gates["minimumCompletedTaskFraction"] == 1.0
        and gates["maximumRootValueError"] == 1e-10
        and gates["minimumRootOptimalSetMembershipRate"] == 1.0
        and gates["maximumIndependentPolicyEvaluationError"] == 1e-10
        and gates["minimumBeliefAndObservationNormalizationRate"] == 1.0
        and gates["minimumFiniteValueRate"] == 1.0
        and gates["minimumDelayedConsequenceSensitivePolicyFraction"] == 0.125
        and gates["minimumDelayedSensitiveTasksPerHistoryClass"] == 1
        and gates["minimumRootActionOrValueChangeTasks"] == 2
        and all(gates[key] == 0 for key in (
            "maximumTruthFieldAccessBeforePolicyEvaluationCount",
            "maximumFutureObservationAccessCount",
            "maximumCandidateActionOmissionCount",
            "maximumCanonicalTieBreakViolationCount",
            "maximumHistoryAndPolicyEvaluationStreamCollisionCount",
        ))
    )
    if not gates_ok:
        errors.append("V55r1 gates are incomplete or compensatory")

    firewall_ok = (
        set(config["firewall"].values()) == {"forbidden"}
        and config["stageAuthorization"] == {
            "writeAndAuditV55r1Implementation": True,
            "constructV55r1Population": False,
            "runV55r1Evaluation": False,
            "preregisterFormalVerification": False,
            "runFormalVerification": False,
            "languageGrounding": False,
            "modelAccess": False,
        }
    )
    if not firewall_ok:
        errors.append("V55r1 firewall or stage authorization is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v55r1-design-lock.json",
            "configs/v55r1-implementation-lock.json",
            "configs/v55r1-evaluation-implementation-lock.json",
            "configs/v55r1-population-seal.json",
            "configs/v55r1-outcome-lock.json",
            "data/v55r1-delayed-consequence-adequacy-confirmation",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/implementation-audit.json",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation-attempt.json",
            "outputs/v55r1-delayed-consequence-adequacy-confirmation/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V55r1 downstream artifact exists before design lock")

    audit = {
        "schema_version": 55,
        "revision": "r1",
        "experiment": "v55r1_design_audit",
        "passed": not errors,
        "decision": "authorize_v55r1_design_lock" if not errors else "repair_v55r1_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(outcome_path),
        "source_localization": str(diagnosis_path.relative_to(PROJECT_ROOT)),
        "source_localization_sha256": file_sha256(diagnosis_path),
        "source_implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "source_implementation_lock_sha256": file_sha256(implementation_path),
        "checks": {
            "source_failure_localization_and_binding": source_ok,
            "v55_remains_failed": relationship_ok,
            "claim_boundary": boundary_ok,
            "unchanged_planning_semantics": planning_ok,
            "planning_specific_registry": registry_ok,
            "truth_independent_population_and_streams": population_ok,
            "causally_isolated_counterfactual": counterfactual_ok,
            "exact_references_and_decision_relevance_fixtures": exact_ok,
            "noncompensatory_gates": gates_ok,
            "firewall_and_stage_authorization": firewall_ok,
            "downstream_absent": downstream_absent,
        },
        "data_access": {
            "v55r1_candidate_population_records_accessed": 0,
            "v55r1_planning_evaluation_runs": 0,
            "additional_v55_planning_evaluation_runs": 0,
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
