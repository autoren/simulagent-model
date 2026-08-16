#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def _bound_outcome(path, pass_field):
    lock = json.loads(path.read_text())
    return (
        bool(lock[pass_field])
        and file_sha256(PROJECT_ROOT / lock["result"]) == lock["result_sha256"]
        and file_sha256(PROJECT_ROOT / lock["post_result_audit"])
        == lock["post_result_audit_sha256"]
        and file_sha256(PROJECT_ROOT / lock["summary"])
        == lock["summary_sha256"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v59-budgeted-root-sampled-planning.json"
    )
    parser.add_argument(
        "--plan", default="docs/v59-budgeted-root-sampled-planning-plan.md"
    )
    parser.add_argument(
        "--deferral", default="docs/v58-deferred-status.md"
    )
    parser.add_argument(
        "--output",
        default="outputs/v59-budgeted-root-sampled-planning/design-audit.json",
    )
    args = parser.parse_args()
    config_path, plan_path, deferral_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.deferral, args.output)
    )
    config = json.loads(config_path.read_text())
    source_specs = (
        (config["sourceV53r2OutcomeLock"], "qualification_passed"),
        (config["sourceV55r1OutcomeLock"], "combined_planning_layer_qualification_passed"),
        (config["sourceV56OutcomeLock"], "qualification_passed"),
    )
    source_paths = [PROJECT_ROOT / path for path, _ in source_specs]
    errors: list[str] = []

    sources_ok = all(
        _bound_outcome(PROJECT_ROOT / path, field)
        for path, field in source_specs
    )
    if not sources_ok:
        errors.append("V53r2, V55r1, or V56 frozen source outcome is not intact")

    deferral_text = deferral_path.read_text()
    v58_ok = (
        "deferred without release or collection" in deferral_text
        and "never be placed in V58 human-submission locations" in deferral_text
        and not (PROJECT_ROOT / "configs/v58-pilot-release-lock.json").exists()
        and not (PROJECT_ROOT / "data/v58-human-authored-known-ontology-language/pilot-submissions").exists()
        and not (PROJECT_ROOT / "data/v58-human-authored-known-ontology-language/evaluation-submissions").exists()
    )
    if not v58_ok:
        errors.append("V58 is not cleanly deferred at its frozen no-human-data boundary")

    boundary = config["claimBoundary"]
    boundary_ok = (
        all(boundary[key] for key in (
            "exactFrozenJointBeliefInput", "rootSampledStaticAndDynamicLatents",
            "observationContingentUCT", "boundedSimulationBudgets",
            "exactReferenceAtHorizonThree", "longerHorizonScalingAtFiveAndSeven",
            "posteriorExpectedMonteCarloPolicyReturn",
            "equalBudgetObservationBlindControl",
        ))
        and not any(boundary[key] for key in (
            "exactLongHorizonOptimality", "learnedValueFunction",
            "approximateBeliefInference", "formalPolicyVerification",
            "worstCaseSafety", "unboundedPlanning", "humanAuthoredLanguage",
            "languageGrounding", "modelAccess", "adapterTraining",
        ))
    )
    if not boundary_ok:
        errors.append("V59 bounded approximate-search claim boundary is invalid")

    model = config["planningModel"]
    model_ok = (
        model["horizons"] == [3, 5, 7]
        and model["candidateCount"] == 5
        and model["entityCount"] == 2
        and model["actionCost"] == {"pulse": 0.01, "route": 0.01, "wait": 0.0}
        and model["discountFactor"] == 1.0
    )
    if not model_ok:
        errors.append("V59 frozen planning model or horizon ladder is invalid")

    search = config["candidateSearch"]
    search_ok = (
        search["algorithm"] == "root_sampled_history_tree_uct"
        and search["searchBudgets"] == [64, 256, 1024]
        and search["replicatesPerTaskBudget"] == 3
        and abs(search["explorationConstant"] ** 2 - 2.0) <= 1e-12
        and not search["observationProgressiveWidening"]
        and not search["actionPruning"]
        and not search["truthAccess"]
        and not search["futureObservationAccess"]
        and "independent_of_the_sampled_latent_identity" in search["rolloutPolicy"]
    )
    if not search_ok:
        errors.append("V59 root-sampled UCT algorithm or budget ladder is invalid")

    controls = config["controls"]
    controls_ok = (
        len(controls) == 6
        and "identical_root_sampling_budget" in controls["observationBlindUCT"]
        and set(controls) == {
            "observationBlindUCT", "nonpersistentStaticLatentMutant",
            "latentConditionedRolloutMutant", "observationPermutationMutant",
            "actionCostOmissionMutant", "budgetOffByOneMutant",
        }
    )
    if not controls_ok:
        errors.append("V59 equal-budget control or implementation mutants are incomplete")

    population = config["population"]
    population_ok = (
        population["tasks"] == 24
        and population["tasksPerHorizon"] == {"3": 8, "5": 8, "7": 8}
        and population["tasksPerGeneratingTemplate"] == 3
        and sum(population["historyClasses"].values()) == 24
        and population["publicAndAuditFilesSeparatedBeforeCandidateAccess"]
        and population["supportEpisodesPerTask"] == 4
    )
    if not population_ok:
        errors.append("V59 population census or public/audit split is invalid")

    evaluation = config["evaluation"]
    evaluation_ok = (
        evaluation["posteriorEpisodesPerPolicy"] == 2048
        and evaluation["commonRandomNumbersWithinCandidateControlPair"]
        and evaluation["exactReferenceHorizon"] == 3
        and evaluation["evaluationRuns"] == 1
    )
    if not evaluation_ok:
        errors.append("V59 exact-reference or independent policy evaluation is invalid")

    gates = config["gates"]
    gates_ok = (
        len(gates) == 17
        and gates["minimumCompletedTaskBudgetReplicateFraction"] == 1.0
        and gates["minimumExactReferenceHighBudgetRootOptimalSetMembershipRate"] == 0.75
        and gates["maximumExactReferenceHighBudgetMeanRootRegret"] == 0.02
        and gates["minimumHighMinusLowBudgetCandidateReturn"] == -0.005
        and gates["minimumScaleTaskPositiveObservationContingencyFraction"] == 0.25
        and gates["maximumAnalyticRootSampleStaticTotalVariation"] == 0.03
        and gates["minimumSimulationBudgetAccountingRate"] == 1.0
        and gates["minimumDeterministicReplayRate"] == 1.0
        and gates["minimumImplementationMutantKillRate"] == 1.0
        and all(gates[key] == 0 for key in (
            "maximumTruthFieldAccessCount", "maximumFutureObservationAccessCount",
            "maximumLatentConditionedRolloutAccessCount",
            "maximumUnexpectedEvaluationAttemptCount",
        ))
    )
    if not gates_ok:
        errors.append("V59 calibration, scale, control, or integrity gates are invalid")

    stage = config["stageAuthorization"]
    firewall_ok = (
        set(config["firewall"].values()) == {"forbidden"}
        and stage == {
            "auditAndFreezeDesign": True,
            "writeAndAuditCandidateSearch": False,
            "constructPopulation": False,
            "runEvaluation": False,
            "simulateHumanV58Records": False,
            "modelAccess": False,
        }
    )
    if not firewall_ok:
        errors.append("V59 firewall or initial stage authorization is invalid")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v59-design-lock.json",
            "configs/v59-implementation-lock.json",
            "configs/v59-population-seal.json",
            "configs/v59-evaluation-implementation-lock.json",
            "configs/v59-outcome-lock.json",
            "data/v59-budgeted-root-sampled-planning",
            "outputs/v59-budgeted-root-sampled-planning/implementation-audit.json",
            "outputs/v59-budgeted-root-sampled-planning/evaluation-attempt.json",
            "outputs/v59-budgeted-root-sampled-planning/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V59 downstream artifact exists before design lock")

    checks = {
        "frozen_source_outcomes_bound": sources_ok,
        "v58_deferred_without_fabricated_human_data": v58_ok,
        "bounded_approximate_search_claim_boundary": boundary_ok,
        "frozen_planning_model_and_horizon_ladder": model_ok,
        "root_sampled_uct_and_budget_ladder": search_ok,
        "equal_budget_control_and_mutants": controls_ok,
        "population_census_and_public_audit_split": population_ok,
        "exact_reference_and_independent_policy_evaluation": evaluation_ok,
        "noncompensatory_gates": gates_ok,
        "firewall_and_stage_authorization": firewall_ok,
        "downstream_absent": downstream_absent,
    }
    audit = {
        "schema_version": 59,
        "experiment": "v59_design_audit",
        "passed": not errors,
        "decision": "authorize_v59_design_lock" if not errors else "repair_v59_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "v58_deferral": str(deferral_path.relative_to(PROJECT_ROOT)),
        "v58_deferral_sha256": file_sha256(deferral_path),
        "source_outcome_locks_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in source_paths
        },
        "checks": checks,
        "data_access": {
            "v59_population_records_accessed": 0,
            "v59_candidate_search_runs": 0,
            "v59_evaluation_runs": 0,
            "human_authored_records_collected": 0,
            "model_forward_passes": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

