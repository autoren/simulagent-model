#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v54-exact-one-step-eig.json")
    parser.add_argument("--plan", default="docs/v54-exact-one-step-eig-plan.md")
    parser.add_argument(
        "--output", default="outputs/v54-exact-one-step-eig/design-audit.json"
    )
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.output)
    )
    config = json.loads(config_path.read_text())
    source_path = (PROJECT_ROOT / config["sourceV53r2OutcomeLock"]).resolve()
    source = json.loads(source_path.read_text())
    errors = []

    source_bound = (
        source["qualification_passed"]
        and source["authorization"]["preregister_exact_one_step_expected_information_gain"]
        and not source["authorization"]["construct_active_population"]
        and not source["authorization"]["reward_or_planning"]
        and file_sha256(PROJECT_ROOT / source["result"]) == source["result_sha256"]
        and file_sha256(PROJECT_ROOT / source["post_result_audit"])
        == source["post_result_audit_sha256"]
        and file_sha256(PROJECT_ROOT / source["summary"]) == source["summary_sha256"]
    )
    if not source_bound:
        errors.append("V53r2 does not authorize or bind the V54 preregistration")

    boundary = config["claimBoundary"]
    boundary_ok = (
        boundary["activeInterventionSelection"]
        and boundary["oneStepOpenLoopAssaySelection"]
        and boundary["exactHistoryDependentJointBelief"]
        and boundary["exactProgramEnumeration"]
        and boundary["exactContinuousParameterQuadrature"]
        and boundary["exactHiddenConfigurationMarginalization"]
        and boundary["targetLatent"] == "joint_program_identity_and_continuous_theta"
        and boundary["hiddenConfigurationRole"]
        == "nuisance_integrated_out_not_an_information_target"
        and not any(boundary[key] for key in (
            "rewardOrPlanning", "multiStepContingentDesign", "learnedAcquisition",
            "approximateParticleAcquisition", "languageGrounding", "noisySensors",
            "openOntology", "modelAccess", "adapterTraining", "verification",
            "finalEvaluation",
        ))
    )
    if not boundary_ok:
        errors.append("V54 claim boundary is too broad or omits exact active design")

    target = config["targetAndObjective"]
    target_ok = (
        target["primaryTarget"] == ["program_identity", "continuous_theta"]
        and target["nuisanceLatent"] == "current_world_queue_configuration"
        and target["logarithmBase"] == "natural"
        and target["selection"] == "argmax_over_complete_finite_candidate_set"
        and target["forbidOutcomeConditionedSelection"]
        and target["tieToleranceNats"] == 1e-12
    )
    if not target_ok:
        errors.append("V54 target, EIG objective, or tie policy is invalid")

    oracle = config["exactOracle"]
    oracle_ok = (
        oracle["programTemplates"] == 8
        and oracle["quadratureNodes"] == 257
        and oracle["dynamicStateReference"]
        == "exact_stepwise_world_queue_history_enumeration"
        and oracle["primaryImplementation"] != oracle["independentReference"]
        and len(oracle["analyticFixtures"]) == 4
    )
    if not oracle_ok:
        errors.append("V54 exact oracle or independent reference is underspecified")

    interventions = config["interventionSet"]
    interventions_ok = (
        interventions["actionIds"] == ["pulse", "route", "wait"]
        and interventions["bindingRule"]
        == "all_ordered_distinct_actor_target_pairs_for_non_wait_actions"
        and interventions["assayTicks"] == 3
        and not interventions["withinAssayAdaptation"]
        and interventions["observationPanel"]
        == "full_world_atom_universe_at_every_assay_tick"
        and interventions["candidateCountsByEntityCount"] == {"2": 5, "3": 13}
        and set(interventions["equalEvidenceCost"].values()) == {1, 3}
    )
    if not interventions_ok:
        errors.append("V54 assay is not complete, equal-cost, or non-contingent")

    population = config["population"]
    class_counts = population["historyClasses"]
    population_ok = (
        population["selectionRecords"] == 64
        and population["recordsPerGeneratingTemplate"] == 8
        and sum(class_counts.values()) == 64
        and set(class_counts) == {
            "prior_like_all_wait", "mixed_informative", "pending_delayed_event"
        }
        and population["reuseParameterizedTemplatesToIsolateAcquisition"]
        and len(population["freshPublicHistoriesAgainst"]) == 5
        and len(population["freshObservationDesignsAgainst"]) == 5
    )
    if not population_ok:
        errors.append("V54 selection population quotas or freshness are inconsistent")

    sbc = config["adaptiveSbc"]
    sbc_ok = (
        sbc["replications"] == 256
        and sbc["rankSupportSize"] == sbc["posteriorDrawsPerReplication"] + 1
        and sbc["replications"] / sbc["rankBins"] == sbc["expectedRanksPerBin"]
        and len(sbc["testQuantities"]) == 3
        and not sbc["selectionOnSbcOutcomes"]
        and not sbc["reuseSelectionPopulation"]
    )
    if not sbc_ok:
        errors.append("V54 adaptive SBC design is inconsistent")

    controls = config["controls"]
    control_keys = [key for key in controls if key.endswith("Control")]
    controls_ok = (
        len(control_keys) == 7
        and controls["minimumControlsDetectedOrDominated"]
        == config["gates"]["minimumControlsDetectedOrDominated"] == 5
    )
    if not controls_ok:
        errors.append("V54 controls are missing or inconsistent")

    gates = config["gates"]
    gates_ok = (
        gates["minimumCompletedSelectionFraction"] == 1.0
        and gates["minimumCandidateAndPredictiveNormalizationRate"] == 1.0
        and gates["minimumOptimalSetMembershipRate"] == 1.0
        and gates["maximumSelectedEigRegret"] == 1e-10
        and gates["minimumInformativeRecordFraction"] == 0.25
        and gates["minimumMeanOracleMinusUniformRandomEigNats"] == 0.001
        and gates["minimumPostSelectionNormalizationRate"] == 1.0
        and gates["minimumRankChiSquarePValue"] == 0.001
        and all(gates[key] == 0 for key in (
            "maximumTruthFieldAccessCount",
            "maximumRealizedOutcomeAccessBeforeSelectionCount",
            "maximumCandidateOmissionCount",
            "maximumCanonicalTieBreakViolationCount",
            "maximumHistoryAndOutcomeStreamCollisionCount",
        ))
    )
    if not gates_ok:
        errors.append("V54 non-compensatory correctness, efficiency, or integrity gates are invalid")

    firewall = config["firewall"]
    forbidden = {"forbidden", "forbidden_while_exact_likelihood_is_available"}
    firewall_ok = (
        set(firewall.values()) <= forbidden
        and config["stageAuthorization"] == {
            "writeAndAuditExactEigImplementation": True,
            "constructActivePopulations": False,
            "runActiveEvaluation": False,
            "rewardOrPlanning": False,
            "languageGrounding": False,
            "modelAccess": False,
        }
    )
    if not firewall_ok:
        errors.append("V54 firewall or stage authorization is invalid")

    seed_values = [
        value for key, value in population.items()
        if key.endswith("Seed") and isinstance(value, int)
    ]
    seeds_ok = len(seed_values) == len(set(seed_values)) == 10
    if not seeds_ok:
        errors.append("V54 random-stream root seeds are not distinct")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v54-design-lock.json",
            "configs/v54-implementation-lock.json",
            "configs/v54-population-seal.json",
            "configs/v54-outcome-lock.json",
            "data/v54-exact-one-step-eig",
            "outputs/v54-exact-one-step-eig/implementation-audit.json",
            "outputs/v54-exact-one-step-eig/evaluation-attempt.json",
            "outputs/v54-exact-one-step-eig/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V54 downstream artifact exists before design lock")

    audit = {
        "schema_version": 54,
        "experiment": "v54_design_audit",
        "passed": not errors,
        "decision": "authorize_v54_design_lock" if not errors else "repair_v54_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "checks": {
            "source_v53r2_authorization_and_binding": source_bound,
            "claim_boundary": boundary_ok,
            "static_latent_target_and_eig_objective": target_ok,
            "exact_oracle_and_independent_reference": oracle_ok,
            "complete_equal_cost_noncontingent_assays": interventions_ok,
            "selection_population_quotas_and_freshness": population_ok,
            "adaptive_sbc": sbc_ok,
            "controls": controls_ok,
            "noncompensatory_gates": gates_ok,
            "firewall_and_stage_authorization": firewall_ok,
            "distinct_root_seeds": seeds_ok,
            "downstream_absent": downstream_absent,
        },
        "data_access": {
            "v54_candidate_population_records_accessed": 0,
            "v54_active_evaluation_runs": 0,
            "v54_adaptive_sbc_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
