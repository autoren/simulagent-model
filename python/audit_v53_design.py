#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v53-continuous-parameter-smc2.json")
    parser.add_argument("--plan", default="docs/v53-continuous-parameter-smc2-plan.md")
    parser.add_argument(
        "--output",
        default="outputs/v53-continuous-parameter-smc2/design-audit.json",
    )
    args = parser.parse_args()
    config_path, plan_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.output)
    )
    config = json.loads(config_path.read_text())
    source_path = PROJECT_ROOT / config["sourceV52r2OutcomeLock"]
    source = json.loads(source_path.read_text())
    errors = []

    source_bound = (
        source["qualification_passed"]
        and source["authorization"]["preregister_continuous_parameter_smc_squared"]
        and file_sha256(PROJECT_ROOT / source["result"]) == source["result_sha256"]
        and file_sha256(PROJECT_ROOT / source["post_result_audit"])
        == source["post_result_audit_sha256"]
    )
    if not source_bound:
        errors.append("V52r2 does not authorize or bind the V53 preregistration")

    boundary = config["claimBoundary"]
    boundary_ok = (
        boundary["continuousStochasticParameter"]
        and boundary["smcSquared"]
        and boundary["particleMcmcReference"]
        and boundary["exactProgramEnumeration"]
        and boundary["exactSmallCaseQuadratureOracle"]
        and not any(boundary[key] for key in (
            "activeInterventionSelection", "rewardOrPlanning", "languageGrounding",
            "noisySensors", "openOntology", "modelAccess", "adapterTraining",
            "finalEvaluation",
        ))
    )
    if not boundary_ok:
        errors.append("V53 claim boundary is too broad or omits required inference views")

    parameter = config["parameterModel"]
    parameter_ok = (
        parameter["support"] == [0.05, 0.95]
        and parameter["prior"] == "scaled_beta"
        and parameter["alpha"] == parameter["beta"] == 2.0
        and parameter["scope"]
        == "one_shared_branch_probability_per_program_across_all_episodes_and_ticks"
    )
    if not parameter_ok:
        errors.append("V53 continuous parameter model is not the frozen shared scaled-Beta model")

    population = config["population"]
    population_ok = (
        population["programTemplates"] == 8
        and population["templatesPerFamily"] == 2
        and len(population["families"]) == 4
        and config["exactBenchmark"]["records"] == 32
        and config["exactBenchmark"]["recordsPerTemplate"] == 4
        and config["sbc"]["replications"] == 256
        and config["pmcmcReference"]["records"] == 16
        and config["scaleStress"]["records"] == 32
        and config["scaleStress"]["recordsPerTemplate"] == 4
    )
    if not population_ok:
        errors.append("V53 population quotas are inconsistent")

    smc = config["smcSquared"]
    smc_ok = (
        smc["outerThetaParticleBudgets"] == [31, 127, 509]
        and smc["primaryOuterThetaParticleBudget"] == 509
        and smc["innerStateParticleBudget"] == 127
        and smc["independentRepeatsOnExactBenchmark"] == 3
        and smc["outerRejuvenation"] == "particle_marginal_metropolis_hastings"
        and smc["rejuvenationStepsPerOuterResampling"] == 2
        and not smc["adaptiveProposal"]
        and len(smc["randomStreamKey"]) == len(set(smc["randomStreamKey"]))
    )
    if not smc_ok:
        errors.append("V53 SMC-squared nesting, budgets, or fixed rejuvenation are invalid")

    exact_ok = (
        config["exactBenchmark"]["quadratureNodes"] == 257
        and config["exactBenchmark"]["dynamicStateReference"]
        == "exact_stepwise_world_queue_enumeration"
        and len(config["exactBenchmark"]["comparisonTargets"]) == 6
    )
    pmcmc = config["pmcmcReference"]
    pmcmc_ok = (
        pmcmc["algorithm"] == "particle_marginal_metropolis_hastings"
        and pmcmc["role"]
        == "offline_conditional_theta_reference_not_program_evidence_estimator"
        and pmcmc["conditionedProgram"] == "generating_program"
        and pmcmc["chains"] == 4
        and pmcmc["warmupIterationsPerChain"] == 1000
        and pmcmc["retainedIterationsPerChain"] == 2000
        and pmcmc["innerStateParticleBudget"] == 509
        and not pmcmc["adaptiveProposal"]
    )
    if not exact_ok or not pmcmc_ok:
        errors.append("V53 exact oracle or PMCMC reference is underspecified")

    sbc = config["sbc"]
    sbc_ok = (
        sbc["rankSupportSize"] == sbc["posteriorDrawsPerReplication"] + 1
        and sbc["replications"] / sbc["rankBins"] == sbc["expectedRanksPerBin"]
        and len(sbc["testQuantities"]) == 5
        and not sbc["selectionOnSbcReplications"]
    )
    controls_ok = (
        config["controls"]["minimumControlsDetectedOrDominated"]
        == config["gates"]["minimumControlsDetectedOrDominated"] == 4
        and len([key for key in config["controls"] if key.endswith("Control")]) == 5
    )
    if not sbc_ok or not controls_ok:
        errors.append("V53 SBC or negative controls are inconsistent")

    gates = config["gates"]
    gate_ok = (
        gates["minimumNormalizationRate"] == 1.0
        and gates["minimumScaleStressNormalizationRate"] == 1.0
        and gates["maximumUnintendedStreamCollisions"] == 0
        and gates["minimumRankChiSquarePValue"] == 0.001
        and gates["maximumPmcmcSplitRhat"] == 1.05
        and gates["minimumPmcmcBulkEss"] == 200
        and gates["maximumPmcmcThetaWasserstein"] == 0.04
    )
    firewall_ok = (
        set(config["firewall"].values())
        <= {"forbidden", "forbidden_while_exact_likelihood_is_available"}
        and config["stageAuthorization"] == {
            "writeAndAuditImplementation": True,
            "constructSealedPopulations": False,
            "runSmcSquaredEvaluation": False,
            "activeInterventionSelection": False,
            "modelAccess": False,
        }
    )
    if not gate_ok or not firewall_ok:
        errors.append("V53 gates, firewall, or stage authorization are invalid")

    seed_values = [
        value for key, value in population.items()
        if key.endswith("Seed") and isinstance(value, int)
    ]
    seeds_ok = len(seed_values) == len(set(seed_values)) == 10
    if not seeds_ok:
        errors.append("V53 random-stream root seeds are not distinct")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v53-design-lock.json",
            "configs/v53-implementation-lock.json",
            "configs/v53-population-seal.json",
            "configs/v53-outcome-lock.json",
            "data/v53-continuous-parameter-smc2",
            "outputs/v53-continuous-parameter-smc2/evaluation-attempt.json",
            "outputs/v53-continuous-parameter-smc2/evaluation",
        )
    )
    if not downstream_absent:
        errors.append("V53 downstream artifact exists before design lock")

    audit = {
        "schema_version": 53,
        "experiment": "v53_design_audit",
        "passed": not errors,
        "decision": "authorize_v53_design_lock" if not errors else "repair_v53_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "checks": {
            "source_v52r2_authorization_and_binding": source_bound,
            "claim_boundary": boundary_ok,
            "continuous_parameter_model": parameter_ok,
            "population_quotas": population_ok,
            "smc_squared_specification": smc_ok,
            "exact_quadrature_oracle": exact_ok,
            "pmcmc_reference": pmcmc_ok,
            "sbc_specification": sbc_ok,
            "controls": controls_ok,
            "noncompensatory_gates": gate_ok,
            "firewall_and_stage_authorization": firewall_ok,
            "distinct_root_seeds": seeds_ok,
            "downstream_absent": downstream_absent,
        },
        "data_access": {
            "v53_population_records_accessed": 0,
            "smc_squared_evaluation_runs": 0,
            "pmcmc_reference_runs": 0,
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
