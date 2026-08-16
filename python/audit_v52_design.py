#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/v52-rao-blackwellized-particle-filtering.json"
    )
    parser.add_argument(
        "--output", default="outputs/v52-rao-blackwellized-particle-filtering/design-audit.json"
    )
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text())
    errors = []
    source_path = PROJECT_ROOT / config["sourceV51r1OutcomeLock"]
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    metrics = source.get("metrics", {})
    primary = metrics.get("primary_sbc", {})
    interpretation = config["v51r1Interpretation"]
    source_matches = (
        source.get("qualification_passed") is True
        and metrics.get("replications") == interpretation["replications"]
        and metrics.get("normalization_rate") == interpretation["normalizationRate"]
        and metrics.get("maximum_exact_path_tv") == interpretation["maximumExactPathTv"]
        and primary.get("minimum_chi_square_p_value")
        == interpretation["minimumPrimaryRankChiSquarePValue"]
        and primary.get("maximum_absolute_rank_bin_z")
        == interpretation["maximumPrimaryRankBinZ"]
        and primary.get("maximum_absolute_coverage_z")
        == interpretation["maximumPrimaryCoverageZ"]
        and metrics.get("bug_controls_rejected") == interpretation["bugControlsRejected"]
        and source.get("authorization", {}).get("preregister_scalable_particle_inference") is True
    )
    if not source_matches:
        errors.append("V52 interpretation is not bound to sealed V51r1 outcome")

    boundary = config["claimBoundary"]
    boundary_ok = (
        boundary["approximateParticleInference"]
        and boundary["raoBlackwellizedStaticLatents"]
        and boundary["exactProgramAndProbabilityEnumeration"]
        and boundary["particleHiddenState"]
        and boundary["exactOracleComparison"]
        and boundary["simulationBasedCalibration"]
        and boundary["scaleStress"]
        and not boundary["smcSquared"]
        and not boundary["particleMcmc"]
        and not boundary["continuousProbabilities"]
        and not boundary["activeInterventionSelection"]
        and not boundary["rewardOrPlanning"]
        and not boundary["languageGrounding"]
        and not boundary["modelAccess"]
        and not boundary["adapterTraining"]
        and not boundary["finalEvaluation"]
    )
    if not boundary_ok:
        errors.append("V52 does not isolate dynamic-state particle approximation")

    algorithm = config["algorithm"]
    algorithm_ok = (
        "enumerate_all_48" in algorithm["outerStaticLayer"]
        and "one_step_stochastic_branches_exactly" in algorithm["raoBlackwellization"]
        and algorithm["resamplingEssThresholdFraction"] == 0.5
        and algorithm["forcedResamplingWhenWeightedSupportExceedsBudget"]
        and algorithm["duplicateConfigurationMerging"]
        and not algorithm["rejuvenation"]
        and not algorithm["adaptiveParticleCount"]
    )
    if not algorithm_ok:
        errors.append("V52 particle algorithm boundary is inconsistent")

    population = config["population"]
    exact = config["exactBenchmark"]
    budgets = config["particleBudgets"]
    sbc = config["sbc"]
    scale = config["scaleStress"]
    populations_ok = (
        population["mechanics"]
        == len(population["families"])
        * len(population["probabilityVocabulary"])
        * population["mechanicsPerFamilyProbabilityCell"]
        and population["observationDesignIdentity"]
        == ["entities", "initial_state", "actions", "masks"]
        and exact["records"] == population["mechanics"] * exact["recordsPerMechanic"]
        and scale["records"] == population["mechanics"] * scale["recordsPerMechanic"]
        and budgets["primaryBudget"] == max(budgets["budgets"])
        and budgets["independentRepeatsOnExactBenchmark"] >= 3
        and not budgets["selectionOnSealedResults"]
        and sbc["posteriorDrawsPerReplication"] + 1 == sbc["rankSupportSize"]
        and sbc["rankSupportSize"] % sbc["rankBins"] == 0
        and sbc["expectedRanksPerBin"] == sbc["replications"] // sbc["rankBins"]
        and sbc["expectedRanksPerBin"] >= 5
        and not sbc["selectionOnSbcReplications"]
        and not scale["exactEnumerationRequired"]
    )
    if not populations_ok:
        errors.append("V52 populations, budgets, or rank design are inconsistent")

    controls = config["collapseAndCorrelationControls"]
    controls_ok = (
        len(controls["independentStreamKey"]) == 7
        and controls["minimumControlsDetectedOrDominated"] >= 3
        and "map" in controls["mapProgramControl"]
        and "map" in controls["mapConfigurationControl"]
        and "square" in controls["likelihoodSquaredControl"]
        and "detect" in controls["streamCollisionControl"]
    )
    if not controls_ok:
        errors.append("V52 collapse or correlation controls are insufficient")

    gates = config["gates"]
    gates_ok = (
        gates["maximumPrimaryMeanSupportProgramTv"] <= 0.02
        and gates["maximumPrimaryMeanQueryProgramTv"] <= 0.025
        and gates["maximumPrimaryMeanJointBeliefTv"] <= 0.04
        and gates["maximumPrimaryMeanSuffixPredictiveTv"] <= 0.035
        and gates["maximumTargetProgramExtinctionRate"] <= 0.01
        and gates["maximumFalseStaticCollapseRate"] <= 0.02
        and gates["maximumFalseConfigurationLossRate"] <= 0.02
        and gates["minimumAmbiguousProgramEntropyRatio"] >= 0.90
        and gates["maximumUnintendedStreamCollisions"] == 0
        and gates["minimumRankChiSquarePValue"] >= 0.001
        and gates["maximumAbsoluteRankBinZ"] <= 4.75
        and gates["maximumAbsoluteCoverageZ"] <= 4.75
        and gates["minimumControlsDetectedOrDominated"] >= 3
        and gates["minimumScaleStressCompletionFraction"] == 1.0
        and gates["minimumScaleStressNormalizationRate"] == 1.0
    )
    if not gates_ok:
        errors.append("V52 approximation, calibration, or integrity gates are too weak")

    firewall = config["firewall"]
    firewall_ok = all(value == "forbidden" for value in firewall.values())
    if not firewall_ok:
        errors.append("V52 firewall permits a downstream capability")
    downstream = (
        "configs/v52-design-lock.json",
        "configs/v52-implementation-lock.json",
        "data/v52-rao-blackwellized-particle-filtering",
        "outputs/v52-rao-blackwellized-particle-filtering/evaluation",
    )
    downstream_absent = not any((PROJECT_ROOT / path).exists() for path in downstream)
    if not downstream_absent:
        errors.append("V52 downstream artifact exists before design lock")

    audit = {
        "schema_version": 52,
        "experiment": "v52_design_audit",
        "passed": not errors,
        "decision": "authorize_v52_design_lock" if not errors else "repair_v52_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "source_v51r1_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v51r1_outcome_lock_sha256": (
            file_sha256(source_path) if source_path.is_file() else None
        ),
        "checks": {
            "v51r1_result_bound": source_matches,
            "dynamic_particle_approximation_isolated": boundary_ok,
            "rao_blackwellized_algorithm_frozen": algorithm_ok,
            "three_populations_and_budgets_consistent": populations_ok,
            "collapse_and_correlation_controls_frozen": controls_ok,
            "noncompensatory_gates_strict": gates_ok,
            "downstream_capabilities_forbidden": firewall_ok,
            "downstream_absent": downstream_absent,
        },
        "data_access": {
            "particle_populations_constructed": 0,
            "particle_evaluation_runs": 0,
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
