#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v49-passive-partial-observation.json")
    parser.add_argument("--output", default="outputs/v49-passive-partial-observation/design-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text())
    errors: list[str] = []

    source_path = PROJECT_ROOT / config["sourceV48OutcomeLock"]
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    if not source.get("qualification_passed") or not source.get("authorization", {}).get(
        "preregister_passive_partial_observation"
    ):
        errors.append("V48 does not authorize V49 preregistration")

    boundary = config["claimBoundary"]
    required_boundary = (
        boundary["partialObservation"]
        and boundary["persistentLatentState"]
        and boundary["unknownProgramAndProbability"]
        and boundary["knownObservationProcess"]
        and not boundary["noisySensors"]
        and not boundary["languageGrounding"]
        and not boundary["activeInterventionSelection"]
        and not boundary["continuousProbabilityLearning"]
        and not boundary["openOntology"]
        and not boundary["modelAccess"]
        and not boundary["adapterTraining"]
        and not boundary["finalEvaluation"]
    )
    if not required_boundary:
        errors.append("V49 does not isolate passive partial observation")

    population = config["population"]
    if (
        population["mechanics"] != 48
        or population["mechanicsPerFamily"] * len(population["families"]) != 48
        or population["developmentFitMechanics"] + population["developmentEvaluationMechanics"] != 48
        or sum(population["probabilityCounts"].values()) != 48
        or len(set(population["probabilityCounts"].values())) != 1
    ):
        errors.append("V49 population quotas are inconsistent")
    if population["visibleFractions"] != [0.25, 0.5, 0.75]:
        errors.append("V49 visibility strata changed")
    if max(population["queryEvidencePrefixLengths"]) >= min(population["sequenceLengths"]):
        errors.append("V49 query prefix may consume the whole shortest sequence")

    observation = config["observationContract"]
    if not (
        observation["maskKnownToAgent"]
        and observation["maskSampledIndependentlyOfProgramAndRealizedValues"]
        and observation["sensorNoise"] == "none"
        and observation["queryOutcomesScorerOnly"]
        and observation["delayedEventQueueIsLatentState"]
        and observation["passiveFixedInterventions"]
    ):
        errors.append("V49 observation contract is not passive, known, and noiseless")

    inference = config["inference"]
    if (
        not inference["primary"].startswith("exact_forward_sum_product")
        or inference["perEpisodeOracleChoices"]
        or inference["selectionOnDevelopmentEvaluation"]
    ):
        errors.append("V49 inference is not fixed exact marginalization")

    expected_conditions = {
        "partiallyObservedJointInference",
        "matchedFullyObservedInference",
        "oracleProgramPartialObservation",
        "mapLatentStateCollapse",
        "observationHistoryAblation",
        "uniformizedOutcomeMass",
        "literalMaskedTraceLookup",
    }
    if set(config["conditions"]) != expected_conditions:
        errors.append("V49 comparisons or controls are incomplete")

    requirements = config["constructionRequirements"]
    if (
        requirements["minimumFractionQueriesWithNondegenerateOracleLatentBelief"] < 0.75
        or requirements["minimumFractionQueriesWhereHiddenStateCanAffectScoredSuffix"] < 0.75
        or requirements["supportQueryStructuralOverlap"] != 0
        or not requirements["sameLatentDrawsAcrossPartialAndFullConditions"]
    ):
        errors.append("V49 construction does not guarantee a genuine paired hidden-state test")

    gates = config["gates"]
    if any(
        gates[key] != 1.0
        for key in (
            "minimumLikelihoodNormalization",
            "minimumBeliefNormalization",
            "minimumPredictiveNormalization",
            "minimumFiniteLogLossRate",
            "minimumTargetLatentContinuationRetention",
        )
    ):
        errors.append("V49 validity gates are not exact")
    if gates["maximumOracleProgramPartialMeanTv"] > 1e-10:
        errors.append("V49 oracle filter tolerance is too broad")
    if gates["minimumMapLatentCollapseLogLossDisadvantageNats"] <= 0 or gates[
        "minimumObservationHistoryAblationLogLossDisadvantageNats"
    ] <= 0:
        errors.append("V49 does not require evidence of persistent belief use")

    downstream = (
        "configs/v49-design-lock.json",
        "configs/v49-implementation-lock.json",
        "data/v49-passive-partial-observation",
        "outputs/v49-passive-partial-observation/development",
    )
    if any((PROJECT_ROOT / path).exists() for path in downstream):
        errors.append("V49 downstream artifact exists before design lock")

    audit = {
        "schema_version": 49,
        "experiment": "v49_design_audit",
        "passed": not errors,
        "decision": "authorize_v49_design_lock" if not errors else "repair_v49_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "source_v48_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v48_outcome_lock_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "checks": {
            "source_authorized": source.get("authorization", {}).get("preregister_passive_partial_observation", False),
            "fresh_balanced_population_required": population["programOverlapWithV46V47OrV48"] == 0
            and population["caseOverlapWithV47OrV48"] == 0
            and len(set(population["probabilityCounts"].values())) == 1,
            "known_value_independent_masks": observation["maskKnownToAgent"]
            and observation["maskSampledIndependentlyOfProgramAndRealizedValues"],
            "latent_queue_included": observation["delayedEventQueueIsLatentState"],
            "exact_marginalization": inference["primary"].startswith("exact_forward_sum_product"),
            "matched_full_observation": config["conditions"]["matchedFullyObservedInference"].startswith("same_programs"),
            "belief_controls_registered": "mapLatentStateCollapse" in config["conditions"]
            and "observationHistoryAblation" in config["conditions"],
            "no_language_active_selection_or_training": not boundary["languageGrounding"]
            and not boundary["activeInterventionSelection"]
            and not boundary["modelAccess"]
            and not boundary["adapterTraining"],
            "non_final": config["firewall"]["finalEvaluation"] == "forbidden",
        },
        "data_access": {
            "partial_observation_mechanics_constructed": 0,
            "sampled_realizations": 0,
            "development_runs": 0,
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
