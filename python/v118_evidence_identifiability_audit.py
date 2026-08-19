from __future__ import annotations

import math
from typing import Any

from v116_typed_clarification_voi import (
    choice_maps, hypothesis_action_cost, prior_distribution, truth_choice,
)
from v117_causal_clarification_simulator import joint_distribution


def posterior_threshold_envelopes() -> dict[str, dict[str, float]]:
    # Exact-known versus abstain, with all noncandidate mass respectively NOVEL or ABSTAIN.
    candidate_low = 8.5 / 9.5
    candidate_high = 10.0 / 11.0
    # UNSUPPORTED versus abstain, with all alternative mass respectively NOVEL or KNOWN.
    unsupported_low = 5.0 / 7.0
    unsupported_high = 5.0 / 6.0
    return {
        "exact_candidate": {"minimum": candidate_low, "maximum": candidate_high},
        "unsupported": {"minimum": unsupported_low, "maximum": unsupported_high},
    }


def required_bayes_factor(prior: float, posterior: float) -> float:
    return (posterior / (1.0 - posterior)) / (prior / (1.0 - prior))


def prior_requirements(config: dict[str, Any], identifiers: list[str]) -> dict[str, Any]:
    envelopes = posterior_threshold_envelopes()
    output = {}
    for row in config["priorRegimes"]:
        candidate_prior = row["candidateProbability"]
        unsupported_prior = (1.0 - candidate_prior) / (len(identifiers) - 1)
        output[row["id"]] = {
            "candidate_prior": candidate_prior,
            "unsupported_prior": unsupported_prior,
            "exact_candidate_bayes_factor": {
                bound: required_bayes_factor(candidate_prior, threshold)
                for bound, threshold in envelopes["exact_candidate"].items()
            },
            "unsupported_bayes_factor": {
                bound: required_bayes_factor(unsupported_prior, threshold)
                for bound, threshold in envelopes["unsupported"].items()
            },
        }
    return output


def decisive_posterior(
    candidate: str, observation: tuple[str, str], reliability: float, correlation: float,
    candidate_prior: float, catalog: dict[str, Any], baseline_config: dict[str, Any],
    v117_config: dict[str, Any],
) -> dict[str, Any]:
    identifiers, by_id, known = choice_maps(catalog)
    prior = prior_distribution(candidate, candidate_prior, identifiers)
    weights = {
        hypothesis: prior[hypothesis] * joint_distribution(
            hypothesis, candidate, reliability, correlation, by_id, v117_config,
        )[observation]
        for hypothesis in identifiers
    }
    normalizer = sum(weights.values())
    posterior = {hypothesis: value / normalizer for hypothesis, value in weights.items()}
    actions = [("ABSTAIN", None), ("UNSUPPORTED", None)] + [
        ("KNOWN", row["intent_id"]) for row in known
    ]
    expected_costs = {
        f"{action[0]}::{action[1] or ''}": sum(
            posterior[hypothesis] * hypothesis_action_cost(
                hypothesis, action, by_id, baseline_config,
            ) for hypothesis in identifiers
        ) for action in actions
    }
    best = min(expected_costs, key=lambda action: (expected_costs[action], action))
    kinds = {
        kind: sum(value for hypothesis, value in posterior.items() if by_id[hypothesis]["kind"] == kind)
        for kind in ("KNOWN", "NOVEL", "UNSUPPORTED", "ABSTAIN")
    }
    return {
        "posterior_candidate": posterior[candidate],
        "posterior_by_kind": kinds,
        "best_action": best,
        "best_expected_cost": expected_costs[best],
        "abstain_expected_cost": expected_costs["ABSTAIN::"],
        "unsupported_expected_cost": expected_costs["UNSUPPORTED::"],
    }


def supplemental_unsupported_factor(posterior_by_kind: dict[str, float]) -> float:
    known = posterior_by_kind["KNOWN"]
    novel = posterior_by_kind["NOVEL"]
    unsupported = posterior_by_kind["UNSUPPORTED"]
    insufficient = posterior_by_kind["ABSTAIN"]
    # Multiply only the unsupported hypothesis by B and solve C(UNSUPPORTED) <= C(ABSTAIN).
    return max(1.0, (5.0 * known + 2.5 * novel + 4.0 * insufficient) / unsupported)


def run_audit(
    population: dict[str, Any], historical: dict[str, Any], catalog: dict[str, Any],
    baseline_config: dict[str, Any], v117_config: dict[str, Any], config: dict[str, Any],
    v117_result: dict[str, Any],
) -> dict[str, Any]:
    identifiers, by_id, _ = choice_maps(catalog)
    reference = config["auditPoints"]["candidateReferenceChoice"]
    reliability = config["auditPoints"]["reliability"]
    correlations = config["auditPoints"]["requiredSharedFailureCorrelations"]
    candidate_observation = tuple(config["auditPoints"]["candidateDecisiveObservation"])
    unsupported_observation = tuple(config["auditPoints"]["unsupportedDecisiveObservation"])

    known_rows = [row for row in population["selected_population"] if row["class_label"].startswith("known_")]
    correct_known_candidates = sum(
        historical["fixtures"][row["population_id"]]["candidate_choice_id"] == truth_choice(row, catalog)
        for row in known_rows
    )
    perfect_known_ceiling = correct_known_candidates / len(known_rows)

    requirements = prior_requirements(v117_config, identifiers)
    channel = {}
    _, _, known_choices = choice_maps(catalog)
    other_known = next(row["choice_id"] for row in known_choices if row["choice_id"] != reference)
    for rho in correlations:
        rkey = f"{rho:.2f}"
        reference_likelihood = joint_distribution(reference, reference, reliability, rho, by_id, v117_config)[candidate_observation]
        other_likelihood = joint_distribution(other_known, reference, reliability, rho, by_id, v117_config)[candidate_observation]
        channel[rkey] = {
            "exact_candidate_vs_other_known_bayes_factor": reference_likelihood / other_likelihood,
            "exact_candidate_information_bits": math.log2(reference_likelihood / other_likelihood),
        }

    strong_prior = next(row["candidateProbability"] for row in v117_config["priorRegimes"] if row["id"] == "strong_candidate")
    strong_unsupported = {}
    for rho in correlations:
        rkey = f"{rho:.2f}"
        posterior = decisive_posterior(
            reference, unsupported_observation, reliability, rho, strong_prior,
            catalog, baseline_config, v117_config,
        )
        posterior["minimum_additional_unsupported_specific_bayes_factor"] = supplemental_unsupported_factor(
            posterior["posterior_by_kind"]
        )
        strong_unsupported[rkey] = posterior

    unit_factor = channel["0.00"]["exact_candidate_vs_other_known_bayes_factor"]
    uniform_required_max = requirements["uniform_safe_universe"]["exact_candidate_bayes_factor"]["maximum"]
    unit_count = math.ceil(math.log(uniform_required_max) / math.log(unit_factor))
    thresholds = posterior_threshold_envelopes()
    gates = config["outcomeGates"]
    frozen_uniform = v117_result["summary"]["conditions"]["uniform_safe_universe"]["0.95"]
    frozen_strong = v117_result["summary"]["conditions"]["strong_candidate"]["0.95"]
    checks = {
        "posterior_threshold_envelopes_exact": (
            abs(thresholds["exact_candidate"]["minimum"] - gates["requiredExactCandidatePosteriorThresholdMinimum"]) <= 1e-12
            and abs(thresholds["exact_candidate"]["maximum"] - gates["requiredExactCandidatePosteriorThresholdMaximum"]) <= 1e-12
            and abs(thresholds["unsupported"]["minimum"] - gates["requiredUnsupportedPosteriorThresholdMinimum"]) <= 1e-12
            and abs(thresholds["unsupported"]["maximum"] - gates["requiredUnsupportedPosteriorThresholdMaximum"]) <= 1e-12
        ),
        "uniform_candidate_bayes_factor_requirement_exact": abs(
            requirements["uniform_safe_universe"]["exact_candidate_bayes_factor"]["minimum"]
            - gates["requiredUniformExactCandidateBayesFactorMinimum"]
        ) <= 1e-12,
        "frozen_candidate_evidence_below_uniform_requirement": max(
            item["exact_candidate_vs_other_known_bayes_factor"] for item in channel.values()
        ) <= gates["maximumFrozenCandidateBayesFactorThroughRequiredCorrelation"],
        "uniform_known_failure_reproduced": all(
            frozen_uniform[f"{rho:.2f}"]["correlation_aware"]["known_exact_probability"] == 0.0
            for rho in correlations
        ),
        "perfect_known_ceiling_exact": abs(perfect_known_ceiling - gates["requiredPerfectKnownCeiling"]) <= 1e-12,
        "strong_unsupported_action_transition_explained": (
            strong_unsupported["0.00"]["best_action"] == "UNSUPPORTED::"
            and strong_unsupported["0.25"]["best_action"] == "ABSTAIN::"
            and strong_unsupported["0.50"]["best_action"] == "ABSTAIN::"
            and frozen_strong["0.00"]["correlation_aware"]["unsupported_correct_probability"] > 0.80
            and frozen_strong["0.25"]["correlation_aware"]["unsupported_correct_probability"] == 0.0
            and frozen_strong["0.50"]["correlation_aware"]["unsupported_correct_probability"] == 0.0
        ),
        "finite_asymmetric_candidate_evidence_requirement": unit_count == gates["requiredIndependentExactConfirmationUnitCount"],
        "finite_supplemental_unsupported_evidence_requirement": max(
            row["minimum_additional_unsupported_specific_bayes_factor"]
            for row in strong_unsupported.values()
        ) <= gates["maximumSupplementalUnsupportedSpecificBayesFactorAtStrongPrior"],
        "true_hypothesis_retention": gates["requiredTrueHypothesisRetention"] == 1.0,
        "aggregate_only": gates["maximumIndividualRecordEmissionCount"] == 0,
        "zero_actual_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {
        "hypothesis_count": len(identifiers),
        "known_record_count": len(known_rows),
        "correct_known_candidate_count": correct_known_candidates,
        "perfect_known_ceiling": perfect_known_ceiling,
        "posterior_threshold_envelopes": thresholds,
        "prior_bayes_factor_requirements": requirements,
        "frozen_v117_candidate_evidence": channel,
        "strong_prior_unsupported_decisive_posterior": strong_unsupported,
        "independent_exact_confirmation_unit_bayes_factor": unit_factor,
        "minimum_independent_exact_confirmation_unit_count": unit_count,
        "outcome_gates": checks,
        "outcome_pass": passed,
        "decision": config["decisionRule"]["ifAllOutcomeAndAccessGatesPass"] if passed else config["decisionRule"]["otherwise"],
        "true_hypothesis_retention": 1.0,
        "individual_record_emission_count": 0,
        "actual_execution_count": 0,
    }


__all__ = [
    "decisive_posterior", "posterior_threshold_envelopes", "prior_requirements",
    "required_bayes_factor", "run_audit", "supplemental_unsupported_factor",
]
