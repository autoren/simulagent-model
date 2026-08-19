from __future__ import annotations

import itertools
from typing import Any

from v116_typed_clarification_voi import (
    choice_maps, hypothesis_action_cost, prior_distribution, truth_choice,
)


A_ANSWERS = ("CONFIRM", "REJECT", "UNSURE")
B_ANSWERS = ("DECLARED", "UNDECLARED_VISIBLE", "OUTSIDE_VISIBLE", "UNSURE")


def true_observations(hypothesis: str, candidate: str, by_id: dict[str, dict[str, Any]]) -> tuple[str, str]:
    kind = by_id[hypothesis]["kind"]
    first = "UNSURE" if kind == "ABSTAIN" else "CONFIRM" if kind == "KNOWN" and hypothesis == candidate else "REJECT"
    second = {
        "KNOWN": "DECLARED", "NOVEL": "UNDECLARED_VISIBLE",
        "UNSUPPORTED": "OUTSIDE_VISIBLE", "ABSTAIN": "UNSURE",
    }[kind]
    return first, second


def error_distribution(truth: str, answers: tuple[str, ...], unsure: str, unsure_fraction: float) -> dict[str, float]:
    output = {answer: 0.0 for answer in answers}
    if truth != unsure:
        output[unsure] = unsure_fraction
        actionable = [answer for answer in answers if answer not in {truth, unsure}]
        for answer in actionable:
            output[answer] = (1.0 - unsure_fraction) / len(actionable)
    else:
        actionable = [answer for answer in answers if answer != unsure]
        for answer in actionable:
            output[answer] = 1.0 / len(actionable)
    return output


def marginal_distribution(truth: str, answers: tuple[str, ...], unsure: str, reliability: float, unsure_fraction: float) -> dict[str, float]:
    errors = error_distribution(truth, answers, unsure, unsure_fraction)
    return {answer: reliability * (answer == truth) + (1.0 - reliability) * errors[answer] for answer in answers}


def joint_distribution(
    hypothesis: str, candidate: str, reliability: float, correlation: float,
    by_id: dict[str, dict[str, Any]], config: dict[str, Any],
) -> dict[tuple[str, str], float]:
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must be bounded")
    truth_a, truth_b = true_observations(hypothesis, candidate, by_id)
    fraction = config["channel"]["nonCorrectMassToUnsure"]
    errors_a = error_distribution(truth_a, A_ANSWERS, "UNSURE", fraction)
    errors_b = error_distribution(truth_b, B_ANSWERS, "UNSURE", fraction)
    marginal_a = marginal_distribution(truth_a, A_ANSWERS, "UNSURE", reliability, fraction)
    marginal_b = marginal_distribution(truth_b, B_ANSWERS, "UNSURE", reliability, fraction)
    output = {}
    for first, second in itertools.product(A_ANSWERS, B_ANSWERS):
        independent = marginal_a[first] * marginal_b[second]
        shared = (
            reliability * (first == truth_a and second == truth_b)
            + (1.0 - reliability) * errors_a[first] * errors_b[second]
        )
        output[(first, second)] = (1.0 - correlation) * independent + correlation * shared
    if abs(sum(output.values()) - 1.0) > 1e-12:
        raise AssertionError("joint observation channel is not normalized")
    return output


def bayes_policy(
    candidate: str, reliability: float, assumed_correlation: float, prior_probability: float,
    catalog: dict[str, Any], baseline_config: dict[str, Any], config: dict[str, Any],
) -> dict[tuple[str, str], tuple[str, str | None]]:
    identifiers, by_id, known = choice_maps(catalog)
    prior = prior_distribution(candidate, prior_probability, identifiers)
    likelihoods = {
        hypothesis: joint_distribution(
            hypothesis, candidate, reliability, assumed_correlation, by_id, config,
        ) for hypothesis in identifiers
    }
    actions = [("ABSTAIN", None), ("UNSUPPORTED", None)] + [("KNOWN", row["intent_id"]) for row in known]
    policy = {}
    for observation in itertools.product(A_ANSWERS, B_ANSWERS):
        weights = {hypothesis: prior[hypothesis] * likelihoods[hypothesis][observation] for hypothesis in identifiers}
        normalizer = sum(weights.values())
        if normalizer == 0.0:
            policy[observation] = ("ABSTAIN", None)
            continue
        policy[observation] = min(actions, key=lambda action: (
            sum(
                weights[hypothesis] / normalizer
                * hypothesis_action_cost(hypothesis, action, by_id, baseline_config)
                for hypothesis in identifiers
            ), action,
        ))
    return policy


def evaluate_condition(
    rows: list[dict[str, Any]], candidates: dict[str, str], reliability: float,
    actual_correlation: float, assumed_correlation: float, prior_probability: float,
    catalog: dict[str, Any], baseline_config: dict[str, Any], config: dict[str, Any],
) -> dict[str, float]:
    _, by_id, _ = choice_maps(catalog)
    cache = {
        candidate: bayes_policy(
            candidate, reliability, assumed_correlation, prior_probability,
            catalog, baseline_config, config,
        ) for candidate in sorted(set(candidates.values()))
    }
    total = known_exact = unsupported_correct = false_known = nonabstain = 0.0
    known_count = unsupported_count = nonknown_count = 0
    for row in rows:
        identifier = row["population_id"]
        candidate = candidates[identifier]
        truth = truth_choice(row, catalog)
        kind = by_id[truth]["kind"]
        observations = joint_distribution(truth, candidate, reliability, actual_correlation, by_id, config)
        if kind == "KNOWN": known_count += 1
        if kind == "UNSUPPORTED": unsupported_count += 1
        if kind != "KNOWN": nonknown_count += 1
        exact = ("KNOWN", by_id[truth]["intent_id"]) if kind == "KNOWN" else ("UNSUPPORTED", None) if kind == "UNSUPPORTED" else ("ABSTAIN", None)
        for observation, probability in observations.items():
            action = cache[candidate][observation]
            total += probability * hypothesis_action_cost(truth, action, by_id, baseline_config)
            known_exact += probability * (kind == "KNOWN" and action == exact)
            unsupported_correct += probability * (kind == "UNSUPPORTED" and action[0] == "UNSUPPORTED")
            false_known += probability * (kind != "KNOWN" and action[0] == "KNOWN")
            nonabstain += probability * (action[0] != "ABSTAIN")
    count = len(rows)
    return {
        "mean_regret_including_clarification": total / count + config["channel"]["totalClarificationCost"],
        "known_exact_probability": known_exact / known_count,
        "unsupported_correct_probability": unsupported_correct / unsupported_count,
        "false_known_probability": false_known / nonknown_count,
        "non_abstain_action_probability": nonabstain / count,
    }


def run_simulator(
    population: dict[str, Any], historical_result: dict[str, Any], catalog: dict[str, Any],
    baseline_config: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    rows = population["selected_population"]
    candidates = {row["population_id"]: historical_result["fixtures"][row["population_id"]]["candidate_choice_id"] for row in rows}
    conditions = {}
    for prior in config["priorRegimes"]:
        conditions[prior["id"]] = {}
        for reliability in config["channel"]["marginalCorrectness"]:
            rkey = f"{reliability:.2f}"; conditions[prior["id"]][rkey] = {}
            for rho in config["channel"]["sharedFailureCorrelation"]:
                key = f"{rho:.2f}"
                conditions[prior["id"]][rkey][key] = {
                    "correlation_aware": evaluate_condition(
                        rows, candidates, reliability, rho, rho, prior["candidateProbability"],
                        catalog, baseline_config, config,
                    ),
                    "independence_assumed": evaluate_condition(
                        rows, candidates, reliability, rho, 0.0, prior["candidateProbability"],
                        catalog, baseline_config, config,
                    ),
                }
    baseline = historical_result["summary"]["policy_metrics"]["V112_policy_from_pass_one"]
    gates = config["outcomeGates"]
    reliability = f"{gates['requiredReliability']:.2f}"
    required_rhos = [rho for rho in config["channel"]["sharedFailureCorrelation"] if rho <= gates["maximumRequiredCorrelation"]]
    aware = [
        conditions[prior["id"]][reliability][f"{rho:.2f}"]["correlation_aware"]
        for prior in config["priorRegimes"] for rho in required_rhos
    ]
    misspecified = [
        conditions[prior["id"]][reliability]["0.50"]["independence_assumed"]
        for prior in config["priorRegimes"]
    ]
    perfect = [conditions[prior["id"]]["1.00"]["0.00"]["correlation_aware"] for prior in config["priorRegimes"]]
    checks = {
        "aware_mean_regret_every_prior_and_required_correlation": all(item["mean_regret_including_clarification"] <= gates["maximumAwareMeanRegretEveryPriorAndRequiredCorrelation"] for item in aware),
        "aware_known_exact_every_prior_and_required_correlation": all(item["known_exact_probability"] >= gates["minimumAwareKnownExactEveryPriorAndRequiredCorrelation"] for item in aware),
        "aware_unsupported_every_prior_and_required_correlation": all(item["unsupported_correct_probability"] >= gates["minimumAwareUnsupportedCorrectEveryPriorAndRequiredCorrelation"] for item in aware),
        "aware_false_known_every_prior_and_required_correlation": all(item["false_known_probability"] <= gates["maximumAwareFalseKnownEveryPriorAndRequiredCorrelation"] for item in aware),
        "misspecified_regret_at_050_every_prior": all(item["mean_regret_including_clarification"] <= gates["maximumMisspecifiedMeanRegretAt050EveryPrior"] for item in misspecified),
        "misspecified_false_known_at_050_every_prior": all(item["false_known_probability"] <= gates["maximumMisspecifiedFalseKnownAt050EveryPrior"] for item in misspecified),
        "perfect_channel_mean_regret_every_prior": all(abs(item["mean_regret_including_clarification"] - gates["requiredPerfectChannelMeanRegretEveryPrior"]) <= 1e-12 for item in perfect),
        "rho_one_stress_reported": all("1.00" in conditions[prior["id"]][reliability] for prior in config["priorRegimes"]),
        "true_hypothesis_retention": gates["requiredTrueHypothesisRetention"] == 1.0,
        "zero_actual_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {
        "historical_record_count": len(rows),
        "baseline_mean_regret": baseline["mean_regret"],
        "conditions": conditions, "outcome_gates": checks, "outcome_pass": passed,
        "decision": config["decisionRule"]["ifAllOutcomeAndAccessGatesPass"] if passed else config["decisionRule"]["otherwise"],
        "true_hypothesis_retention": 1.0, "actual_execution_count": 0,
        "individual_record_emission_count": 0,
    }


__all__ = [
    "A_ANSWERS", "B_ANSWERS", "bayes_policy", "error_distribution",
    "evaluate_condition", "joint_distribution", "run_simulator", "true_observations",
]
