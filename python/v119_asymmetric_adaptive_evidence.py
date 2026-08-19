from __future__ import annotations

import itertools
from typing import Any

from v116_typed_clarification_voi import choice_maps, hypothesis_action_cost, prior_distribution, truth_choice
from v117_causal_clarification_simulator import error_distribution, marginal_distribution


ROOT_ANSWERS = ("CONFIRM", "REJECT", "UNSURE")
IDENTITY_ANSWERS = ("MATCH", "MISMATCH", "UNSURE")
STATUS_ANSWERS = ("DECLARED", "UNDECLARED_VISIBLE", "OUTSIDE_VISIBLE", "UNSURE")


def truths(hypothesis: str, candidate: str, by_id: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    kind = by_id[hypothesis]["kind"]
    exact = kind == "KNOWN" and hypothesis == candidate
    root = "UNSURE" if kind == "ABSTAIN" else "CONFIRM" if exact else "REJECT"
    identity = "UNSURE" if kind == "ABSTAIN" else "MATCH" if exact else "MISMATCH"
    status = {"KNOWN": "DECLARED", "NOVEL": "UNDECLARED_VISIBLE", "UNSUPPORTED": "OUTSIDE_VISIBLE", "ABSTAIN": "UNSURE"}[kind]
    return root, identity, status


def joint_distribution(
    hypothesis: str, candidate: str, reliability: float, correlation: float,
    by_id: dict[str, dict[str, Any]], config: dict[str, Any],
) -> dict[tuple[str, str, str, str], float]:
    if not 0.0 <= correlation <= 1.0: raise ValueError("correlation must be bounded")
    truth_root, truth_identity, truth_status = truths(hypothesis, candidate, by_id)
    fraction = config["channel"]["nonCorrectMassToUnsure"]
    root_error = error_distribution(truth_root, ROOT_ANSWERS, "UNSURE", fraction)
    root_marginal = marginal_distribution(truth_root, ROOT_ANSWERS, "UNSURE", reliability, fraction)
    output: dict[tuple[str, str, str, str], float] = {}
    for first in ROOT_ANSWERS:
        if first == "CONFIRM":
            branch = "IDENTITY"; answers = IDENTITY_ANSWERS; truth_branch = truth_identity
        else:
            branch = "STATUS"; answers = STATUS_ANSWERS; truth_branch = truth_status
        branch_error = error_distribution(truth_branch, answers, "UNSURE", fraction)
        branch_marginal = marginal_distribution(truth_branch, answers, "UNSURE", reliability, fraction)
        for second, third in itertools.product(answers, repeat=2):
            independent = root_marginal[first] * branch_marginal[second] * branch_marginal[third]
            shared = (
                reliability * (first == truth_root and second == truth_branch and third == truth_branch)
                + (1.0 - reliability) * root_error[first] * branch_error[second] * branch_error[third]
            )
            output[(first, branch, second, third)] = (1.0 - correlation) * independent + correlation * shared
    if abs(sum(output.values()) - 1.0) > 1e-12: raise AssertionError("adaptive channel is not normalized")
    return output


def bayes_policy(
    candidate: str, reliability: float, assumed_correlation: float, prior_probability: float,
    catalog: dict[str, Any], baseline_config: dict[str, Any], config: dict[str, Any],
) -> dict[tuple[str, str, str, str], tuple[str, str | None]]:
    identifiers, by_id, known = choice_maps(catalog)
    prior = prior_distribution(candidate, prior_probability, identifiers)
    likelihoods = {h: joint_distribution(h, candidate, reliability, assumed_correlation, by_id, config) for h in identifiers}
    observations = sorted(next(iter(likelihoods.values())))
    actions = [("ABSTAIN", None), ("UNSUPPORTED", None)] + [("KNOWN", row["intent_id"]) for row in known]
    policy = {}
    for observation in observations:
        weights = {h: prior[h] * likelihoods[h][observation] for h in identifiers}
        normalizer = sum(weights.values())
        if normalizer == 0.0:
            policy[observation] = ("ABSTAIN", None); continue
        policy[observation] = min(actions, key=lambda action: (
            sum(weights[h] / normalizer * hypothesis_action_cost(h, action, by_id, baseline_config) for h in identifiers), action,
        ))
    return policy


def evaluate_condition(
    rows: list[dict[str, Any]], candidates: dict[str, str], reliability: float,
    actual_correlation: float, assumed_correlation: float, prior_probability: float,
    catalog: dict[str, Any], baseline_config: dict[str, Any], config: dict[str, Any],
) -> dict[str, float]:
    _, by_id, _ = choice_maps(catalog)
    cache = {candidate: bayes_policy(candidate, reliability, assumed_correlation, prior_probability, catalog, baseline_config, config) for candidate in sorted(set(candidates.values()))}
    total = known_exact = unsupported_correct = false_known = nonabstain = 0.0
    known_count = unsupported_count = nonknown_count = 0
    for row in rows:
        candidate = candidates[row["population_id"]]; truth = truth_choice(row, catalog); kind = by_id[truth]["kind"]
        if kind == "KNOWN": known_count += 1
        if kind == "UNSUPPORTED": unsupported_count += 1
        if kind != "KNOWN": nonknown_count += 1
        exact = ("KNOWN", by_id[truth]["intent_id"]) if kind == "KNOWN" else ("UNSUPPORTED", None) if kind == "UNSUPPORTED" else ("ABSTAIN", None)
        for observation, probability in joint_distribution(truth, candidate, reliability, actual_correlation, by_id, config).items():
            action = cache[candidate][observation]
            total += probability * hypothesis_action_cost(truth, action, by_id, baseline_config)
            known_exact += probability * (kind == "KNOWN" and action == exact)
            unsupported_correct += probability * (kind == "UNSUPPORTED" and action[0] == "UNSUPPORTED")
            false_known += probability * (kind != "KNOWN" and action[0] == "KNOWN")
            nonabstain += probability * (action[0] != "ABSTAIN")
    count = len(rows)
    return {
        "mean_regret_including_clarification": total / count + config["adaptiveTree"]["totalCostEveryPath"],
        "known_exact_probability": known_exact / known_count,
        "unsupported_correct_probability": unsupported_correct / unsupported_count,
        "false_known_probability": false_known / nonknown_count,
        "non_abstain_action_probability": nonabstain / count,
    }


def run_simulator(
    population: dict[str, Any], historical: dict[str, Any], catalog: dict[str, Any],
    baseline_config: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    rows = population["selected_population"]
    candidates = {row["population_id"]: historical["fixtures"][row["population_id"]]["candidate_choice_id"] for row in rows}
    conditions = {}
    for prior in config["priorRegimes"]:
        conditions[prior["id"]] = {}
        for reliability in config["channel"]["marginalCorrectness"]:
            rkey = f"{reliability:.2f}"; conditions[prior["id"]][rkey] = {}
            for rho in config["channel"]["sharedFailureCorrelation"]:
                key = f"{rho:.2f}"
                conditions[prior["id"]][rkey][key] = {
                    "correlation_aware": evaluate_condition(rows, candidates, reliability, rho, rho, prior["candidateProbability"], catalog, baseline_config, config),
                    "independence_assumed": evaluate_condition(rows, candidates, reliability, rho, 0.0, prior["candidateProbability"], catalog, baseline_config, config),
                }
    baseline = historical["summary"]["policy_metrics"]["V112_policy_from_pass_one"]
    gates = config["outcomeGates"]; reliability = f"{gates['requiredReliability']:.2f}"
    required_rhos = [rho for rho in config["channel"]["sharedFailureCorrelation"] if rho <= gates["maximumRequiredCorrelation"]]
    aware = [conditions[p["id"]][reliability][f"{rho:.2f}"]["correlation_aware"] for p in config["priorRegimes"] for rho in required_rhos]
    misspecified = [conditions[p["id"]][reliability]["0.50"]["independence_assumed"] for p in config["priorRegimes"]]
    perfect = [conditions[p["id"]]["1.00"]["0.00"]["correlation_aware"] for p in config["priorRegimes"]]
    checks = {
        "aware_mean_regret_every_prior_and_required_correlation": all(x["mean_regret_including_clarification"] <= gates["maximumAwareMeanRegretEveryPriorAndRequiredCorrelation"] for x in aware),
        "aware_known_exact_every_prior_and_required_correlation": all(x["known_exact_probability"] >= gates["minimumAwareKnownExactEveryPriorAndRequiredCorrelation"] for x in aware),
        "aware_unsupported_every_prior_and_required_correlation": all(x["unsupported_correct_probability"] >= gates["minimumAwareUnsupportedCorrectEveryPriorAndRequiredCorrelation"] for x in aware),
        "aware_false_known_every_prior_and_required_correlation": all(x["false_known_probability"] <= gates["maximumAwareFalseKnownEveryPriorAndRequiredCorrelation"] for x in aware),
        "misspecified_regret_at_050_every_prior": all(x["mean_regret_including_clarification"] <= gates["maximumMisspecifiedMeanRegretAt050EveryPrior"] for x in misspecified),
        "misspecified_false_known_at_050_every_prior": all(x["false_known_probability"] <= gates["maximumMisspecifiedFalseKnownAt050EveryPrior"] for x in misspecified),
        "perfect_channel_mean_regret_every_prior": all(abs(x["mean_regret_including_clarification"] - gates["requiredPerfectChannelMeanRegretEveryPrior"]) <= 1e-12 for x in perfect),
        "rho_one_stress_reported": all("1.00" in conditions[p["id"]][reliability] for p in config["priorRegimes"]),
        "true_hypothesis_retention": gates["requiredTrueHypothesisRetention"] == 1.0,
        "zero_actual_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {"historical_record_count": len(rows), "baseline_mean_regret": baseline["mean_regret"], "conditions": conditions, "outcome_gates": checks, "outcome_pass": passed, "decision": config["decisionRule"]["ifAllOutcomeAndAccessGatesPass"] if passed else config["decisionRule"]["otherwise"], "true_hypothesis_retention": 1.0, "actual_execution_count": 0, "individual_record_emission_count": 0}


__all__ = ["IDENTITY_ANSWERS", "ROOT_ANSWERS", "STATUS_ANSWERS", "bayes_policy", "joint_distribution", "run_simulator", "truths"]
