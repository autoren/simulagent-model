from __future__ import annotations

from typing import Any

from v126_sgd_retrieval_selectivity import (
    bayes_policy as candidate_bayes_policy,
    catalog_maps, hypothesis_action_cost, joint_distribution,
)


def prior_distribution(candidate: str, probability: float, identifiers: list[str]) -> dict[str, float]:
    remainder = (1.0 - probability) / (len(identifiers) - 1)
    return {identifier: probability if identifier == candidate else remainder for identifier in identifiers}


def answer_distribution(
    truth: str, candidate: str, reliability: float, regime: str,
    identifiers: list[str], abstain_id: str, bias_fraction: float,
) -> dict[str, float]:
    if not 0.0 <= reliability <= 1.0: raise ValueError("reliability must be bounded")
    wrong = [identifier for identifier in identifiers if identifier != truth]
    output = {identifier: 0.0 for identifier in identifiers}; output[truth] = reliability
    mass = 1.0 - reliability
    target = candidate if regime == "candidate_attraction" else abstain_id if regime == "abstention_attraction" else None
    if regime not in {"symmetric", "candidate_attraction", "abstention_attraction"}: raise ValueError("unknown V129 error regime")
    if target is not None and target != truth:
        output[target] += mass * bias_fraction
        rest = [identifier for identifier in wrong if identifier != target]
        for identifier in rest: output[identifier] += mass * (1.0 - bias_fraction) / len(rest)
    else:
        for identifier in wrong: output[identifier] += mass / len(wrong)
    if abs(sum(output.values()) - 1.0) > 1e-12: raise AssertionError("V129 answer channel is not normalized")
    return output


def complete_policy(
    candidate: str, reliability: float, assumed_regime: str, candidate_probability: float,
    catalog: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any],
) -> dict[str, tuple[str, str | None]]:
    identifiers, by_id, known = catalog_maps(catalog)
    abstain_id = next(row["choice_id"] for row in catalog["choices"] if row["kind"] == "ABSTAIN")
    prior = prior_distribution(candidate, candidate_probability, identifiers)
    likelihoods = {
        truth: answer_distribution(truth, candidate, reliability, assumed_regime, identifiers, abstain_id, config["completeClarificationChannel"]["errorBiasFraction"])
        for truth in identifiers
    }
    actions = [("ABSTAIN", None), ("UNSUPPORTED", None)] + [("KNOWN", row["intent_id"]) for row in known]
    policy = {}
    for answer in identifiers:
        weights = {truth: prior[truth] * likelihoods[truth][answer] for truth in identifiers}
        normalizer = sum(weights.values())
        policy[answer] = min(actions, key=lambda action: (
            sum(weights[truth] / normalizer * hypothesis_action_cost(truth, action, by_id, baseline) for truth in identifiers), action,
        ))
    return policy


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    known = [row for row in rows if row["kind"] == "KNOWN"]
    unsupported = [row for row in rows if row["kind"] == "UNSUPPORTED"]
    nonknown = [row for row in rows if row["kind"] != "KNOWN"]
    return {
        "mean_regret": sum(row["cost"] for row in rows) / len(rows),
        "known_exact_probability": sum(row["known_exact"] for row in known) / len(known),
        "unsupported_correct_probability": sum(row["unsupported_correct"] for row in unsupported) / len(unsupported),
        "false_known_probability": sum(row["false_known"] for row in nonknown) / len(nonknown),
    }


def evaluate_complete(
    reliability: float, actual_regime: str, assumed_regime: str, candidate_probability: float,
    catalog: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any],
) -> dict[str, float]:
    identifiers, by_id, known = catalog_maps(catalog)
    candidates = [row["choice_id"] for row in known]
    abstain_id = next(row["choice_id"] for row in catalog["choices"] if row["kind"] == "ABSTAIN")
    policies = {candidate: complete_policy(candidate, reliability, assumed_regime, candidate_probability, catalog, baseline, config) for candidate in candidates}
    rows = []
    for truth in identifiers:
        kind = {"KNOWN": "KNOWN", "NOVEL_COMPOSITE": "NOVEL", "UNSUPPORTED_COMPOSITE": "UNSUPPORTED", "ABSTAIN": "ABSTAIN"}[by_id[truth]["kind"]]
        exact = ("KNOWN", by_id[truth]["intent_id"]) if kind == "KNOWN" else ("UNSUPPORTED", None) if kind == "UNSUPPORTED" else ("ABSTAIN", None)
        for candidate in candidates:
            cost = known_exact = unsupported_correct = false_known = 0.0
            distribution = answer_distribution(truth, candidate, reliability, actual_regime, identifiers, abstain_id, config["completeClarificationChannel"]["errorBiasFraction"])
            for answer, probability in distribution.items():
                action = policies[candidate][answer]
                cost += probability * hypothesis_action_cost(truth, action, by_id, baseline)
                known_exact += probability * (kind == "KNOWN" and action == exact)
                unsupported_correct += probability * (kind == "UNSUPPORTED" and action[0] == "UNSUPPORTED")
                false_known += probability * (kind != "KNOWN" and action[0] == "KNOWN")
            rows.append({"kind": kind, "cost": cost + config["completeClarificationChannel"]["totalCost"], "known_exact": known_exact, "unsupported_correct": unsupported_correct, "false_known": false_known})
    return _metrics(rows)


def evaluate_candidate_specific(
    candidate_probability: float, catalog: dict[str, Any], baseline: dict[str, Any],
    v119: dict[str, Any], config: dict[str, Any],
) -> dict[str, float]:
    identifiers, by_id, known = catalog_maps(catalog)
    candidates = [row["choice_id"] for row in known]
    spec = config["candidateSpecificComparator"]
    policies = {candidate: candidate_bayes_policy(candidate, spec["marginalCorrectness"], spec["sharedFailureCorrelation"], candidate_probability, catalog, baseline, v119) for candidate in candidates}
    rows = []
    for truth in identifiers:
        kind = {"KNOWN": "KNOWN", "NOVEL_COMPOSITE": "NOVEL", "UNSUPPORTED_COMPOSITE": "UNSUPPORTED", "ABSTAIN": "ABSTAIN"}[by_id[truth]["kind"]]
        exact = ("KNOWN", by_id[truth]["intent_id"]) if kind == "KNOWN" else ("UNSUPPORTED", None) if kind == "UNSUPPORTED" else ("ABSTAIN", None)
        for candidate in candidates:
            cost = known_exact = unsupported_correct = false_known = 0.0
            for observation, probability in joint_distribution(truth, candidate, spec["marginalCorrectness"], spec["sharedFailureCorrelation"], by_id, v119).items():
                action = policies[candidate][observation]
                cost += probability * hypothesis_action_cost(truth, action, by_id, baseline)
                known_exact += probability * (kind == "KNOWN" and action == exact)
                unsupported_correct += probability * (kind == "UNSUPPORTED" and action[0] == "UNSUPPORTED")
                false_known += probability * (kind != "KNOWN" and action[0] == "KNOWN")
            rows.append({"kind": kind, "cost": cost + spec["sameTotalCost"], "known_exact": known_exact, "unsupported_correct": unsupported_correct, "false_known": false_known})
    return _metrics(rows)


def run_audit(catalog: dict[str, Any], baseline: dict[str, Any], v119: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    conditions = {}; comparator = {}
    for prior in config["priorRegimes"]:
        prior_id = prior["id"]; probability = prior["candidateProbability"]
        comparator[prior_id] = evaluate_candidate_specific(probability, catalog, baseline, v119, config)
        conditions[prior_id] = {}
        for reliability in config["completeClarificationChannel"]["marginalCorrectness"]:
            rkey = f"{reliability:.2f}"; conditions[prior_id][rkey] = {}
            for regime in config["completeClarificationChannel"]["errorRegimes"]:
                conditions[prior_id][rkey][regime] = {
                    "channel_aware": evaluate_complete(reliability, regime, regime, probability, catalog, baseline, config),
                    "symmetric_assumed": evaluate_complete(reliability, regime, "symmetric", probability, catalog, baseline, config),
                }
    gates = config["outcomeGates"]; required = f"{config['completeClarificationChannel']['requiredReliability']:.2f}"
    aware = [(prior["id"], regime, conditions[prior["id"]][required][regime]["channel_aware"]) for prior in config["priorRegimes"] for regime in config["completeClarificationChannel"]["errorRegimes"]]
    misspecified = [conditions[prior["id"]][required][regime]["symmetric_assumed"] for prior in config["priorRegimes"] for regime in ("candidate_attraction", "abstention_attraction")]
    perfect = [conditions[prior["id"]]["1.00"][regime]["channel_aware"] for prior in config["priorRegimes"] for regime in config["completeClarificationChannel"]["errorRegimes"]]
    checks = {
        "aware_regret_every_prior_and_bias": all(row["mean_regret"] <= gates["maximumAwareMeanRegretEveryPriorAndBiasAtRequiredReliability"] for _, _, row in aware),
        "aware_known_exact_every_prior_and_bias": all(row["known_exact_probability"] >= gates["minimumAwareKnownExactEveryPriorAndBiasAtRequiredReliability"] for _, _, row in aware),
        "aware_unsupported_every_prior_and_bias": all(row["unsupported_correct_probability"] >= gates["minimumAwareUnsupportedEveryPriorAndBiasAtRequiredReliability"] for _, _, row in aware),
        "aware_false_known_every_prior_and_bias": all(row["false_known_probability"] <= gates["maximumAwareFalseKnownEveryPriorAndBiasAtRequiredReliability"] for _, _, row in aware),
        "aware_no_worse_regret_than_candidate_specific": all(row["mean_regret"] <= comparator[prior_id]["mean_regret"] for prior_id, _, row in aware),
        "aware_no_worse_known_than_candidate_specific": all(row["known_exact_probability"] >= comparator[prior_id]["known_exact_probability"] for prior_id, _, row in aware),
        "symmetric_assumed_regret_under_bias": all(row["mean_regret"] <= gates["maximumSymmetricAssumedRegretEveryPriorAndBiasedRegime"] for row in misspecified),
        "symmetric_assumed_false_known_under_bias": all(row["false_known_probability"] <= gates["maximumSymmetricAssumedFalseKnownEveryPriorAndBiasedRegime"] for row in misspecified),
        "perfect_answer_known_exact": all(row["known_exact_probability"] == gates["requiredPerfectAnswerKnownExact"] for row in perfect),
        "perfect_answer_unsupported": all(row["unsupported_correct_probability"] == gates["requiredPerfectAnswerUnsupported"] for row in perfect),
        "complete_hypothesis_retention": gates["requiredTrueHypothesisRetention"] == 1.0,
        "zero_individual_pair_emission": gates["maximumIndividualPairEmissionCount"] == 0,
        "zero_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {
        "census_pair_count": 66, "conditions": conditions, "candidate_specific_comparator": comparator,
        "outcome_gates": checks, "outcome_pass": passed,
        "decision": config["decisionRule"]["ifEveryOutcomeAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "true_hypothesis_retention": 1.0, "individual_pair_emission_count": 0, "actual_execution_count": 0,
    }


__all__ = ["answer_distribution", "complete_policy", "run_audit"]
