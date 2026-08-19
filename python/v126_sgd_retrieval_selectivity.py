from __future__ import annotations

from collections import Counter
import itertools
from typing import Any

from v117_causal_clarification_simulator import error_distribution, marginal_distribution
from v119_asymmetric_adaptive_evidence import IDENTITY_ANSWERS, ROOT_ANSWERS, STATUS_ANSWERS


def catalog_maps(catalog: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    identifiers = [row["choice_id"] for row in catalog["choices"]]
    if len(identifiers) != 11 or len(set(identifiers)) != 11:
        raise ValueError("V126 requires the exact eleven-choice universe")
    by_id = {row["choice_id"]: row for row in catalog["choices"]}
    known = [row for row in catalog["choices"] if row["kind"] == "KNOWN"]
    return identifiers, by_id, known


def truth_choice(record: dict[str, Any], catalog: dict[str, Any]) -> str:
    if record["class_label"] == "known":
        pair = f"{record['service']}::{record['intent']}"
        matches = [row["choice_id"] for row in catalog["choices"] if row["kind"] == "KNOWN" and row["intent_id"] == pair]
    elif record["class_label"] == "novel_valid":
        matches = [row["choice_id"] for row in catalog["choices"] if row["kind"] == "NOVEL_COMPOSITE" and row["domain"] == record["domain"]]
    elif record["class_label"] == "unsupported":
        matches = [row["choice_id"] for row in catalog["choices"] if row["kind"] == "UNSUPPORTED_COMPOSITE"]
    else:
        matches = [row["choice_id"] for row in catalog["choices"] if row["kind"] == "ABSTAIN"]
    if len(matches) != 1:
        raise ValueError("record does not map to exactly one V126 hypothesis")
    return matches[0]


def normalized_kind(row: dict[str, Any]) -> str:
    return {"KNOWN": "KNOWN", "NOVEL_COMPOSITE": "NOVEL", "UNSUPPORTED_COMPOSITE": "UNSUPPORTED", "ABSTAIN": "ABSTAIN"}[row["kind"]]


def hypothesis_action_cost(
    truth_id: str, action: tuple[str, str | None], by_id: dict[str, dict[str, Any]], baseline: dict[str, Any],
) -> float:
    truth = by_id[truth_id]
    kind = normalized_kind(truth)
    status, intent = action
    costs = baseline["decisionCosts"]
    if kind == "KNOWN":
        key = "exact_known" if status == "KNOWN" and intent == truth["intent_id"] else "wrong_known" if status == "KNOWN" else status.lower()
        return float(costs["known"][key])
    if kind == "NOVEL":
        return float(costs["novel"][status.lower()])
    if kind == "UNSUPPORTED":
        return float(costs["unsupported"][status.lower()])
    return float(costs["insufficient"][status.lower()])


def truths(hypothesis: str, candidate: str, by_id: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    kind = normalized_kind(by_id[hypothesis])
    exact = kind == "KNOWN" and hypothesis == candidate
    root = "UNSURE" if kind == "ABSTAIN" else "CONFIRM" if exact else "REJECT"
    identity = "UNSURE" if kind == "ABSTAIN" else "MATCH" if exact else "MISMATCH"
    status = {"KNOWN": "DECLARED", "NOVEL": "UNDECLARED_VISIBLE", "UNSUPPORTED": "OUTSIDE_VISIBLE", "ABSTAIN": "UNSURE"}[kind]
    return root, identity, status


def joint_distribution(
    hypothesis: str, candidate: str, reliability: float, correlation: float,
    by_id: dict[str, dict[str, Any]], v119_config: dict[str, Any],
) -> dict[tuple[str, str, str, str], float]:
    truth_root, truth_identity, truth_status = truths(hypothesis, candidate, by_id)
    fraction = v119_config["channel"]["nonCorrectMassToUnsure"]
    root_error = error_distribution(truth_root, ROOT_ANSWERS, "UNSURE", fraction)
    root_marginal = marginal_distribution(truth_root, ROOT_ANSWERS, "UNSURE", reliability, fraction)
    output: dict[tuple[str, str, str, str], float] = {}
    for first in ROOT_ANSWERS:
        if first == "CONFIRM":
            branch, answers, branch_truth = "IDENTITY", IDENTITY_ANSWERS, truth_identity
        else:
            branch, answers, branch_truth = "STATUS", STATUS_ANSWERS, truth_status
        branch_error = error_distribution(branch_truth, answers, "UNSURE", fraction)
        branch_marginal = marginal_distribution(branch_truth, answers, "UNSURE", reliability, fraction)
        for second, third in itertools.product(answers, repeat=2):
            independent = root_marginal[first] * branch_marginal[second] * branch_marginal[third]
            shared = reliability * (first == truth_root and second == branch_truth and third == branch_truth) + (1.0 - reliability) * root_error[first] * branch_error[second] * branch_error[third]
            output[(first, branch, second, third)] = (1.0 - correlation) * independent + correlation * shared
    if abs(sum(output.values()) - 1.0) > 1e-12:
        raise AssertionError("V126 channel is not normalized")
    return output


def bayes_policy(
    candidate: str, reliability: float, correlation: float, candidate_probability: float,
    catalog: dict[str, Any], baseline: dict[str, Any], v119_config: dict[str, Any],
) -> dict[tuple[str, str, str, str], tuple[str, str | None]]:
    identifiers, by_id, known = catalog_maps(catalog)
    remainder = (1.0 - candidate_probability) / (len(identifiers) - 1)
    prior = {identifier: candidate_probability if identifier == candidate else remainder for identifier in identifiers}
    likelihoods = {identifier: joint_distribution(identifier, candidate, reliability, correlation, by_id, v119_config) for identifier in identifiers}
    observations = sorted(next(iter(likelihoods.values())))
    actions = [("ABSTAIN", None), ("UNSUPPORTED", None)] + [("KNOWN", row["intent_id"]) for row in known]
    output = {}
    for observation in observations:
        weights = {identifier: prior[identifier] * likelihoods[identifier][observation] for identifier in identifiers}
        normalizer = sum(weights.values())
        if normalizer == 0.0:
            output[observation] = ("ABSTAIN", None)
        else:
            output[observation] = min(actions, key=lambda action: (sum(weights[identifier] / normalizer * hypothesis_action_cost(identifier, action, by_id, baseline) for identifier in identifiers), action))
    return output


def candidate_choice(nearest_intent: str, catalog: dict[str, Any]) -> str:
    matches = [row["choice_id"] for row in catalog["choices"] if row["kind"] == "KNOWN" and row["intent_id"] == nearest_intent]
    if len(matches) != 1:
        raise ValueError("retrieval candidate is not one declared known choice")
    return matches[0]


def retrieval_band(observation: dict[str, Any], config: dict[str, Any]) -> str:
    similarity = observation["similarity"]
    if similarity >= config["frozenRetrieval"]["knownThreshold"]:
        return "high_known"
    if similarity <= config["frozenRetrieval"]["unsupportedThreshold"]:
        return "low_unsupported"
    return "ambiguous_query"


def no_query_action(observation: dict[str, Any], config: dict[str, Any]) -> tuple[str, str | None]:
    band = retrieval_band(observation, config)
    if band == "high_known":
        return "KNOWN", observation["nearest_intent"]
    if band == "low_unsupported":
        return "UNSUPPORTED", None
    return "ABSTAIN", None


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    known = [row for row in rows if row["kind"] == "KNOWN"]
    unsupported = [row for row in rows if row["kind"] == "UNSUPPORTED"]
    non_known = [row for row in rows if row["kind"] != "KNOWN"]
    return {
        "mean_regret": sum(row["cost"] for row in rows) / len(rows),
        "known_exact_probability": sum(row["known_exact"] for row in known) / len(known),
        "unsupported_correct_probability": sum(row["unsupported_correct"] for row in unsupported) / len(unsupported),
        "false_known_probability": sum(row["false_known"] for row in non_known) / len(non_known),
    }


def evaluate_condition(
    records: list[dict[str, Any]], retrieval: dict[str, dict[str, Any]], catalog: dict[str, Any],
    baseline: dict[str, Any], v119_config: dict[str, Any], config: dict[str, Any],
    candidate_probability: float, correlation: float,
) -> dict[str, Any]:
    _, by_id, _ = catalog_maps(catalog)
    reliability = config["queryChannel"]["marginalCorrectness"]
    fee = config["queryChannel"]["totalCost"]
    candidates = {row["record_id"]: candidate_choice(retrieval[row["record_id"]]["nearest_intent"], catalog) for row in records}
    policies = {candidate: bayes_policy(candidate, reliability, correlation, candidate_probability, catalog, baseline, v119_config) for candidate in sorted(set(candidates.values()))}
    no_query_rows, always_rows, selective_rows, values = [], [], [], []
    for record in records:
        identifier = record["record_id"]
        observation = retrieval[identifier]
        truth = truth_choice(record, catalog)
        kind = normalized_kind(by_id[truth])
        exact = ("KNOWN", by_id[truth]["intent_id"]) if kind == "KNOWN" else ("UNSUPPORTED", None) if kind == "UNSUPPORTED" else ("ABSTAIN", None)
        skip_action = no_query_action(observation, config)
        skip_cost = hypothesis_action_cost(truth, skip_action, by_id, baseline)
        query_cost = query_known = query_unsupported = query_false_known = 0.0
        for observed, probability in joint_distribution(truth, candidates[identifier], reliability, correlation, by_id, v119_config).items():
            action = policies[candidates[identifier]][observed]
            query_cost += probability * hypothesis_action_cost(truth, action, by_id, baseline)
            query_known += probability * (kind == "KNOWN" and action == exact)
            query_unsupported += probability * (kind == "UNSUPPORTED" and action[0] == "UNSUPPORTED")
            query_false_known += probability * (kind != "KNOWN" and action[0] == "KNOWN")
        skip_known = float(kind == "KNOWN" and skip_action == exact)
        skip_unsupported = float(kind == "UNSUPPORTED" and skip_action[0] == "UNSUPPORTED")
        skip_false_known = float(kind != "KNOWN" and skip_action[0] == "KNOWN")
        query = retrieval_band(observation, config) == "ambiguous_query"
        no_query = {"kind": kind, "cost": skip_cost, "known_exact": skip_known, "unsupported_correct": skip_unsupported, "false_known": skip_false_known}
        always = {"kind": kind, "cost": query_cost + fee, "known_exact": query_known, "unsupported_correct": query_unsupported, "false_known": query_false_known}
        selective = always if query else no_query
        no_query_rows.append(no_query); always_rows.append(always); selective_rows.append(selective)
        values.append({"query": query, "query_value": skip_cost - query_cost, "no_query": no_query, "always": always})
    queried = [row for row in values if row["query"]]
    skipped = [row for row in values if not row["query"]]
    oracle_skip_ids = {id(row) for row in sorted(values, key=lambda row: row["query_value"])[:len(skipped)]}
    oracle_rows = [row["no_query"] if id(row) in oracle_skip_ids else row["always"] for row in values]
    return {
        "record_count": len(records),
        "retrieval_band_counts": dict(sorted(Counter(retrieval_band(retrieval[row["record_id"]], config) for row in records).items())),
        "skip_fraction": len(skipped) / len(values),
        "query_fraction": len(queried) / len(values),
        "population_average_query_value": sum(row["query_value"] for row in values) / len(values),
        "queried_average_query_value": sum(row["query_value"] for row in queried) / len(queried) if queried else 0.0,
        "skipped_average_query_value": sum(row["query_value"] for row in skipped) / len(skipped) if skipped else 0.0,
        "ask_always": {"mean_regret": config["outcomeGates"]["maximumSelectiveMeanRegretEveryPriorAndCorrelation"]},
        "no_query_retrieval": summarize(no_query_rows),
        "always_query": summarize(always_rows),
        "selective_query": summarize(selective_rows),
        "oracle_same_skip_fraction": summarize(oracle_rows),
    }


def run_evaluation(
    records: list[dict[str, Any]], retrieval: dict[str, dict[str, Any]], catalog: dict[str, Any],
    baseline: dict[str, Any], v119_config: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    conditions = {}
    for prior in config["queryChannel"]["priorRegimes"]:
        for correlation in config["queryChannel"]["sharedFailureCorrelations"]:
            key = f"{prior['id']}@{correlation:.2f}"
            conditions[key] = evaluate_condition(records, retrieval, catalog, baseline, v119_config, config, prior["candidateProbability"], correlation)
    gates = config["outcomeGates"]
    selective = [row["selective_query"] for row in conditions.values()]
    checks = {
        "selective_regret_every_prior_and_correlation": all(row["mean_regret"] <= gates["maximumSelectiveMeanRegretEveryPriorAndCorrelation"] for row in selective),
        "selective_known_exact_every_prior_and_correlation": all(row["known_exact_probability"] >= gates["minimumSelectiveKnownExactEveryPriorAndCorrelation"] for row in selective),
        "selective_unsupported_every_prior_and_correlation": all(row["unsupported_correct_probability"] >= gates["minimumSelectiveUnsupportedCorrectEveryPriorAndCorrelation"] for row in selective),
        "selective_false_known_every_prior_and_correlation": all(row["false_known_probability"] <= gates["maximumSelectiveFalseKnownEveryPriorAndCorrelation"] for row in selective),
        "nontrivial_skip_fraction": all(gates["minimumSkipFraction"] <= row["skip_fraction"] <= gates["maximumSkipFraction"] for row in conditions.values()),
        "queried_average_value_covers_cost": all(row["queried_average_query_value"] >= gates["minimumQueriedAverageQueryValueEveryPriorAndCorrelation"] for row in conditions.values()),
        "skipped_average_value_not_above_cost": all(row["skipped_average_query_value"] <= gates["maximumSkippedAverageQueryValueEveryPriorAndCorrelation"] for row in conditions.values()),
        "selective_no_worse_than_always_query": all(row["selective_query"]["mean_regret"] <= row["always_query"]["mean_regret"] for row in conditions.values()),
        "one_trigger_zero_fit_and_selection": config["primaryTrigger"]["candidateCount"] == 1 and config["primaryTrigger"]["selectionCount"] == 0 and config["frozenRetrieval"]["fitCount"] == 0,
        "complete_hypothesis_retention": gates["requiredTrueHypothesisRetention"] == 1.0,
        "zero_individual_record_emission": gates["maximumIndividualRecordEmissionCount"] == 0,
        "zero_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {
        "record_count": len(records), "conditions": conditions, "outcome_gates": checks,
        "outcome_pass": passed,
        "decision": config["decisionRule"]["ifEveryOutcomeAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"],
        "primary_trigger_count": 1, "threshold_fit_count": 0, "trigger_selection_count": 0,
        "true_hypothesis_retention": 1.0, "individual_record_emission_count": 0, "actual_execution_count": 0,
    }


__all__ = ["catalog_maps", "joint_distribution", "run_evaluation", "truth_choice"]
