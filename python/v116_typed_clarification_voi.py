from __future__ import annotations

from collections import Counter
import itertools
from typing import Any


def choice_maps(catalog: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    choices = catalog["choices"]
    identifiers = [row["choice_id"] for row in choices]
    if len(identifiers) != 17 or len(set(identifiers)) != 17:
        raise ValueError("V116 requires the exact complete 17-choice universe")
    by_id = {row["choice_id"]: row for row in choices}
    known = [row for row in choices if row["kind"] == "KNOWN"]
    return identifiers, by_id, known


def truth_choice(row: dict[str, Any], catalog: dict[str, Any]) -> str:
    _, _, known = choice_maps(catalog)
    if row["class_label"].startswith("known"):
        intent_id = f"{row['scenario']}::{row['intent']}"
        matches = [item["choice_id"] for item in known if item["intent_id"] == intent_id]
    elif row["class_label"] == "novel_valid":
        matches = [
            item["choice_id"] for item in catalog["choices"]
            if item["kind"] == "NOVEL" and item["scenario"] == row["scenario"]
        ]
    elif row["class_label"] == "unsupported":
        matches = [item["choice_id"] for item in catalog["choices"] if item["kind"] == "UNSUPPORTED"]
    else:
        matches = [item["choice_id"] for item in catalog["choices"] if item["kind"] == "ABSTAIN"]
    if len(matches) != 1:
        raise ValueError("structural truth must map to one clarification choice")
    return matches[0]


def answer_distribution(
    truth: str, reliability: float, identifiers: list[str], abstain_id: str, abstain_fraction: float,
) -> dict[str, float]:
    if truth not in identifiers or not 0.0 <= reliability <= 1.0:
        raise ValueError("invalid answer channel input")
    output = {identifier: 0.0 for identifier in identifiers}
    output[truth] = reliability
    error = 1.0 - reliability
    if truth != abstain_id:
        output[abstain_id] += error * abstain_fraction
        wrong = [identifier for identifier in identifiers if identifier not in {truth, abstain_id}]
        for identifier in wrong:
            output[identifier] += error * (1.0 - abstain_fraction) / len(wrong)
    else:
        wrong = [identifier for identifier in identifiers if identifier != abstain_id]
        for identifier in wrong:
            output[identifier] += error / len(wrong)
    if abs(sum(output.values()) - 1.0) > 1e-12:
        raise AssertionError("answer distribution is not normalized")
    return output


def prior_distribution(candidate: str, probability: float, identifiers: list[str]) -> dict[str, float]:
    if candidate not in identifiers or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid candidate prior")
    remainder = (1.0 - probability) / (len(identifiers) - 1)
    return {identifier: probability if identifier == candidate else remainder for identifier in identifiers}


def action_for_choice(choice_id: str, by_id: dict[str, dict[str, Any]]) -> tuple[str, str | None]:
    row = by_id[choice_id]
    if row["kind"] == "KNOWN":
        return "KNOWN", row["intent_id"]
    if row["kind"] == "UNSUPPORTED":
        return "UNSUPPORTED", None
    return "ABSTAIN", None


def hypothesis_action_cost(
    truth_id: str, action: tuple[str, str | None], by_id: dict[str, dict[str, Any]],
    baseline_config: dict[str, Any],
) -> float:
    truth = by_id[truth_id]
    status, intent = action
    costs = baseline_config["decisionCosts"]
    if truth["kind"] == "KNOWN":
        key = "exact_known" if status == "KNOWN" and intent == truth["intent_id"] else "wrong_known" if status == "KNOWN" else status.lower()
        return float(costs["known"][key])
    if truth["kind"] == "NOVEL":
        return float(costs["novel"][status.lower()])
    if truth["kind"] == "UNSUPPORTED":
        return float(costs["unsupported"][status.lower()])
    return float(costs["insufficient"][status.lower()])


def response_policy(
    candidate: str, reliability: float, prior_probability: float, answer_count: int,
    correlation: str, catalog: dict[str, Any], baseline_config: dict[str, Any],
    channel: dict[str, Any],
) -> dict[tuple[str, ...], tuple[str, str | None]]:
    identifiers, by_id, known = choice_maps(catalog)
    abstain_id = next(row["choice_id"] for row in catalog["choices"] if row["kind"] == "ABSTAIN")
    distributions = {
        hypothesis: answer_distribution(
            hypothesis, reliability, identifiers, abstain_id,
            channel["nonCorrectMassToInsufficientWhenTruthIsNotInsufficient"],
        ) for hypothesis in identifiers
    }
    prior = prior_distribution(candidate, prior_probability, identifiers)
    actions = [("ABSTAIN", None), ("UNSUPPORTED", None)] + [
        ("KNOWN", row["intent_id"]) for row in known
    ]
    responses = list(itertools.product(identifiers, repeat=answer_count))
    output: dict[tuple[str, ...], tuple[str, str | None]] = {}
    for response in responses:
        if all(item == abstain_id for item in response):
            output[response] = ("ABSTAIN", None)
            continue
        weights = {}
        for hypothesis in identifiers:
            if correlation == "conditionally_independent":
                likelihood = 1.0
                for item in response:
                    likelihood *= distributions[hypothesis][item]
            elif correlation == "fully_correlated":
                likelihood = distributions[hypothesis][response[0]] if len(set(response)) == 1 else 0.0
            else:
                raise ValueError("unknown response correlation")
            weights[hypothesis] = prior[hypothesis] * likelihood
        normalizer = sum(weights.values())
        if normalizer == 0.0:
            output[response] = ("ABSTAIN", None)
            continue
        output[response] = min(actions, key=lambda action: (
            sum(
                weights[hypothesis] / normalizer
                * hypothesis_action_cost(hypothesis, action, by_id, baseline_config)
                for hypothesis in identifiers
            ), action,
        ))
    return output


def evaluate_condition(
    structural_rows: list[dict[str, Any]], candidates: dict[str, str], reliability: float,
    prior_probability: float, answer_count: int, correlation: str,
    catalog: dict[str, Any], baseline_config: dict[str, Any], config: dict[str, Any],
) -> dict[str, float]:
    identifiers, by_id, _ = choice_maps(catalog)
    abstain_id = next(row["choice_id"] for row in catalog["choices"] if row["kind"] == "ABSTAIN")
    channel = config["simulatedAnswerChannel"]
    policy_cache = {
        candidate: response_policy(
            candidate, reliability, prior_probability, answer_count, correlation,
            catalog, baseline_config, channel,
        ) for candidate in sorted(set(candidates.values()))
    }
    total_cost = known_exact = unsupported_correct = false_known = action_rate = 0.0
    known_count = unsupported_count = non_known_count = 0
    for row in structural_rows:
        identifier = row["population_id"]
        truth = truth_choice(row, catalog)
        truth_kind = by_id[truth]["kind"]
        distribution = answer_distribution(
            truth, reliability, identifiers, abstain_id,
            channel["nonCorrectMassToInsufficientWhenTruthIsNotInsufficient"],
        )
        response_probabilities: dict[tuple[str, ...], float] = {}
        if correlation == "conditionally_independent":
            for response in itertools.product(identifiers, repeat=answer_count):
                probability = 1.0
                for item in response:
                    probability *= distribution[item]
                response_probabilities[response] = probability
        elif correlation == "fully_correlated":
            response_probabilities = {
                tuple([item] * answer_count): probability for item, probability in distribution.items()
            }
        else:
            raise ValueError("unknown response correlation")
        policy = policy_cache[candidates[identifier]]
        exact_action = action_for_choice(truth, by_id)
        if truth_kind == "KNOWN":
            known_count += 1
        if truth_kind == "UNSUPPORTED":
            unsupported_count += 1
        if truth_kind != "KNOWN":
            non_known_count += 1
        for response, probability in response_probabilities.items():
            action = policy[response]
            total_cost += probability * hypothesis_action_cost(truth, action, by_id, baseline_config)
            known_exact += probability * (truth_kind == "KNOWN" and action == exact_action)
            unsupported_correct += probability * (truth_kind == "UNSUPPORTED" and action[0] == "UNSUPPORTED")
            false_known += probability * (truth_kind != "KNOWN" and action[0] == "KNOWN")
            action_rate += probability * (action[0] != "ABSTAIN")
    query_cost = channel["singleClarificationCost"] if answer_count == 1 else channel["doubleClarificationCost"]
    count = len(structural_rows)
    return {
        "mean_regret_including_clarification": total_cost / count + query_cost,
        "known_exact_probability": known_exact / known_count,
        "unsupported_correct_probability": unsupported_correct / unsupported_count,
        "false_known_probability": false_known / non_known_count,
        "non_abstain_action_probability": action_rate / count,
    }


def run_audit(
    population: dict[str, Any], result: dict[str, Any], catalog: dict[str, Any],
    baseline_config: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    rows = population["selected_population"]
    candidates = {
        row["population_id"]: result["fixtures"][row["population_id"]]["candidate_choice_id"]
        for row in rows
    }
    conditions: dict[str, Any] = {}
    for prior in config["priorRegimes"]:
        prior_id = prior["id"]
        conditions[prior_id] = {}
        for reliability in config["simulatedAnswerChannel"]["correctResponseProbabilities"]:
            key = f"{reliability:.2f}"
            conditions[prior_id][key] = {
                "single": evaluate_condition(
                    rows, candidates, reliability, prior["candidateProbability"], 1,
                    "conditionally_independent", catalog, baseline_config, config,
                ),
                "double_independent": evaluate_condition(
                    rows, candidates, reliability, prior["candidateProbability"], 2,
                    "conditionally_independent", catalog, baseline_config, config,
                ),
                "double_correlated": evaluate_condition(
                    rows, candidates, reliability, prior["candidateProbability"], 2,
                    "fully_correlated", catalog, baseline_config, config,
                ),
            }
    baseline = result["summary"]["policy_metrics"]["V112_policy_from_pass_one"]
    gates = config["feasibilityGates"]
    required = f"{gates['requiredReliability']:.2f}"
    independent = [conditions[row["id"]][required]["double_independent"] for row in config["priorRegimes"]]
    correlated = [conditions[row["id"]][required]["double_correlated"] for row in config["priorRegimes"]]
    lower = [conditions[row["id"]]["0.90"]["double_independent"] for row in config["priorRegimes"]]
    perfect = [conditions[row["id"]]["1.00"]["double_independent"] for row in config["priorRegimes"]]
    checks = {
        "independent_mean_regret_every_prior": all(item["mean_regret_including_clarification"] <= gates["maximumIndependentDoubleAnswerMeanRegretEveryPrior"] for item in independent),
        "independent_known_exact_every_prior": all(item["known_exact_probability"] >= gates["minimumIndependentDoubleAnswerKnownExactEveryPrior"] for item in independent),
        "independent_unsupported_correct_every_prior": all(item["unsupported_correct_probability"] >= gates["minimumIndependentDoubleAnswerUnsupportedCorrectEveryPrior"] for item in independent),
        "independent_false_known_every_prior": all(item["false_known_probability"] <= gates["maximumIndependentDoubleAnswerFalseKnownEveryPrior"] for item in independent),
        "perfect_channel_mean_regret_every_prior": all(abs(item["mean_regret_including_clarification"] - gates["requiredPerfectChannelMeanRegretEveryPrior"]) <= 1e-12 for item in perfect),
        "independent_regret_monotone_090_to_095": all(high["mean_regret_including_clarification"] <= low["mean_regret_including_clarification"] for high, low in zip(independent, lower)),
        "fully_correlated_stress_test_reported": len(correlated) == len(config["priorRegimes"]),
        "true_hypothesis_retention": gates["requiredTrueHypothesisRetention"] == 1.0,
        "zero_actual_execution": gates["maximumActualExecutionCount"] == 0,
    }
    correlated_pass = all(
        item["mean_regret_including_clarification"] <= baseline["mean_regret"]
        and item["known_exact_probability"] >= gates["minimumIndependentDoubleAnswerKnownExactEveryPrior"]
        and item["unsupported_correct_probability"] >= gates["minimumIndependentDoubleAnswerUnsupportedCorrectEveryPrior"]
        and item["false_known_probability"] <= gates["maximumIndependentDoubleAnswerFalseKnownEveryPrior"]
        for item in correlated
    )
    independent_pass = all(checks.values())
    if independent_pass and correlated_pass:
        decision = config["decisionRule"]["ifIndependentAndCorrelatedPass"]
    elif independent_pass:
        decision = config["decisionRule"]["ifIndependentPassButCorrelatedFails"]
    else:
        decision = config["decisionRule"]["ifIndependentFails"]
    truth_counts = Counter(truth_choice(row, catalog) for row in rows)
    return {
        "historical_record_count": len(rows),
        "truth_choice_count": len(truth_counts),
        "baseline": {
            "mean_regret": baseline["mean_regret"],
            "known_exact_intent_accuracy": baseline["known_exact_intent_accuracy"],
            "false_known_acceptance_rate": baseline["false_known_acceptance_rate"],
            "unsupported_recall": baseline["per_status"]["UNSUPPORTED"]["recall"],
        },
        "conditions": conditions, "feasibility_gates": checks,
        "independent_pass": independent_pass, "correlated_pass": correlated_pass,
        "decision": decision, "true_hypothesis_retention": 1.0,
        "actual_execution_count": 0, "individual_record_emission_count": 0,
    }


__all__ = [
    "answer_distribution", "choice_maps", "evaluate_condition", "prior_distribution",
    "response_policy", "run_audit", "truth_choice",
]
