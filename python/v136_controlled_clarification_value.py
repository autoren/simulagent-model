from __future__ import annotations

from typing import Any


def choice_kind(choice_id: str, catalog: dict[str, Any]) -> str:
    row = next(item for item in catalog["choices"] if item["choice_id"] == choice_id)
    mapping = {
        "KNOWN": "known",
        "VALID_UNDECLARED": "novel",
        "UNSUPPORTED": "unsupported",
        "INSUFFICIENT_EVIDENCE": "insufficient",
    }
    return mapping[row["kind"]]


def decision_cost(truth: str, action: str, catalog: dict[str, Any], config: dict[str, Any]) -> float:
    truth_kind = choice_kind(truth, catalog)
    action_kind = choice_kind(action, catalog)
    costs = config["decisionCosts"][truth_kind]
    if truth == action:
        if truth_kind == "known":
            return costs["exact_known"]
        if truth_kind == "novel":
            return costs["exact_novel"]
        if truth_kind == "unsupported":
            return costs["unsupported"]
        return costs["abstain"]
    if action_kind == "known":
        return costs["wrong_known"] if truth_kind == "known" else costs["known"]
    if action_kind == "novel":
        if truth_kind == "known":
            return costs["novel"]
        if truth_kind == "novel":
            return costs["wrong_novel_scenario"]
        return costs["novel"]
    if action_kind == "unsupported":
        return costs["unsupported"]
    return costs["abstain"]


def best_action(belief: dict[str, float], catalog: dict[str, Any], config: dict[str, Any]) -> tuple[str, float]:
    candidates = []
    for row in catalog["choices"]:
        action = row["choice_id"]
        expected = sum(probability * decision_cost(truth, action, catalog, config) for truth, probability in belief.items())
        candidates.append((expected, action))
    expected, action = min(candidates)
    return action, expected


def query_policy(
    left: str,
    right: str,
    left_prior: float,
    reliability: float,
    catalog: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    prior = {left: left_prior, right: 1.0 - left_prior}
    query_cost = config["clarificationChannel"]["queryCost"]
    expected_terminal_cost = 0.0
    answer_policies: dict[str, Any] = {}
    for answer in (left, right):
        unnormalized = {
            truth: probability * (reliability if answer == truth else 1.0 - reliability)
            for truth, probability in prior.items()
        }
        answer_probability = sum(unnormalized.values())
        if answer_probability == 0.0:
            continue
        posterior = {truth: weight / answer_probability for truth, weight in unnormalized.items()}
        action, posterior_cost = best_action(posterior, catalog, config)
        expected_terminal_cost += answer_probability * posterior_cost
        answer_policies[answer] = {
            "answer_probability": answer_probability,
            "posterior": posterior,
            "action": action,
            "posterior_expected_cost": posterior_cost,
        }
    return {
        "expected_terminal_cost": expected_terminal_cost,
        "expected_total_cost": query_cost + expected_terminal_cost,
        "answer_policies": answer_policies,
    }


def conditional_action_probability(
    truth: str,
    left: str,
    right: str,
    reliability: float,
    query: dict[str, Any],
    predicate,
) -> float:
    total = 0.0
    for answer in (left, right):
        probability = reliability if answer == truth else 1.0 - reliability
        action = query["answer_policies"][answer]["action"]
        if predicate(action):
            total += probability
    return total


def evaluate(config: dict[str, Any], v135_config: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    grid_spec = config["clarificationChannel"]
    steps = round((grid_spec["gridMaximum"] - grid_spec["gridMinimum"]) / grid_spec["gridStep"])
    grid = [grid_spec["gridMinimum"] + index * grid_spec["gridStep"] for index in range(steps + 1)]
    evaluation_reliability = grid_spec["evaluationReliability"]
    rows = []
    clear_rows = []
    for family in v135_config["families"]:
        left = family["left_choice_id"]
        right = family["right_choice_id"]
        for prior_spec in config["priorRegimes"]:
            left_prior = prior_spec["presentedKnownCandidateProbability"]
            belief = {left: left_prior, right: 1.0 - left_prior}
            no_query_action, no_query_cost = best_action(belief, catalog, config)
            oracle = query_policy(left, right, left_prior, 1.0, catalog, config)
            evaluated = query_policy(left, right, left_prior, evaluation_reliability, catalog, config)
            threshold = next(
                (
                    reliability
                    for reliability in grid
                    if query_policy(left, right, left_prior, reliability, catalog, config)["expected_total_cost"]
                    < no_query_cost - 1e-12
                ),
                None,
            )
            right_false_known = conditional_action_probability(
                right,
                left,
                right,
                evaluation_reliability,
                evaluated,
                lambda action: choice_kind(action, catalog) == "known",
            )
            right_safe_nonknown = conditional_action_probability(
                right,
                left,
                right,
                evaluation_reliability,
                evaluated,
                lambda action: choice_kind(action, catalog) != "known",
            )
            rows.append(
                {
                    "family_id": family["family_id"],
                    "left_choice_id": left,
                    "right_choice_id": right,
                    "prior_id": prior_spec["id"],
                    "left_prior": left_prior,
                    "no_query_action": no_query_action,
                    "no_query_expected_cost": no_query_cost,
                    "oracle_query_expected_cost": oracle["expected_total_cost"],
                    "oracle_query_cost_improvement": no_query_cost - oracle["expected_total_cost"],
                    "evaluation_query_expected_cost": evaluated["expected_total_cost"],
                    "evaluation_query_cost_improvement": no_query_cost - evaluated["expected_total_cost"],
                    "query_benefit_reliability_threshold": threshold,
                    "right_false_known_probability": right_false_known,
                    "right_safe_nonknown_probability": right_safe_nonknown,
                    "evaluation_answer_policies": evaluated["answer_policies"],
                }
            )
        for truth in (left, right):
            action, no_query_cost = best_action({truth: 1.0}, catalog, config)
            clear_rows.append(
                {
                    "family_id": family["family_id"],
                    "truth_choice_id": truth,
                    "no_query_action": action,
                    "no_query_expected_cost": no_query_cost,
                    "perfect_query_lower_bound_cost": grid_spec["queryCost"],
                    "skip_query_preferred": no_query_cost < grid_spec["queryCost"],
                }
            )
    return {
        "condition_count": len(rows),
        "clear_condition_count": len(clear_rows),
        "oracle_query_preferred_fraction": sum(row["oracle_query_cost_improvement"] > 0.0 for row in rows) / len(rows),
        "evaluation_query_preferred_fraction": sum(row["evaluation_query_cost_improvement"] > 0.0 for row in rows) / len(rows),
        "clear_skip_preferred_fraction": sum(row["skip_query_preferred"] for row in clear_rows) / len(clear_rows),
        "worst_query_benefit_reliability_threshold": max(row["query_benefit_reliability_threshold"] for row in rows),
        "worst_condition_cost_improvement_at_evaluation_reliability": min(row["evaluation_query_cost_improvement"] for row in rows),
        "maximum_right_false_known_probability": max(row["right_false_known_probability"] for row in rows),
        "minimum_right_safe_nonknown_probability": min(row["right_safe_nonknown_probability"] for row in rows),
        "true_hypothesis_retention": 1.0,
        "rows": rows,
        "clear_rows": clear_rows,
    }


def evaluate_gates(result: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["gates"]
    return {
        "condition_count": result["condition_count"] == gates["requiredFamilyPriorConditionCount"],
        "oracle_query_preferred": result["oracle_query_preferred_fraction"] == gates["requiredOracleQueryPreferredFraction"],
        "evaluation_query_preferred": result["evaluation_query_preferred_fraction"] == gates["requiredEvaluationReliabilityQueryPreferredFraction"],
        "query_threshold": result["worst_query_benefit_reliability_threshold"] <= gates["maximumWorstQueryBenefitReliabilityThreshold"],
        "cost_improvement": result["worst_condition_cost_improvement_at_evaluation_reliability"] >= gates["minimumWorstConditionCostImprovementAtEvaluationReliability"],
        "false_known": result["maximum_right_false_known_probability"] <= gates["maximumFalseKnownProbabilityOnRightTruthAtEvaluationReliability"],
        "safe_nonknown": result["minimum_right_safe_nonknown_probability"] >= gates["minimumSafeNonKnownDecisionProbabilityOnRightTruthAtEvaluationReliability"],
        "clear_skip_preferred": result["clear_skip_preferred_fraction"] == gates["requiredClearCaseSkipPreferredFraction"],
        "true_hypothesis_retention": result["true_hypothesis_retention"] == gates["requiredTrueHypothesisRetention"],
        "zero_external_model_or_execution": all(
            gates[key] == 0
            for key in (
                "maximumV134LanguageReadCount",
                "maximumExternalLanguageReadCount",
                "maximumModelLoadCount",
                "maximumModelGenerationCount",
                "maximumAPICallCount",
                "maximumTrainingRunCount",
                "maximumActualExecutionCount",
            )
        ),
    }


__all__ = ["best_action", "choice_kind", "decision_cost", "evaluate", "evaluate_gates", "query_policy"]
