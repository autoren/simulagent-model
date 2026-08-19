from __future__ import annotations

from typing import Any

from v136_controlled_clarification_value import decision_cost


def frechet_all_lower(probabilities: list[float]) -> float:
    return max(0.0, sum(probabilities) - (len(probabilities) - 1))


def candidate_q_values(ambiguous_abstain_lower: float, decidable_exact_lower: float) -> list[float]:
    values = {ambiguous_abstain_lower, 1.0}
    breakpoint = 1.0 - decidable_exact_lower
    if ambiguous_abstain_lower <= breakpoint <= 1.0:
        values.add(breakpoint)
    return sorted(values)


def family_sequential_bounds(
    family: dict[str, Any],
    ambiguous_abstain_lower: float,
    decidable_exact_lower: float,
    catalog: dict[str, Any],
    v136_config: dict[str, Any],
) -> dict[str, float]:
    left = family["left_choice_id"]
    right = family["right_choice_id"]
    candidate = family["presented_candidate_choice_id"]
    abstain = "A00"
    query_cost = v136_config["clarificationChannel"]["queryCost"]
    actions = [row["choice_id"] for row in catalog["choices"]]
    left_wrong_max = max(decision_cost(left, action, catalog, v136_config) for action in actions if action != left)
    right_wrong_max = max(decision_cost(right, action, catalog, v136_config) for action in actions if action != right)
    rows = []
    for query_probability in candidate_q_values(ambiguous_abstain_lower, decidable_exact_lower):
        query_and_left_failure = min(query_probability, 1.0 - decidable_exact_lower)
        query_and_right_correct = max(0.0, query_probability + decidable_exact_lower - 1.0)
        left_sequential = (
            (1.0 - query_probability) * decision_cost(left, candidate, catalog, v136_config)
            + query_probability * query_cost
            + query_and_left_failure * left_wrong_max
        )
        right_sequential = (
            (1.0 - query_probability) * decision_cost(right, candidate, catalog, v136_config)
            + query_probability * query_cost
            + (query_probability - query_and_right_correct) * right_wrong_max
        )
        sequential_mean = (left_sequential + right_sequential) / 2.0
        no_query_mean = (
            query_probability * decision_cost(left, abstain, catalog, v136_config)
            + (1.0 - query_probability) * decision_cost(left, candidate, catalog, v136_config)
            + query_probability * decision_cost(right, abstain, catalog, v136_config)
            + (1.0 - query_probability) * decision_cost(right, candidate, catalog, v136_config)
        ) / 2.0
        right_false_known = min(
            1.0,
            (1.0 - query_probability) + min(query_probability, 1.0 - decidable_exact_lower),
        )
        rows.append(
            {
                "query_probability": query_probability,
                "sequential_mean_cost_upper": sequential_mean,
                "no_query_mean_cost": no_query_mean,
                "improvement_lower": no_query_mean - sequential_mean,
                "right_false_known_upper": right_false_known,
                "right_safe_nonknown_lower": 1.0 - right_false_known,
            }
        )
    return {
        "family_id": family["family_id"],
        "maximum_sequential_mean_cost_upper": max(row["sequential_mean_cost_upper"] for row in rows),
        "minimum_improvement_lower": min(row["improvement_lower"] for row in rows),
        "maximum_right_false_known_upper": max(row["right_false_known_upper"] for row in rows),
        "minimum_right_safe_nonknown_lower": min(row["right_safe_nonknown_lower"] for row in rows),
        "q_rows": rows,
    }


def evaluate_point(
    finalizer_validity: float,
    ambiguity_sensitivity: float,
    decidable_specificity: float,
    proposal_correctness: float,
    config: dict[str, Any],
    v135_config: dict[str, Any],
    catalog: dict[str, Any],
    v136_config: dict[str, Any],
) -> dict[str, Any]:
    decidable_exact_lower = frechet_all_lower(
        [finalizer_validity, decidable_specificity, proposal_correctness]
    )
    ambiguous_failure_upper = min(finalizer_validity, 1.0 - ambiguity_sensitivity)
    ambiguous_abstain_lower = 1.0 - ambiguous_failure_upper
    weights = config["populationWeights"]
    overall_lower = (
        (weights["clear"] + weights["clarified"]) * decidable_exact_lower
        + weights["ambiguous"] * ambiguous_abstain_lower
    )
    group_lower = frechet_all_lower(
        [decidable_exact_lower] * weights["decidableFixturesPerGroup"]
        + [ambiguous_abstain_lower] * weights["ambiguousFixturesPerGroup"]
    )
    decisive_nonknown_false_known_upper = min(
        finalizer_validity,
        decidable_specificity,
        1.0 - proposal_correctness,
    )
    fixture_false_known_upper = max(ambiguous_failure_upper, decisive_nonknown_false_known_upper)
    family_rows = [
        family_sequential_bounds(
            family,
            ambiguous_abstain_lower,
            decidable_exact_lower,
            catalog,
            v136_config,
        )
        for family in v135_config["families"]
    ]
    metrics = {
        "structured_validity": finalizer_validity,
        "overall_exact_accuracy_lower_bound": overall_lower,
        "clear_exact_accuracy_lower_bound": decidable_exact_lower,
        "ambiguous_abstention_lower_bound": ambiguous_abstain_lower,
        "clarified_exact_accuracy_lower_bound": decidable_exact_lower,
        "full_five_stage_group_accuracy_lower_bound": group_lower,
        "false_known_rate_on_nonknown_truths_upper_bound": fixture_false_known_upper,
        "sequential_mean_decision_cost_upper_bound": max(row["maximum_sequential_mean_cost_upper"] for row in family_rows),
        "sequential_improvement_over_no_query_lower_bound": min(row["minimum_improvement_lower"] for row in family_rows),
        "sequential_false_known_on_right_truth_upper_bound": max(row["maximum_right_false_known_upper"] for row in family_rows),
        "sequential_safe_nonknown_on_right_truth_lower_bound": min(row["minimum_right_safe_nonknown_lower"] for row in family_rows),
        "true_hypothesis_retention": 1.0,
        "actual_execution_count": 0,
        "candidate_attracted_fraction_of_errors_bound": None,
    }
    gates = config["qualificationGates"]
    checks = {
        "structured_validity": metrics["structured_validity"] >= gates["minimumStructuredValidity"] - 1e-12,
        "overall_exact_accuracy": metrics["overall_exact_accuracy_lower_bound"] >= gates["minimumOverallExactAccuracyLowerBound"] - 1e-12,
        "clear_exact_accuracy": metrics["clear_exact_accuracy_lower_bound"] >= gates["minimumClearExactAccuracyLowerBound"] - 1e-12,
        "ambiguous_abstention": metrics["ambiguous_abstention_lower_bound"] >= gates["minimumAmbiguousAbstentionLowerBound"] - 1e-12,
        "clarified_exact_accuracy": metrics["clarified_exact_accuracy_lower_bound"] >= gates["minimumClarifiedExactAccuracyLowerBound"] - 1e-12,
        "full_group_accuracy": metrics["full_five_stage_group_accuracy_lower_bound"] >= gates["minimumFullFiveStageGroupAccuracyLowerBound"] - 1e-12,
        "false_known": metrics["false_known_rate_on_nonknown_truths_upper_bound"] <= gates["maximumFalseKnownRateOnNonKnownTruthsUpperBound"] + 1e-12,
        "sequential_cost": metrics["sequential_mean_decision_cost_upper_bound"] <= gates["maximumSequentialMeanDecisionCostUpperBound"] + 1e-12,
        "sequential_improvement": metrics["sequential_improvement_over_no_query_lower_bound"] >= gates["minimumSequentialImprovementOverNoQueryLowerBound"] - 1e-12,
        "sequential_false_known": metrics["sequential_false_known_on_right_truth_upper_bound"] <= gates["maximumSequentialFalseKnownOnRightTruthUpperBound"] + 1e-12,
        "sequential_safe_nonknown": metrics["sequential_safe_nonknown_on_right_truth_lower_bound"] >= gates["minimumSequentialSafeNonKnownOnRightTruthLowerBound"] - 1e-12,
        "true_hypothesis_retention": metrics["true_hypothesis_retention"] == gates["requiredTrueHypothesisRetention"],
        "zero_execution": metrics["actual_execution_count"] == gates["maximumActualExecutionCount"],
    }
    return {
        "reliabilities": {
            "finalizer_validity": finalizer_validity,
            "ambiguity_sensitivity": ambiguity_sensitivity,
            "decidable_specificity": decidable_specificity,
            "proposal_correctness": proposal_correctness,
        },
        "metrics": metrics,
        "gates": checks,
        "qualified_on_bounded_gates": all(checks.values()),
        "candidate_attraction_requires_fresh_empirical_evaluation": True,
        "family_rows": family_rows,
    }


def reliability_grid(config: dict[str, Any]) -> list[float]:
    spec = config["reliabilityGrid"]
    steps = round((spec["maximum"] - spec["minimum"]) / spec["step"])
    return [round(spec["minimum"] + index * spec["step"], 12) for index in range(steps + 1)]


def evaluate(
    config: dict[str, Any],
    v135_config: dict[str, Any],
    catalog: dict[str, Any],
    v136_config: dict[str, Any],
) -> dict[str, Any]:
    grid = reliability_grid(config)
    spec = config["reliabilityGrid"]
    reference_values = {
        "finalizer_validity": spec["referenceFinalizerValidity"],
        "ambiguity_sensitivity": spec["referenceAmbiguitySensitivity"],
        "decidable_specificity": spec["referenceDecidableSpecificity"],
        "proposal_correctness": spec["referenceProposalCorrectness"],
    }
    reference = evaluate_point(**reference_values, config=config, v135_config=v135_config, catalog=catalog, v136_config=v136_config)
    symmetric_threshold = next(
        (
            reliability
            for reliability in grid
            if evaluate_point(
                reliability,
                reliability,
                reliability,
                reliability,
                config,
                v135_config,
                catalog,
                v136_config,
            )["qualified_on_bounded_gates"]
        ),
        None,
    )
    individual_thresholds = {}
    keys = list(reference_values)
    for target in keys:
        threshold = None
        for reliability in grid:
            values = dict(reference_values)
            values[target] = reliability
            point = evaluate_point(**values, config=config, v135_config=v135_config, catalog=catalog, v136_config=v136_config)
            if point["qualified_on_bounded_gates"]:
                threshold = reliability
                break
        individual_thresholds[target] = threshold
    return {
        "reference": reference,
        "symmetric_marginal_reliability_threshold": symmetric_threshold,
        "individual_thresholds_with_other_marginals_at_reference": individual_thresholds,
        "grid_count": len(grid),
        "arbitrary_within_decision_dependence": True,
        "arbitrary_cross_fixture_group_dependence": True,
        "independence_assumption_used": False,
        "candidate_attraction_certified": False,
        "true_hypothesis_retention": 1.0,
        "actual_execution_count": 0,
    }


def evaluate_gates(result: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    gates = config["qualificationGates"]
    access = config["accessGates"]
    return {
        "reference_bounded_gates": result["reference"]["qualified_on_bounded_gates"],
        "symmetric_threshold": result["symmetric_marginal_reliability_threshold"] is not None and result["symmetric_marginal_reliability_threshold"] <= gates["maximumSymmetricMarginalReliabilityThreshold"] + 1e-12,
        "no_independence_assumption": result["arbitrary_within_decision_dependence"] and result["arbitrary_cross_fixture_group_dependence"] and not result["independence_assumption_used"],
        "candidate_attraction_not_overclaimed": not result["candidate_attraction_certified"] and gates["maximumReferenceCandidateAttractedFractionOfErrors"] is None,
        "true_hypothesis_retention": result["true_hypothesis_retention"] == gates["requiredTrueHypothesisRetention"],
        "zero_execution": result["actual_execution_count"] == gates["maximumActualExecutionCount"],
        "zero_external_model_or_execution_access": all(value == 0 for value in access.values()),
    }


__all__ = [
    "candidate_q_values",
    "evaluate",
    "evaluate_gates",
    "evaluate_point",
    "family_sequential_bounds",
    "frechet_all_lower",
    "reliability_grid",
]
