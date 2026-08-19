from __future__ import annotations

from typing import Any


def run_audit(parent_result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    parent = parent_result["summary"]
    cost = parent["fixed_clarification_cost"]
    conditions = []
    for row in parent["failing_conditions"]:
        average_value = row["decision_value_relative_to_baseline"]
        excess = row["regret_excess_over_baseline"]
        skip_bounds = []
        for fraction in config["skipFractions"]:
            maximum_skipped_value = cost - excess / fraction
            skip_bounds.append({
                "skip_fraction": fraction,
                "maximum_average_query_value_in_skipped_subset": maximum_skipped_value,
                "maximum_skipped_value_as_fraction_of_population_average": maximum_skipped_value / average_value,
            })
        conditions.append({
            "condition_id": row["condition_id"],
            "population_average_query_value": average_value,
            "minimum_queried_subset_value": cost,
            "minimum_queried_value_lift_ratio": cost / average_value,
            "minimum_queried_value_lift_percent": 100.0 * (cost / average_value - 1.0),
            "skip_value_bounds": skip_bounds,
        })
    gates = config["outcomeGates"]
    hard = [row for row in conditions if row["condition_id"] in {"uniform_safe_universe@0.25", "strong_candidate@0.50"}]
    checks = {
        "failing_condition_count_exact": len(conditions) == gates["requiredFailingConditionCount"],
        "queried_value_lift_is_finite_but_positive": max(row["minimum_queried_value_lift_ratio"] for row in conditions) <= gates["maximumMinimumQueriedValueLift"] and all(row["minimum_queried_value_lift_ratio"] > 1.0 for row in conditions),
        "five_percent_skip_requires_low_value_cases_in_hard_conditions": max(next(item["maximum_average_query_value_in_skipped_subset"] for item in row["skip_value_bounds"] if item["skip_fraction"] == 0.05) for row in hard) <= gates["maximumFivePercentSkipValueForHardConditions"],
        "all_preregistered_skip_bounds_nonnegative": all(sum(item["maximum_average_query_value_in_skipped_subset"] >= 0.0 for item in row["skip_value_bounds"]) >= gates["minimumNonnegativeSkipFractionCountEveryCondition"] for row in conditions),
        "aggregate_metrics_explicitly_insufficient_to_certify_trigger": gates["requireAggregateMetricsInsufficientToCertifyAnyTrigger"],
        "aggregate_only": gates["maximumIndividualRecordReadCount"] == 0 and gates["maximumIndividualRecordEmissionCount"] == 0,
        "zero_actual_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {
        "query_cost": cost, "condition_count": len(conditions), "conditions": conditions,
        "outcome_gates": checks, "outcome_pass": passed,
        "decision": config["decisionRule"]["ifAllOutcomeAndAccessGatesPass"] if passed else config["decisionRule"]["otherwise"],
        "trigger_certified": False,
        "required_future_evidence": "paired fresh pre-query signal, query-value, safety, and regret validation",
        "individual_record_read_count": 0, "individual_record_emission_count": 0,
        "actual_execution_count": 0,
    }


__all__ = ["run_audit"]
