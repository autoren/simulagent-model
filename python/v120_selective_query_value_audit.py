from __future__ import annotations

from typing import Any


def run_audit(parent_result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    summary = parent_result["summary"]
    baseline = summary["baseline_mean_regret"]
    fixed_cost = config["decomposition"]["fixedClarificationCost"]
    reliability = config["requiredSlice"]["reliability"]
    planner = config["requiredSlice"]["planner"]
    rows = []
    for prior, by_reliability in summary["conditions"].items():
        for rho in config["requiredSlice"]["correlations"]:
            metrics = by_reliability[reliability][rho][planner]
            total = metrics["mean_regret_including_clarification"]
            decision = total - fixed_cost
            excess = max(0.0, total - baseline)
            rows.append({
                "condition_id": f"{prior}@{rho}",
                "total_regret": total,
                "decision_regret_excluding_query_cost": decision,
                "decision_value_relative_to_baseline": baseline - decision,
                "maximum_affordable_average_query_cost": baseline - decision,
                "regret_excess_over_baseline": excess,
                "minimum_zero_loss_skip_fraction": excess / fixed_cost,
                "fails_baseline": total > baseline,
            })
    failing = [row for row in rows if row["fails_baseline"]]
    expected = config["requiredSlice"]["expectedFailingConditionIds"]
    parent_gates = summary["outcome_gates"]
    gates = config["outcomeGates"]
    required_parent_passes = [
        "aware_known_exact_every_prior_and_required_correlation",
        "aware_unsupported_every_prior_and_required_correlation",
        "aware_false_known_every_prior_and_required_correlation",
        "misspecified_regret_at_050_every_prior",
        "misspecified_false_known_at_050_every_prior",
        "perfect_channel_mean_regret_every_prior",
        "rho_one_stress_reported", "true_hypothesis_retention", "zero_actual_execution",
    ]
    checks = {
        "failing_conditions_exact": sorted(row["condition_id"] for row in failing) == sorted(expected) and len(failing) == gates["requiredFailingConditionCount"],
        "regret_excess_is_narrow": max(row["regret_excess_over_baseline"] for row in failing) <= gates["maximumRegretExcessOverBaseline"],
        "required_zero_loss_skip_fraction_is_small": max(row["minimum_zero_loss_skip_fraction"] for row in failing) <= gates["maximumMinimumZeroLossSkipFraction"],
        "decision_regret_below_baseline_every_failing_condition": all(row["decision_regret_excluding_query_cost"] < baseline for row in failing),
        "parent_accuracy_safety_control_and_misspecification_gates_pass": all(parent_gates[key] for key in required_parent_passes),
        "aggregate_only_no_record_reads_or_emissions": gates["maximumIndividualRecordReadCount"] == 0 and gates["maximumIndividualRecordEmissionCount"] == 0,
        "zero_actual_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {
        "baseline_mean_regret": baseline, "fixed_clarification_cost": fixed_cost,
        "required_condition_count": len(rows), "failing_condition_count": len(failing),
        "conditions": rows, "failing_conditions": failing, "outcome_gates": checks,
        "outcome_pass": passed,
        "decision": config["decisionRule"]["ifAllOutcomeAndAccessGatesPass"] if passed else config["decisionRule"]["otherwise"],
        "individual_record_read_count": 0, "individual_record_emission_count": 0,
        "actual_execution_count": 0,
    }


__all__ = ["run_audit"]
