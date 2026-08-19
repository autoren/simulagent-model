from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def run_audit(
    inventory: dict[str, Any], excluded_populations: list[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    excluded = {
        row["candidate_id"]
        for population in excluded_populations
        for row in population["selected_population"]
    }
    spec = config["candidateRequirement"]
    counts: Counter[str] = Counter()
    scenarios: dict[str, set[str]] = defaultdict(set)
    for row in inventory["candidate_index"]:
        if row["partition"] == spec["sourcePartition"] and row["candidate_id"] not in excluded:
            counts[row["class_label"]] += 1
            scenarios[row["class_label"]].add(row["scenario"])
    exact_counts = {label: counts[label] for label in spec["classes"]}
    scenario_counts = {label: len(scenarios[label]) for label in spec["classes"]}
    maximum_balanced = min(exact_counts.values())
    requirement_pass = bool(
        maximum_balanced >= spec["minimumRecordCountPerClass"]
        and scenario_counts["novel_valid"] >= spec["minimumNovelValidScenarioCoverage"]
    )
    gates = config["outcomeGates"]
    checks = {
        "remaining_counts_exact": exact_counts == gates["requireExactRemainingCounts"],
        "maximum_balanced_count_exact": maximum_balanced == gates["requiredMaximumBalancedRecordCountPerClass"],
        "novel_scenario_coverage_exact": scenario_counts["novel_valid"] == gates["requiredRemainingNovelScenarioCoverage"],
        "candidate_requirement_fails": gates["requireCandidateRequirementFailure"] and not requirement_pass,
        "aggregate_only": gates["maximumIndividualCandidateEmissionCount"] == 0,
        "zero_language": gates["maximumLanguageReadCount"] == 0,
        "zero_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {
        "remaining_counts": exact_counts,
        "remaining_scenario_counts": scenario_counts,
        "maximum_balanced_record_count_per_class": maximum_balanced,
        "candidate_requirement_pass": requirement_pass,
        "outcome_gates": checks,
        "outcome_pass": passed,
        "decision": config["decisionRule"]["ifAllOutcomeAndAccessGatesPass"] if passed else config["decisionRule"]["otherwise"],
        "individual_candidate_emission_count": 0,
        "language_read_count": 0,
        "actual_execution_count": 0,
    }


__all__ = ["run_audit"]
