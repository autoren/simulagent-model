from __future__ import annotations
from typing import Any


def build_inventory(config: dict[str, Any]) -> dict[str, Any]:
    included = config["includedSignals"]
    excluded = config["excludedSignals"]
    semantic_independent = sorted(
        {
            row["semanticFamily"]
            for row in included
            if row["llmDependence"] == "none"
            and row["semanticFamily"] != "presence_control"
        }
    )
    gates = config["outcomeGates"]
    checks = {
        "included_signal_count": len(included) >= gates["minimumIncludedSignalCount"],
        "excluded_leakage_signal_count": len(excluded) >= gates["minimumExcludedLeakageSignalCount"],
        "all_included_available_prequery": all(row["availableBeforeClarification"] for row in included),
        "mutability_recorded": gates["requireMutabilityRecorded"] and all(bool(row.get("mutability")) for row in included),
        "llm_independent_semantic_family_exact": semantic_independent == [gates["requiredLLMIndependentSemanticFamily"]] and len(semantic_independent) >= gates["minimumLLMIndependentSemanticFamilyCount"],
        "computational_independence_not_promoted_to_statistical_independence": gates["requireNoStatisticalIndependenceClaim"] and all(row["independenceClaim"] != "statistically_independent" for row in included),
        "no_utility_calibration_or_trigger_claim": gates["requireNoUtilityCalibrationOrTriggerClaim"],
        "aggregate_definition_only": gates["maximumIndividualRecordReadCount"] == 0,
        "zero_actual_execution": gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {
        "included_signal_count": len(included),
        "excluded_signal_count": len(excluded),
        "llm_independent_semantic_families": semantic_independent,
        "included_signals": included,
        "excluded_signals": excluded,
        "outcome_gates": checks,
        "outcome_pass": passed,
        "decision": config["decisionRule"]["ifAllOutcomeAndAccessGatesPass"] if passed else config["decisionRule"]["otherwise"],
        "signal_evaluated_count": 0,
        "trigger_fitted_count": 0,
        "individual_record_read_count": 0,
        "actual_execution_count": 0,
    }


__all__ = ["build_inventory"]
