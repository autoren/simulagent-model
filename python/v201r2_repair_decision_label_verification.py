from __future__ import annotations

from typing import Any


def evaluate_repair(failed_audit: dict[str, Any], repair_result: dict[str, Any], v201_result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    contract = config["repairContract"]
    false_checks = sorted(key for key, value in failed_audit["checks"].items() if not value)
    checks = {
        "V201r1_failed_only_the_result_label_exactness_check": false_checks == sorted(contract["requiredFalseChecks"]),
        "every_substantive_V201r1_repair_check_passed": repair_result["passed"] and all(repair_result["checks"].values()),
        "stored_label_is_exactly_the_preserved_V201_scientific_decision": repair_result["decision"] == v201_result["decision"] == contract["requiredStoredResultDecision"],
        "intended_repair_stage_label_is_fixed_separately": config["decisionRule"]["ifExactDecisionOverwriteAndEverySubstantiveRepairCheckPasses"] == "freeze_V201r2_serialization_repair_and_preserve_V201_negative_result",
        "V201_scientific_result_remains_negative": not repair_result["qualified"] and not v201_result["qualified"],
        "zero_source_mutation_model_raw_API_and_execution": repair_result["source_artifact_mutation_count"] == repair_result["model_or_policy_rerun_count"] == repair_result["raw_model_response_read_count"] == repair_result["API_call_count"] == repair_result["actual_execution_count"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "stored_source_decision": repair_result["decision"], "repair_stage_decision": config["decisionRule"]["ifExactDecisionOverwriteAndEverySubstantiveRepairCheckPasses"]}


__all__ = ["evaluate_repair"]
