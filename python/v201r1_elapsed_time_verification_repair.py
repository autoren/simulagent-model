from __future__ import annotations

from typing import Any

from v201_local_menu_presentation_robustness import evaluate_access_gates


def evaluate_repair(
    failed_audit: dict[str, Any],
    result: dict[str, Any],
    persisted_summary: dict[str, Any],
    access: dict[str, Any],
    v201_config: dict[str, Any],
    repair_config: dict[str, Any],
) -> dict[str, Any]:
    rebuilt_summary = failed_audit["summary"]
    different_keys = sorted(
        key for key in set(rebuilt_summary) | set(persisted_summary)
        if rebuilt_summary.get(key) != persisted_summary.get(key)
    )
    false_checks = sorted(key for key, value in failed_audit["checks"].items() if not value)
    delta = rebuilt_summary["elapsed_seconds"] - persisted_summary["elapsed_seconds"]
    repaired = dict(rebuilt_summary)
    repaired["elapsed_seconds"] = persisted_summary["elapsed_seconds"]
    access_checks = evaluate_access_gates(access, v201_config)
    qualified = bool(persisted_summary["qualified"] and all(access_checks.values()))
    expected_decision = v201_config["decisionRule"][
        "ifEveryPerVariantQualificationAndAccessGatePasses" if qualified else "otherwise"
    ]
    contract = repair_config["repairContract"]
    checks = {
        "original_verifier_failed_only_the_two_expected_exact_comparisons": false_checks == sorted(contract["requiredFalseChecks"]),
        "summary_diff_is_exactly_the_allowed_volatile_field": different_keys == contract["allowedTopLevelSummaryDifferenceKeys"],
        "elapsed_delta_is_small_monotone_and_exactly_explained": bool(
            0.0 <= delta <= contract["maximumAllowedElapsedSecondsDelta"]
            and rebuilt_summary["elapsed_seconds"] == access["elapsed_seconds"]
        ),
        "single_field_comparison_repair_makes_summary_exact": repaired == persisted_summary,
        "result_derives_exactly_from_persisted_summary_and_final_access": bool(
            result["summary"] == persisted_summary and result["access_gates"] == access_checks
            and result["access_gates_passed"] == all(access_checks.values())
            and result["qualification_gates_passed"] == persisted_summary["qualified"]
            and result["qualified"] == qualified and result["decision"] == expected_decision
        ),
        "scientific_result_remains_negative_for_only_frozen_Jaccard_gates": bool(
            not persisted_summary["qualified"]
            and all(
                [key for key, value in row["qualification_gates"].items() if not value] == ["top3_contract_set_jaccard"]
                for row in persisted_summary["variants"]
            )
        ),
        "access_authority_and_execution_remain_closed": bool(
            access["retry_count"] == 0 and access["persisted_raw_response_count"] == 0
            and access["manual_raw_response_inspection_count"] == 0 and access["protected_language_read_count"] == 0
            and access["API_call_count"] == 0 and access["training_run_count"] == 0
            and access["ontology_registration_count"] == 0 and access["trusted_state_mutation_count"] == 0
            and access["real_service_call_count"] == 0 and access["external_side_effect_count"] == 0
            and access["actual_execution_count"] == 0
        ),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "different_summary_keys": different_keys, "elapsed_seconds_delta": delta,
        "qualified": qualified, "decision": expected_decision,
    }


__all__ = ["evaluate_repair"]
