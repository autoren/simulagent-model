from __future__ import annotations

from typing import Any

from v207r1_agentabstain_shadow_feasibility import audit_feasibility


def evaluate_repair(
    source_lock: dict[str, Any],
    failed_audit: dict[str, Any],
    summary: dict[str, Any],
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rebuilt = audit_feasibility(
        summary,
        source_lock["scientific_config_payload"],
        source_lock["config_payload"],
    )
    false_checks = sorted(key for key, value in failed_audit["checks"].items() if not value)
    expected_decision = source_lock["config_payload"]["decisionRule"]["ifOriginalScientificGateFails"]
    checks = {
        "failed_audit_has_only_preregistered_bookkeeping_failure": false_checks == sorted(config["repairContract"]["requiredFalseChecks"]),
        "failed_audit_substantive_reconstruction_passed": all(
            failed_audit["checks"][key]
            for key in failed_audit["checks"]
            if key not in config["repairContract"]["requiredFalseChecks"]
        ),
        "stored_summary_audit_passes": rebuilt["passed"],
        "stored_scientific_result_is_required_negative": bool(
            summary["scientific_feasibility_passed"] is config["repairContract"]["requiredScientificFeasibility"]
            and result["scientific_feasibility_passed"] is config["repairContract"]["requiredScientificFeasibility"]
        ),
        "stored_transport_result_is_required_positive": bool(
            summary["transport_integrity_passed"] is config["repairContract"]["requiredTransportIntegrity"]
            and result["transport_integrity_passed"] is config["repairContract"]["requiredTransportIntegrity"]
        ),
        "stored_result_reconstructs_from_summary": bool(
            result["passed"] == rebuilt["passed"]
            and result["summary"] == summary
            and result["checks"] == rebuilt["checks"]
            and result["access_checks"] == rebuilt["access_checks"]
            and result["decision"] == expected_decision
            and not result["authorization"]["preregister_separate_deterministic_text_extraction_only"]
        ),
        "source_artifacts_and_evaluation_are_not_mutable": bool(
            not config["repairContract"]["sourceArtifactsMayBeModified"]
            and not config["repairContract"]["networkMetadataMayBeReadAgain"]
            and not config["repairContract"]["scientificEvaluationOrModelMayBeRerun"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "rebuilt_audit": rebuilt}


__all__ = ["evaluate_repair"]
