from __future__ import annotations

from typing import Any


def failed_scientific_checks(audit: dict[str, Any]) -> list[str]:
    return sorted(key for key, value in audit["checks"].items() if not value)


def negative_outcome_matches(
    summary: dict[str, Any], result: dict[str, Any], invariant: dict[str, Any]
) -> bool:
    audit = summary["audit"]
    metrics = summary["metrics"]
    return bool(
        audit["passed"] is invariant["expectedV216ScientificPassed"]
        and result["passed"] is invariant["expectedV216ScientificPassed"]
        and audit["branch"] == result["branch"] == invariant["expectedV216Branch"]
        and audit["decision"] == result["decision"] == invariant["expectedV216Decision"]
        and failed_scientific_checks(audit) == sorted(invariant["expectedFailedScientificChecks"])
        and metrics["older_term_count"] == invariant["expectedOlderTermCount"]
        and metrics["newer_term_count"] == invariant["expectedNewerTermCount"]
        and metrics["eligible_record_count"] == invariant["expectedEligibleRecordCount"]
        and metrics["development_group_count"] == invariant["expectedDevelopmentGroupCount"]
        and metrics["protected_group_count"] == invariant["expectedProtectedGroupCount"]
    )

