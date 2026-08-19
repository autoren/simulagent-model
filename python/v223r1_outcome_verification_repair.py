from __future__ import annotations

from typing import Any


def failed_checks(audit: dict[str, Any]) -> list[str]:
    return sorted(name for name, passed in audit.get("checks", {}).items() if passed is not True)


def positive_outcome_matches(
    summary: dict[str, Any], result: dict[str, Any], invariant: dict[str, Any]
) -> bool:
    metrics = summary.get("metrics", {})
    audit = summary.get("audit", {})
    return bool(
        audit.get("passed") is invariant["expectedV223ScientificPassed"]
        and audit.get("branch") == invariant["expectedV223Branch"]
        and result.get("passed") is True
        and result.get("branch") == invariant["expectedV223Branch"]
        and result.get("selected_source_specific_candidate_ids")
        == invariant["expectedSelectedCandidateIds"]
        and metrics.get("source_unit_count") == invariant["expectedSourceUnitCount"]
        and metrics.get("frozen_url_attempt_count") == invariant["expectedFrozenUrlAttemptCount"]
        and metrics.get("successful_url_count") == invariant["expectedSuccessfulUrlCount"]
        and metrics.get("eligible_source_specific_candidate_count")
        == invariant["expectedEligibleCandidateCount"]
        and metrics.get("formal_task_record_body_read_count")
        == invariant["expectedFormalTaskRecordBodyReadCount"]
    )

