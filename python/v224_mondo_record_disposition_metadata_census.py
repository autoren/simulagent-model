from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import re
from typing import Any


SUBSTANTIVE = (
    "ACCEPTED_NEW",
    "EXISTING_OR_DUPLICATE",
    "INSUFFICIENT_OR_CLARIFY",
    "UNSUPPORTED_OR_OUT_OF_SCOPE",
)


def parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def normalize_label(value: str) -> str:
    return " ".join(value.casefold().split())


def human_actor(actor: dict[str, Any] | None, requester: str | None, config: dict[str, Any]) -> bool:
    if not actor or actor.get("__typename") != config["humanActorContract"]["requiredTypename"]:
        return False
    login = actor.get("login")
    if not login or login == requester:
        return False
    lowered = login.casefold()
    return not any(pattern.casefold() in lowered for pattern in config["humanActorContract"]["excludedLoginPatterns"])


def labels_at_cutoff(record: dict[str, Any], cutoff: datetime) -> set[str]:
    labels: set[str] = set()
    events = sorted(record["timelineItems"]["nodes"], key=lambda row: row.get("createdAt", ""))
    for event in events:
        when = parse_time(event.get("createdAt"))
        if when is None or when > cutoff:
            continue
        name = normalize_label((event.get("label") or {}).get("name", ""))
        if event["__typename"] == "LabeledEvent" and name:
            labels.add(name)
        elif event["__typename"] == "UnlabeledEvent" and name:
            labels.discard(name)
    return labels


def latest_event(record: dict[str, Any], typename: str, label: str | None, cutoff: datetime) -> dict[str, Any] | None:
    eligible = []
    for event in record["timelineItems"]["nodes"]:
        when = parse_time(event.get("createdAt"))
        if event.get("__typename") != typename or when is None or when > cutoff:
            continue
        if label is not None and normalize_label((event.get("label") or {}).get("name", "")) != label:
            continue
        eligible.append(event)
    return max(eligible, key=lambda row: row["createdAt"], default=None)


def has_nested_overflow(record: dict[str, Any]) -> bool:
    if record["labels"]["pageInfo"]["hasNextPage"] or record["timelineItems"]["pageInfo"]["hasNextPage"]:
        return True
    pulls = record["closedByPullRequestsReferences"]
    if pulls["pageInfo"]["hasNextPage"]:
        return True
    return any(
        pull["reviews"]["pageInfo"]["hasNextPage"] or pull["files"]["pageInfo"]["hasNextPage"]
        for pull in pulls["nodes"]
    )


def preliminary_disposition(record: dict[str, Any], config: dict[str, Any]) -> tuple[str, list[str]]:
    cutoff = parse_time(config["source"]["eventCutoff"])
    assert cutoff is not None
    issue_number = record.get("number")
    requester_actor = record.get("author") or {}
    requester = requester_actor.get("login")
    reasons: list[str] = []
    if issue_number in config["priorExposure"]["excludedIssueNumbers"]:
        return "AMBIGUOUS_EXCLUDE", ["PRIOR_EXPOSURE"]
    if requester_actor.get("__typename") != "User" or not requester:
        return "AMBIGUOUS_EXCLUDE", ["NON_HUMAN_OR_MISSING_REQUESTER"]
    if any(pattern.casefold() in requester.casefold() for pattern in config["humanActorContract"]["excludedLoginPatterns"]):
        return "AMBIGUOUS_EXCLUDE", ["AUTOMATED_REQUESTER"]
    if has_nested_overflow(record):
        return "AMBIGUOUS_EXCLUDE", ["NESTED_PAGINATION_OVERFLOW"]
    labels = labels_at_cutoff(record, cutoff)
    if config["labels"]["request"] not in labels:
        return "AMBIGUOUS_EXCLUDE", ["NOT_FINAL_NEW_TERM_REQUEST"]
    if labels & set(config["labels"]["aiOrAutomationExclusions"]):
        return "AMBIGUOUS_EXCLUDE", ["AI_OR_AUTOMATION_MARKER"]
    accepted_event = latest_event(record, "LabeledEvent", config["labels"]["accepted"], cutoff)
    clarification_event = latest_event(record, "LabeledEvent", config["labels"]["clarification"], cutoff)
    scope_event = latest_event(record, "LabeledEvent", config["labels"]["outOfScope"], cutoff)
    duplicate_event = latest_event(record, "MarkedAsDuplicateEvent", None, cutoff)
    reopened = latest_event(record, "ReopenedEvent", None, cutoff)
    closed = latest_event(record, "ClosedEvent", None, cutoff)
    signatures = {
        "ACCEPTED_NEW": accepted_event is not None and bool(record["closedByPullRequestsReferences"]["nodes"]),
        "EXISTING_OR_DUPLICATE": duplicate_event is not None,
        "INSUFFICIENT_OR_CLARIFY": clarification_event is not None and config["labels"]["clarification"] in labels,
        "UNSUPPORTED_OR_OUT_OF_SCOPE": scope_event is not None and config["labels"]["outOfScope"] in labels,
    }
    active = [name for name, enabled in signatures.items() if enabled]
    if len(active) != 1:
        return "AMBIGUOUS_EXCLUDE", ["NO_SINGLE_SUBSTANTIVE_SIGNATURE"]
    outcome = active[0]
    event = {
        "ACCEPTED_NEW": accepted_event,
        "EXISTING_OR_DUPLICATE": duplicate_event,
        "INSUFFICIENT_OR_CLARIFY": clarification_event,
        "UNSUPPORTED_OR_OUT_OF_SCOPE": scope_event,
    }[outcome]
    if not human_actor((event or {}).get("actor"), requester, config):
        return "AMBIGUOUS_EXCLUDE", ["ADJUDICATOR_NOT_INDEPENDENT_HUMAN"]
    if reopened and closed and reopened["createdAt"] > closed["createdAt"]:
        return "AMBIGUOUS_EXCLUDE", ["REOPENED_AFTER_TERMINAL_EVENT"]
    if outcome == "INSUFFICIENT_OR_CLARIFY":
        edited = parse_time(record.get("lastEditedAt"))
        event_time = parse_time((event or {}).get("createdAt"))
        if record.get("state") != "OPEN":
            reasons.append("CLARIFICATION_NOT_OPEN_AT_CUTOFF")
        if edited and event_time and edited > event_time:
            reasons.append("BODY_EDITED_AFTER_CLARIFICATION")
    if outcome == "UNSUPPORTED_OR_OUT_OF_SCOPE":
        edited = parse_time(record.get("lastEditedAt"))
        event_time = parse_time((event or {}).get("createdAt"))
        if record.get("state") != "CLOSED" or closed is None or not human_actor(closed.get("actor"), requester, config):
            reasons.append("OUT_OF_SCOPE_NOT_TERMINALLY_HUMAN_CLOSED")
        if edited and event_time and edited > event_time:
            reasons.append("BODY_EDITED_AFTER_SCOPE_LABEL")
    if outcome == "EXISTING_OR_DUPLICATE":
        canonical = (duplicate_event or {}).get("canonical") or {}
        duplicate = (duplicate_event or {}).get("duplicate") or {}
        if canonical.get("__typename") != "Issue" or duplicate.get("number") != issue_number:
            reasons.append("DUPLICATE_EVENT_LACKS_EXACT_CANONICAL_ISSUE")
    return (outcome, reasons) if not reasons else ("AMBIGUOUS_EXCLUDE", reasons)


def selection_key(issue_number: int, config: dict[str, Any]) -> str:
    raw = f'{config["samplingContract"]["seed"]}:{issue_number}'.encode()
    return hashlib.sha256(raw).hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def mondo_ids_from_patch(patch: str) -> list[str]:
    return sorted(set(re.findall(r"(?m)^\+id:\s*(MONDO:\d+)\s*$", patch)))


def score_census(
    records: list[dict[str, Any]],
    preliminary: dict[str, Any],
    deep_rows: list[dict[str, Any]],
    query_manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    recomputed: dict[str, list[int]] = {name: [] for name in (*SUBSTANTIVE, "AMBIGUOUS_EXCLUDE")}
    exclusion_counts: dict[str, int] = {}
    for record in records:
        disposition, reasons = preliminary_disposition(record, config)
        recomputed[disposition].append(record["number"])
        for reason in reasons:
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
    for values in recomputed.values():
        values.sort()
    selected_expected = {
        name: sorted(
            recomputed[name], key=lambda number: selection_key(number, config)
        )[: config["samplingContract"]["maximumDeepAuditRecordsPerStratum"]]
        for name in SUBSTANTIVE
    }
    final_counts = {
        name: sum(row.get("retained") is True and row.get("disposition") == name for row in deep_rows)
        for name in SUBSTANTIVE
    }
    human_rates = {
        name: (
            sum(
                row.get("human_independence") is True
                for row in deep_rows
                if row.get("retained") is True and row.get("disposition") == name
            )
            / final_counts[name]
            if final_counts[name]
            else 0.0
        )
        for name in SUBSTANTIVE
    }
    provenance_rates = {
        name: (
            sum(
                row.get("provenance_complete") is True
                for row in deep_rows
                if row.get("retained") is True and row.get("disposition") == name
            )
            / final_counts[name]
            if final_counts[name]
            else 0.0
        )
        for name in SUBSTANTIVE
    }
    record_numbers = [record["number"] for record in records]
    selected_actual = preliminary.get("selected_issue_numbers", {})
    minimum = config["gates"]["minimumFinalRetainedRecordsPerStratum"]
    metrics = {
        "expected_search_slice_count": config["source"]["expectedSearchSliceCount"],
        "recorded_search_slice_count": len(query_manifest.get("search_slices", [])),
        "search_slice_accounting_rate": (
            len(query_manifest.get("search_slices", [])) / config["source"]["expectedSearchSliceCount"]
        ),
        "record_row_count": len(records),
        "unique_record_count": len(set(record_numbers)),
        "record_deduplication_rate": len(set(record_numbers)) / len(record_numbers) if record_numbers else 1.0,
        "forbidden_selected_field_count": query_manifest.get("forbidden_selected_field_count", -1),
        "excluded_issue_leak_count": len(set(record_numbers) & set(config["priorExposure"]["excludedIssueNumbers"])),
        "task_language_persistence_count": query_manifest.get("task_language_persistence_count", -1),
        "preliminary_counts": {name: len(recomputed[name]) for name in SUBSTANTIVE},
        "ambiguous_exclude_count": len(recomputed["AMBIGUOUS_EXCLUDE"]),
        "preliminary_exclusion_reason_counts": exclusion_counts,
        "preliminary_artifact_reconstructs_exactly": bool(
            preliminary.get("counts") == {name: len(recomputed[name]) for name in (*SUBSTANTIVE, "AMBIGUOUS_EXCLUDE")}
            and selected_actual == selected_expected
        ),
        "preliminary_gate_passed": all(
            len(recomputed[name]) >= config["samplingContract"]["minimumPreliminaryRecordsPerStratum"]
            for name in SUBSTANTIVE
        ),
        "deep_audit_row_count": len(deep_rows),
        "final_retained_counts": final_counts,
        "human_independence_rates": human_rates,
        "provenance_completeness_rates": provenance_rates,
        "all_strata_meet_final_gate": all(
            final_counts[name] >= minimum
            and human_rates[name] == config["gates"]["requiredHumanIndependenceRatePerFinalStratum"]
            and provenance_rates[name] == config["gates"]["requiredProvenanceCompletenessRatePerFinalStratum"]
            for name in SUBSTANTIVE
        ),
    }
    metrics["finite_metrics"] = finite(metrics)
    return metrics


def audit_census(metrics: dict[str, Any], access: dict[str, int], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    checks = {
        "search_frame_and_record_accounting_are_exact": bool(
            metrics["recorded_search_slice_count"] == gates["requiredSearchSliceCount"]
            and metrics["search_slice_accounting_rate"] == gates["requiredSearchSliceAccountingRate"]
            and metrics["record_deduplication_rate"] == gates["requiredRecordDeduplicationRate"]
            and metrics["excluded_issue_leak_count"] == gates["requiredExcludedIssueLeakCount"]
        ),
        "language_firewall_is_exact": bool(
            metrics["forbidden_selected_field_count"] == gates["requiredForbiddenSelectedFieldCount"]
            and metrics["task_language_persistence_count"] == gates["requiredTaskLanguagePersistenceCount"]
        ),
        "preliminary_artifact_reconstructs_exactly": metrics["preliminary_artifact_reconstructs_exactly"],
        "metrics_are_finite": metrics["finite_metrics"] == gates["requiredFiniteMetrics"],
    }
    access_gates = config["accessGates"]
    access_checks = {
        "one_formal_census_run": access["formal_census_run_count"] == access_gates["requiredFormalCensusRunCount"],
        "language_model_protected_and_effect_boundaries_are_zero": all(
            access[key] <= access_gates[gate]
            for key, gate in {
                "task_record_title_read_count": "maximumTaskRecordTitleReadCount",
                "task_record_body_read_count": "maximumTaskRecordBodyReadCount",
                "comment_or_review_text_read_count": "maximumCommentOrReviewTextReadCount",
                "protected_research_record_read_count": "maximumProtectedResearchRecordReadCount",
                "model_load_count": "maximumModelLoadCount",
                "model_generation_count": "maximumModelGenerationCount",
                "model_api_call_count": "maximumModelAPICallCount",
                "training_run_count": "maximumTrainingRunCount",
                "ontology_registration_count": "maximumOntologyRegistrationCount",
                "trusted_state_mutation_count": "maximumTrustedStateMutationCount",
                "service_action_count": "maximumServiceActionCount",
                "external_side_effect_count": "maximumExternalSideEffectCount",
                "actual_execution_count": "maximumActualExecutionCount",
            }.items()
        ),
    }
    integrity_passed = all(checks.values()) and all(access_checks.values())
    eligible = integrity_passed and metrics["all_strata_meet_final_gate"]
    return {
        "passed": integrity_passed,
        "branch": "V225_LANGUAGE_DESIGN_ELIGIBLE" if eligible else "MONDO_B2C_EXTERNAL_VALIDATION_INSUFFICIENT",
        "decision": (
            config["decisionRule"]["ifEverySubstantiveStratumPasses"]
            if eligible
            else config["decisionRule"]["otherwise"]
        ),
        "checks": checks,
        "access_checks": access_checks,
    }
