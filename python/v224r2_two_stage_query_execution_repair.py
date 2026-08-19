from __future__ import annotations

from typing import Any

from v224_mondo_record_disposition_metadata_census import (
    human_actor,
    labels_at_cutoff,
    latest_event,
    normalize_label,
    parse_time,
)


def preliminary_disposition_thin(
    record: dict[str, Any], config: dict[str, Any]
) -> tuple[str, list[str]]:
    cutoff = parse_time(config["source"]["eventCutoff"])
    assert cutoff is not None
    number = record.get("number")
    requester_actor = record.get("author") or {}
    requester = requester_actor.get("login")
    if number in config["priorExposure"]["excludedIssueNumbers"]:
        return "AMBIGUOUS_EXCLUDE", ["PRIOR_EXPOSURE"]
    if requester_actor.get("__typename") != "User" or not requester:
        return "AMBIGUOUS_EXCLUDE", ["NON_HUMAN_OR_MISSING_REQUESTER"]
    lowered = requester.casefold()
    if any(pattern.casefold() in lowered for pattern in config["humanActorContract"]["excludedLoginPatterns"]):
        return "AMBIGUOUS_EXCLUDE", ["AUTOMATED_REQUESTER"]
    if record["labels"]["pageInfo"]["hasNextPage"] or record["timelineItems"]["pageInfo"]["hasNextPage"]:
        return "AMBIGUOUS_EXCLUDE", ["NESTED_PAGINATION_OVERFLOW"]
    labels = labels_at_cutoff(record, cutoff)
    if config["labels"]["request"] not in labels:
        return "AMBIGUOUS_EXCLUDE", ["NOT_FINAL_NEW_TERM_REQUEST"]
    if labels & set(config["labels"]["aiOrAutomationExclusions"]):
        return "AMBIGUOUS_EXCLUDE", ["AI_OR_AUTOMATION_MARKER"]
    accepted = latest_event(record, "LabeledEvent", config["labels"]["accepted"], cutoff)
    clarify = latest_event(record, "LabeledEvent", config["labels"]["clarification"], cutoff)
    scope = latest_event(record, "LabeledEvent", config["labels"]["outOfScope"], cutoff)
    duplicate = latest_event(record, "MarkedAsDuplicateEvent", None, cutoff)
    reopened = latest_event(record, "ReopenedEvent", None, cutoff)
    closed = latest_event(record, "ClosedEvent", None, cutoff)
    signatures = {
        "ACCEPTED_NEW": accepted is not None and record["closedByPullRequestsReferences"]["totalCount"] > 0,
        "EXISTING_OR_DUPLICATE": duplicate is not None,
        "INSUFFICIENT_OR_CLARIFY": clarify is not None and config["labels"]["clarification"] in labels,
        "UNSUPPORTED_OR_OUT_OF_SCOPE": scope is not None and config["labels"]["outOfScope"] in labels,
    }
    active = [name for name, enabled in signatures.items() if enabled]
    if len(active) != 1:
        return "AMBIGUOUS_EXCLUDE", ["NO_SINGLE_SUBSTANTIVE_SIGNATURE"]
    outcome = active[0]
    event = {
        "ACCEPTED_NEW": accepted,
        "EXISTING_OR_DUPLICATE": duplicate,
        "INSUFFICIENT_OR_CLARIFY": clarify,
        "UNSUPPORTED_OR_OUT_OF_SCOPE": scope,
    }[outcome]
    if not human_actor((event or {}).get("actor"), requester, config):
        return "AMBIGUOUS_EXCLUDE", ["ADJUDICATOR_NOT_INDEPENDENT_HUMAN"]
    if reopened and closed and reopened["createdAt"] > closed["createdAt"]:
        return "AMBIGUOUS_EXCLUDE", ["REOPENED_AFTER_TERMINAL_EVENT"]
    reasons: list[str] = []
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
        canonical = (duplicate or {}).get("canonical") or {}
        duplicate_ref = (duplicate or {}).get("duplicate") or {}
        if canonical.get("__typename") != "Issue" or duplicate_ref.get("number") != number:
            reasons.append("DUPLICATE_EVENT_LACKS_EXACT_CANONICAL_ISSUE")
    return (outcome, []) if not reasons else ("AMBIGUOUS_EXCLUDE", reasons)


def install_thin_scorer(core_module: Any) -> None:
    core_module.preliminary_disposition = preliminary_disposition_thin

