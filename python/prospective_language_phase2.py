"""Pure protocol logic for the prospective language pilot Phase 2.

This module has no ML dependency. It validates the frozen semantic proposal,
derives the controller route, and renders the only prompt the local runner may
use.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


PHASE2_SCHEMA_VERSION = "prospective-language-pilot-v1-phase2-design-1.0.0"
OUTPUT_KEYS = {
    "semantic_state",
    "candidate_goals",
    "missing_evidence",
    "evidence_source",
    "clarification_questions",
    "sandbox_plan",
    "defer_reason",
}


class Phase2ProtocolError(ValueError):
    """Raised when the frozen Phase 2 contract is invalid."""


def validate_phase2_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != PHASE2_SCHEMA_VERSION:
        raise Phase2ProtocolError("Unsupported Phase 2 schema version.")
    participant = config.get("participant", {})
    model = config.get("model", {})
    controller = config.get("controller", {})
    if participant.get("required_record_count") != 16:
        raise Phase2ProtocolError("Phase 2 requires exactly 16 locked requests.")
    if not (
        model.get("temperature") == 0.0
        and model.get("samples_per_request") == 1
        and model.get("retry_count") == 0
        and model.get("enable_thinking") is True
        and model.get("reasoning_effort") == "low"
        and model.get("reasoning_phase_maximum_tokens") == 48
        and model.get("final_phase_maximum_tokens") == 320
        and model.get("mechanically_force_close_thinking_before_final_phase") is True
    ):
        raise Phase2ProtocolError("Frozen model or bounded-reasoning condition changed.")
    if controller.get("real_world_execution_enabled") is not False:
        raise Phase2ProtocolError("Phase 2 must disable real-world execution.")
    if config.get("post_run_rule", {}).get("authorizes_api_fallback") is not False:
        raise Phase2ProtocolError("Phase 2 must not authorize an API fallback.")
    if config.get("prelock_disclosure", {}).get("confirmation_claim_is_prohibited") is not True:
        raise Phase2ProtocolError("The post-exposure development limitation must remain explicit.")


def render_phase2_user_payload(record: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    validate_phase2_config(config)
    payload = {
        "record_id": record["record_id"],
        "initial_request": record["initial_request"],
        "assistant_visible_context": record["assistant_visible_context"],
        "instruction": config["prompt"]["instruction"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _valid_short_strings(value: Any, minimum: int, maximum: int, max_chars: int) -> bool:
    return (
        isinstance(value, list)
        and minimum <= len(value) <= maximum
        and all(isinstance(item, str) and 1 <= len(item.strip()) <= max_chars for item in value)
    )


def parse_semantic_proposal(raw: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly parse one final continuation; never repair or retry it."""

    validate_phase2_config(config)
    try:
        value = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return {"structurally_valid": False, "invalid_reason": "invalid_json"}
    if not isinstance(value, dict) or set(value) != OUTPUT_KEYS:
        return {"structurally_valid": False, "invalid_reason": "wrong_keys"}

    schema = config["schema"]
    semantic_state = value["semantic_state"]
    evidence_source = value["evidence_source"]
    candidate_goals = value["candidate_goals"]
    missing_evidence = value["missing_evidence"]
    questions = value["clarification_questions"]
    sandbox_plan = value["sandbox_plan"]
    defer_reason = value["defer_reason"]

    if semantic_state not in set(schema["allowed_semantic_states"]):
        return {"structurally_valid": False, "invalid_reason": "invalid_semantic_state"}
    if evidence_source not in set(schema["allowed_evidence_sources"]):
        return {"structurally_valid": False, "invalid_reason": "invalid_evidence_source"}
    if not _valid_short_strings(
        candidate_goals, 1, schema["maximum_candidate_goals"], 300
    ):
        return {"structurally_valid": False, "invalid_reason": "invalid_candidate_goals"}
    if not isinstance(missing_evidence, list):
        return {"structurally_valid": False, "invalid_reason": "invalid_missing_evidence"}
    if (
        len(missing_evidence) > schema["maximum_missing_evidence_codes"]
        or len(missing_evidence) != len(set(missing_evidence))
        or any(code not in set(schema["allowed_evidence_codes"]) for code in missing_evidence)
    ):
        return {"structurally_valid": False, "invalid_reason": "invalid_missing_evidence"}
    if not isinstance(questions, list):
        return {"structurally_valid": False, "invalid_reason": "invalid_questions"}
    if sandbox_plan is not None and (
        not isinstance(sandbox_plan, str)
        or not 20 <= len(sandbox_plan.strip()) <= schema["maximum_sandbox_plan_characters"]
    ):
        return {"structurally_valid": False, "invalid_reason": "invalid_sandbox_plan"}
    if defer_reason is not None and (
        not isinstance(defer_reason, str)
        or not 10 <= len(defer_reason.strip()) <= schema["maximum_defer_reason_characters"]
    ):
        return {"structurally_valid": False, "invalid_reason": "invalid_defer_reason"}

    if evidence_source == "NONE_REQUIRED" and missing_evidence:
        return {"structurally_valid": False, "invalid_reason": "evidence_source_mismatch"}
    if evidence_source != "NONE_REQUIRED" and not missing_evidence:
        return {"structurally_valid": False, "invalid_reason": "evidence_source_mismatch"}

    if semantic_state == "ONE_CLEAR_READING" and evidence_source == "NONE_REQUIRED":
        route = "PLAN"
    elif semantic_state == "MULTIPLE_PLAUSIBLE_READINGS" or evidence_source == "USER_CAN_SUPPLY":
        route = "CLARIFY"
    elif evidence_source in {"EXTERNAL_SOURCE_REQUIRED", "NOT_OBTAINABLE"}:
        route = "DEFER"
    else:
        return {"structurally_valid": False, "invalid_reason": "unroutable_semantic_state"}

    if route == "PLAN":
        if questions or sandbox_plan is None or defer_reason is not None:
            return {"structurally_valid": False, "invalid_reason": "plan_payload_mismatch"}
    elif route == "CLARIFY":
        if (
            not 1 <= len(questions) <= schema["maximum_clarification_questions"]
            or sandbox_plan is not None
            or defer_reason is not None
            or any(
                not isinstance(question, str)
                or not question.strip().endswith("?")
                or question.count("?") != 1
                or len(question.strip()) > 400
                for question in questions
            )
        ):
            return {"structurally_valid": False, "invalid_reason": "clarify_payload_mismatch"}
    else:
        if questions or sandbox_plan is not None or defer_reason is None:
            return {"structurally_valid": False, "invalid_reason": "defer_payload_mismatch"}

    return {
        "structurally_valid": True,
        "invalid_reason": None,
        "semantic_state": semantic_state,
        "candidate_goals": [item.strip() for item in candidate_goals],
        "missing_evidence": missing_evidence,
        "evidence_source": evidence_source,
        "clarification_questions": [item.strip() for item in questions],
        "sandbox_plan": sandbox_plan.strip() if sandbox_plan else None,
        "defer_reason": defer_reason.strip() if defer_reason else None,
        "controller_route": route,
    }


def controller_output(parsed: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    validate_phase2_config(config)
    if not parsed.get("structurally_valid"):
        return {
            "route": config["controller"]["invalid_output_route"],
            "clarification_questions": [],
            "sandbox_plan": None,
            "defer_message": config["controller"]["invalid_output_message"],
            "used_safe_fallback": True,
        }
    route = parsed["controller_route"]
    return {
        "route": route,
        "clarification_questions": list(parsed["clarification_questions"]),
        "sandbox_plan": parsed["sandbox_plan"],
        "defer_message": parsed["defer_reason"],
        "used_safe_fallback": False,
    }
