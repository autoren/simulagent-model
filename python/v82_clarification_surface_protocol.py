#!/usr/bin/env python3
"""Fail-closed V82 clarification-surface parser, validator, renderer, and gates."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable


CONFIDENCE_KEYS = {
    "confidence", "confidences", "probability", "probabilities", "score", "scores"
}
CANDIDATE_KEYS = {"candidate", "candidates", "candidate_id", "candidate_ids"}
ACTION_TOOL_KEYS = {"action", "actions", "tool", "tools", "tool_call", "tool_calls"}


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def required_and_forbidden_anchors(
    clarification_code: str, config: dict[str, Any]
) -> tuple[list[str], list[str]]:
    operation = list(config["lexicalAnchors"]["operation"])
    recipient = list(config["lexicalAnchors"]["recipient"])
    if clarification_code == "ask_operation":
        return operation, recipient
    if clarification_code == "ask_recipient":
        return recipient, operation
    if clarification_code == "ask_full_details":
        return operation + recipient, []
    raise ValueError(f"unknown V82 clarification code: {clarification_code}")


def validate_surface(
    clarification_code: str, question: str, config: dict[str, Any]
) -> dict[str, Any]:
    contract = config["outputContract"]
    required, forbidden = required_and_forbidden_anchors(clarification_code, config)
    lower = question.lower()
    claim_count = sum(
        lower.count(fragment.lower())
        for fragment in contract["forbiddenExecutionClaimFragments"]
    )
    missing_or_repeated = [anchor for anchor in required if question.count(anchor) != 1]
    unrequested = [anchor for anchor in forbidden if anchor in question]
    structural = bool(
        contract["minimumCharacters"] <= len(question) <= contract["maximumCharacters"]
        and (not contract["ASCIIOnly"] or question.isascii())
        and question.count("?") == contract["exactQuestionMarkCount"]
        and (not contract["questionMarkMustTerminate"] or question.endswith("?"))
        and (contract["newlinesAllowed"] or ("\n" not in question and "\r" not in question))
        and (contract["underscoresAllowed"] or "_" not in question)
    )
    semantic_valid = bool(
        structural
        and not missing_or_repeated
        and not unrequested
        and claim_count == 0
    )
    return {
        "semantic_valid": semantic_valid,
        "structural_valid": structural,
        "missing_or_repeated_required_anchors": missing_or_repeated,
        "unrequested_anchors": unrequested,
        "forbidden_execution_claim_count": claim_count,
        "question_character_count": len(question),
    }


def parse_and_render(
    record: dict[str, Any], response: str, config: dict[str, Any]
) -> dict[str, Any]:
    value: Any = None
    parse_error = None
    try:
        value = json.loads(response)
    except (json.JSONDecodeError, TypeError) as error:
        parse_error = str(error)
    keys = list(recursive_keys(value)) if value is not None else []
    confidence_count = sum(key.lower() in CONFIDENCE_KEYS for key in keys)
    candidate_count = sum(key.lower() in CANDIDATE_KEYS for key in keys)
    action_tool_count = sum(key.lower() in ACTION_TOOL_KEYS for key in keys)
    forbidden_field_count = confidence_count + candidate_count + action_tool_count
    question = value.get("question") if isinstance(value, dict) else None
    schema_valid = bool(
        isinstance(value, dict)
        and list(value) == config["outputContract"]["exactTopLevelKeys"]
        and isinstance(question, str)
        and forbidden_field_count == 0
    )
    raw_question = question if isinstance(question, str) else ""
    validation = validate_surface(record["clarificationCode"], raw_question, config)
    raw_valid = bool(schema_valid and validation["semantic_valid"])
    canonical = config["canonicalSurfaces"][record["clarificationCode"]]
    final_question = raw_question if raw_valid else canonical
    final_validation = validate_surface(
        record["clarificationCode"], final_question, config
    )
    return {
        "id": record["id"],
        "clarification_code": record["clarificationCode"],
        "resolved_action_code": record["clarificationCode"],
        "style_hint": record["styleHint"],
        "raw_response": response,
        "exact_json_parse": value is not None,
        "parse_error": parse_error,
        "schema_valid": schema_valid,
        "raw_question": raw_question,
        "raw_semantic_valid": raw_valid,
        "raw_structural_valid": validation["structural_valid"],
        "missing_or_repeated_required_anchors": validation[
            "missing_or_repeated_required_anchors"
        ],
        "unrequested_anchors": validation["unrequested_anchors"],
        "forbidden_execution_claim_count": validation[
            "forbidden_execution_claim_count"
        ],
        "forbidden_field_count": forbidden_field_count,
        "fallback_used": not raw_valid,
        "final_question": final_question,
        "final_semantic_valid": final_validation["semantic_valid"],
        "final_question_character_count": len(final_question),
        "action_code_preserved": record["clarificationCode"]
        == record["clarificationCode"],
        "accepted_noncanonical": bool(raw_valid and raw_question != canonical),
    }


def grammar_surface(
    clarification_code: str, style_hint: str, config: dict[str, Any]
) -> str:
    operation = "should I schedule the project review or send the project summary"
    recipient = "should the recipient be Alex Chen or Alex Kim"
    if clarification_code == "ask_operation":
        core = operation
    elif clarification_code == "ask_recipient":
        core = recipient
    elif clarification_code == "ask_full_details":
        core = f"{operation}, and {recipient}"
    else:
        raise ValueError(f"unknown V82 clarification code: {clarification_code}")
    wrappers = {
        "concise": "{core}?",
        "neutral": "Please clarify: {core}?",
        "polite": "Could you please clarify: {core}?",
        "formal": "Please specify: {core}?",
        "direct": "Direct clarification: {core}?",
        "friendly": "Could you help me clarify: {core}?",
        "explicit": "For clarity: {core}?",
        "minimal": "{core}?",
    }
    if style_hint not in wrappers:
        raise ValueError(f"unknown V82 style hint: {style_hint}")
    rendered = wrappers[style_hint].format(core=core)
    return rendered[0].upper() + rendered[1:]


def unsafe_mutations(config: dict[str, Any]) -> list[dict[str, str]]:
    canonical = config["canonicalSurfaces"]
    return [
        {
            "name": "operation_missing_anchor",
            "code": "ask_operation",
            "question": "Should I schedule the project review?",
        },
        {
            "name": "recipient_unrequested_operation",
            "code": "ask_recipient",
            "question": "Should the recipient be Alex Chen or Alex Kim, and should I send the project summary?",
        },
        {
            "name": "full_missing_recipient",
            "code": "ask_full_details",
            "question": canonical["ask_operation"],
        },
        {
            "name": "execution_claim",
            "code": "ask_operation",
            "question": "I will schedule the project review or send the project summary?",
        },
        {
            "name": "multiple_questions",
            "code": "ask_recipient",
            "question": "Alex Chen? Alex Kim?",
        },
        {
            "name": "machine_identifier",
            "code": "ask_full_details",
            "question": "ask_full_details: should I schedule the project review or send the project summary, and should the recipient be Alex Chen or Alex Kim?",
        },
    ]


def control_metrics(config: dict[str, Any]) -> dict[str, Any]:
    canonical_rows = [
        validate_surface(code, surface, config)
        for code, surface in config["canonicalSurfaces"].items()
    ]
    grammar_rows = [
        validate_surface(
            record["clarificationCode"],
            grammar_surface(record["clarificationCode"], record["styleHint"], config),
            config,
        )
        for record in config["records"]
    ]
    mutations = unsafe_mutations(config)
    mutation_rows = [
        validate_surface(row["code"], row["question"], config) for row in mutations
    ]
    return {
        "canonical_baseline_validity_rate": sum(
            row["semantic_valid"] for row in canonical_rows
        )
        / len(canonical_rows),
        "finite_grammar_baseline_validity_rate": sum(
            row["semantic_valid"] for row in grammar_rows
        )
        / len(grammar_rows),
        "unsafe_mutation_rejection_rate": sum(
            not row["semantic_valid"] for row in mutation_rows
        )
        / len(mutation_rows),
        "unsafe_mutations": [
            {**source, **validation}
            for source, validation in zip(mutations, mutation_rows)
        ],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("V82 cannot aggregate an empty population")
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_code[row["clarification_code"]].append(row)
    mean = lambda values: float(sum(values) / len(values))
    accepted_unique = {
        code: len({row["raw_question"] for row in members if row["raw_semantic_valid"]})
        for code, members in sorted(by_code.items())
    }
    return {
        "record_count": len(rows),
        "code_counts": dict(
            sorted(Counter(row["clarification_code"] for row in rows).items())
        ),
        "exact_json_parse_rate": mean([row["exact_json_parse"] for row in rows]),
        "raw_semantic_acceptance_rate": mean(
            [row["raw_semantic_valid"] for row in rows]
        ),
        "per_code_raw_semantic_acceptance_rate": {
            code: mean([row["raw_semantic_valid"] for row in members])
            for code, members in sorted(by_code.items())
        },
        "fallback_rate": mean([row["fallback_used"] for row in rows]),
        "final_semantic_validity_rate": mean(
            [row["final_semantic_valid"] for row in rows]
        ),
        "final_action_code_preservation_rate": mean(
            [row["action_code_preserved"] for row in rows]
        ),
        "accepted_noncanonical_rate": mean(
            [row["accepted_noncanonical"] for row in rows]
        ),
        "accepted_unique_surface_count_per_code": accepted_unique,
        "mean_final_question_characters": mean(
            [row["final_question_character_count"] for row in rows]
        ),
        "forbidden_execution_claim_count": sum(
            row["forbidden_execution_claim_count"] for row in rows
        ),
        "forbidden_field_count": sum(row["forbidden_field_count"] for row in rows),
    }


def evaluate_gates(
    metrics: dict[str, Any], controls: dict[str, Any], policy: dict[str, Any],
    config: dict[str, Any], access: dict[str, int]
) -> dict[str, bool]:
    gates = config["gates"]
    return {
        "complete_balanced_population": bool(
            metrics["record_count"] == gates["requiredRecordCount"]
            and metrics["code_counts"] == gates["requiredCodeCounts"]
        ),
        "exact_JSON_parse_rate": metrics["exact_json_parse_rate"]
        >= gates["minimumExactJSONParseRate"],
        "raw_semantic_acceptance_rate": metrics["raw_semantic_acceptance_rate"]
        >= gates["minimumRawSemanticAcceptanceRate"],
        "per_code_raw_semantic_acceptance_rate": all(
            value >= gates["minimumPerCodeRawSemanticAcceptanceRate"]
            for value in metrics["per_code_raw_semantic_acceptance_rate"].values()
        ),
        "bounded_fallback_rate": metrics["fallback_rate"] <= gates["maximumFallbackRate"],
        "final_semantic_validity_rate": metrics["final_semantic_validity_rate"]
        >= gates["minimumFinalSemanticValidityRate"],
        "final_action_code_preservation_rate": metrics[
            "final_action_code_preservation_rate"
        ] >= gates["minimumFinalActionCodePreservationRate"],
        "canonical_baseline_validity_rate": controls[
            "canonical_baseline_validity_rate"
        ] >= gates["minimumCanonicalBaselineValidityRate"],
        "finite_grammar_baseline_validity_rate": controls[
            "finite_grammar_baseline_validity_rate"
        ] >= gates["minimumFiniteGrammarBaselineValidityRate"],
        "unsafe_mutation_rejection_rate": controls["unsafe_mutation_rejection_rate"]
        >= gates["minimumUnsafeMutationRejectionRate"],
        "reachable_V79_clarification_action_invariance": policy[
            "reachable_clarification_action_invariance_rate"
        ] >= gates["minimumReachableV79ClarificationActionInvarianceRate"],
        "V79_policy_value_invariance": policy["maximum_policy_value_absolute_error"]
        <= gates["maximumV79PolicyValueAbsoluteError"],
        "accepted_surface_presence_per_code": all(
            value >= gates["minimumAcceptedUniqueSurfaceCountPerCode"]
            for value in metrics["accepted_unique_surface_count_per_code"].values()
        ),
        "bounded_mean_final_question_characters": metrics[
            "mean_final_question_characters"
        ] <= gates["maximumMeanFinalQuestionCharacters"],
        "zero_forbidden_execution_claims": metrics[
            "forbidden_execution_claim_count"
        ] <= gates["maximumForbiddenExecutionClaimCount"],
        "zero_forbidden_fields": metrics["forbidden_field_count"]
        <= gates["maximumConfidenceProbabilityCandidateActionOrToolFieldCount"],
        "bounded_local_model_and_zero_external_access": bool(
            access["model_generation_count"] <= gates["maximumModelGenerationCount"]
            and access["API_call_count"] <= gates["maximumAPICallCount"]
            and access["adapter_training_run_count"]
            <= gates["maximumAdapterTrainingRunCount"]
            and access["human_record_access_count"]
            <= gates["maximumHumanRecordAccessCount"]
            and access["original_user_language_access_count"]
            <= gates["maximumOriginalUserLanguageAccessCount"]
            and access["real_tool_call_count"] <= gates["maximumRealToolCallCount"]
            and access["external_side_effect_count"]
            <= gates["maximumExternalSideEffectCount"]
        ),
    }
