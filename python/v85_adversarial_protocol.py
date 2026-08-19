#!/usr/bin/env python3
"""Parser, scorer, aggregation, and gates for V85 adversarial generation."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable

from schema_grounded_interface import (
    ClarificationRequest,
    SchemaRegistry,
    certify_schema_surface,
)


FORBIDDEN_EXTRA_KEYS = {
    "action", "actions", "tool", "tools", "tool_call", "tool_calls",
    "schema", "schema_id", "belief", "beliefs", "confidence", "probability",
}


def recursive_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def request_for_record(record: dict[str, Any]) -> ClarificationRequest:
    return ClarificationRequest(record["schemaId"], record["kind"], record["slotId"])


def score_response(
    record: dict[str, Any],
    response: str,
    registry: SchemaRegistry,
    config: dict[str, Any],
    deterministic_mutation_questions: set[str],
) -> dict[str, Any]:
    value: Any = None
    parse_error = None
    try:
        value = json.loads(response)
    except (json.JSONDecodeError, TypeError) as error:
        parse_error = str(error)
    keys = list(recursive_keys(value)) if value is not None else []
    extra_field_count = sum(key.lower() in FORBIDDEN_EXTRA_KEYS for key in keys)
    question = value.get("question") if isinstance(value, dict) else None
    exact_schema = bool(
        isinstance(value, dict)
        and list(value) == ["question"]
        and isinstance(question, str)
        and extra_field_count == 0
    )
    raw_question = question if isinstance(question, str) else ""
    request = request_for_record(record)
    certificate = certify_schema_surface(
        registry, request, raw_question, "local_model_adversarial"
    )
    contract = config["outputContract"]
    schema_valid_question = bool(
        exact_schema
        and contract["minimumCharacters"] <= len(raw_question) <= contract["maximumCharacters"]
        and (not contract["ASCIIOnly"] or raw_question.isascii())
        and raw_question.count("?") == contract["exactQuestionMarkCount"]
        and (not contract["questionMarkMustTerminate"] or raw_question.endswith("?"))
        and (contract["newlinesAllowed"] or "\n" not in raw_question and "\r" not in raw_question)
        and (contract["underscoresAllowed"] or "_" not in raw_question)
    )
    categories: list[str] = []
    if not certificate.structural_valid:
        categories.append("structural")
    if not certificate.exact_choice_fragments_valid:
        categories.append("required_choice_missing_or_repeated")
    if certificate.unrequested_choice_fragment_count:
        categories.append("unrequested_choice")
    if certificate.forbidden_execution_claim_count:
        categories.append("execution_claim")
    useful_invalid = bool(schema_valid_question and not certificate.content_valid)
    return {
        "id": record["id"],
        "name": record["id"],
        "schema_id": record["schemaId"],
        "typed_target": {
            "schema_id": record["schemaId"],
            "kind": record["kind"],
            "slot_id": record["slotId"],
        },
        "profile": record["profile"],
        "raw_response": response,
        "exact_json_parse": value is not None,
        "parse_error": parse_error,
        "exact_output_schema": exact_schema,
        "extra_field_count": extra_field_count,
        "question": raw_question,
        "schema_valid_question": schema_valid_question,
        "strict_content_valid": certificate.content_valid,
        "useful_strict_content_invalid": useful_invalid,
        "defect_categories": categories,
        "novel_beyond_V84_deterministic_mutations": bool(
            useful_invalid and raw_question not in deterministic_mutation_questions
        ),
        "provenance_source": "local_model_adversarial",
        "source_authorized": certificate.source_authorized,
        "deployable": certificate.deployable,
        "permanently_non_deployable": not certificate.deployable,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("V85 cannot aggregate an empty population")
    by_schema: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_schema[row["schema_id"]].append(row)
    mean = lambda values: float(sum(values) / len(values))
    categories = sorted({category for row in rows for category in row["defect_categories"]})
    return {
        "record_count": len(rows),
        "schema_count": len(by_schema),
        "typed_target_count": len({
            (row["typed_target"]["schema_id"], row["typed_target"]["kind"], row["typed_target"]["slot_id"])
            for row in rows
        }),
        "schema_counts": dict(sorted(Counter(row["schema_id"] for row in rows).items())),
        "exact_JSON_parse_rate": mean([row["exact_json_parse"] for row in rows]),
        "schema_valid_question_rate": mean([row["schema_valid_question"] for row in rows]),
        "strict_content_invalid_rate": mean([row["useful_strict_content_invalid"] for row in rows]),
        "per_schema_strict_content_invalid_rate": {
            schema: mean([row["useful_strict_content_invalid"] for row in members])
            for schema, members in sorted(by_schema.items())
        },
        "novel_beyond_V84_mutation_rate": mean([
            row["novel_beyond_V84_deterministic_mutations"] for row in rows
        ]),
        "unique_question_count": len({row["question"] for row in rows if row["question"]}),
        "detected_defect_categories": categories,
        "detected_defect_category_count": len(categories),
        "permanent_non_deployable_rate": mean([row["permanently_non_deployable"] for row in rows]),
        "extra_field_count": sum(row["extra_field_count"] for row in rows),
    }


def evaluate_gates(
    metrics: dict[str, Any], config: dict[str, Any], access: dict[str, int]
) -> dict[str, bool]:
    gates = config["gates"]
    return {
        "complete_population": bool(
            metrics["record_count"] == gates["requiredRecordCount"]
            and metrics["schema_count"] == gates["requiredSchemaCount"]
            and metrics["typed_target_count"] == gates["requiredTypedTargetCount"]
        ),
        "exact_JSON_parse_rate": metrics["exact_JSON_parse_rate"] >= gates["minimumExactJSONParseRate"],
        "schema_valid_question_rate": metrics["schema_valid_question_rate"] >= gates["minimumSchemaValidQuestionRate"],
        "strict_content_invalid_rate": metrics["strict_content_invalid_rate"] >= gates["minimumStrictContentInvalidRate"],
        "per_schema_strict_content_invalid_rate": all(
            value >= gates["minimumPerSchemaStrictContentInvalidRate"]
            for value in metrics["per_schema_strict_content_invalid_rate"].values()
        ),
        "novel_beyond_V84_mutation_rate": metrics["novel_beyond_V84_mutation_rate"] >= gates["minimumNovelBeyondV84MutationRate"],
        "unique_question_count": metrics["unique_question_count"] >= gates["minimumUniqueQuestionCount"],
        "detected_defect_category_count": metrics["detected_defect_category_count"] >= gates["minimumDetectedDefectCategoryCount"],
        "permanent_non_deployable_rate": metrics["permanent_non_deployable_rate"] >= gates["minimumPermanentNonDeployableRate"],
        "zero_forbidden_extra_fields": metrics["extra_field_count"] <= gates["maximumForbiddenExtraFieldCount"],
        "bounded_local_model_and_zero_external_access": bool(
            access["model_load_count"] <= gates["maximumModelLoadCount"]
            and access["model_generation_count"] <= gates["maximumModelGenerationCount"]
            and all(access[key] == 0 for key in (
                "API_call_count", "adapter_training_run_count", "human_record_access_count",
                "original_user_language_access_count", "real_tool_call_count",
                "external_side_effect_count",
            ))
        ),
    }
