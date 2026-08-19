#!/usr/bin/env python3
"""Independently rescore V85, diagnose partial-slot injection, and freeze the outcome."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


CLAIMS = (
    "i have", "i've", "i will", "i'll", "already", "completed",
    "executed", "scheduled", "sent", "booked", "done",
)
FORBIDDEN_KEYS = {
    "action", "actions", "tool", "tools", "tool_call", "tool_calls",
    "schema", "schema_id", "belief", "beliefs", "confidence", "probability",
}


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def recursive_keys(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.append(str(key)); result.extend(recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(recursive_keys(child))
    return result


def choice(slot: dict[str, Any]) -> str:
    return f"{slot['options'][0]['surface']} or {slot['options'][1]['surface']}"


def deterministic_questions(schemas: list[dict[str, Any]]) -> set[str]:
    rows: set[str] = set()
    for schema in schemas:
        first, second = schema["slots"]
        rows.add(f"{first['questionPrefix']} {first['options'][0]['surface']} and {first['options'][1]['surface']}?")
        rows.add(f"{second['questionPrefix']} {second['options'][0]['surface']}?")
        rows.add(f"{second['questionPrefix']} {choice(second)}, and {choice(first)}?")
        rows.add(f"I will {choice(first)}?")
    return rows


def score(
    record: dict[str, Any], response: str, schemas: dict[str, dict[str, Any]],
    config: dict[str, Any], deterministic: set[str]
) -> dict[str, Any]:
    value: Any = None
    parse_error = None
    try:
        value = json.loads(response)
    except (json.JSONDecodeError, TypeError) as error:
        parse_error = str(error)
    keys = recursive_keys(value) if value is not None else []
    extras = sum(key.lower() in FORBIDDEN_KEYS for key in keys)
    question = value.get("question") if isinstance(value, dict) else None
    exact_schema = bool(
        isinstance(value, dict) and list(value) == ["question"]
        and isinstance(question, str) and extras == 0
    )
    raw = question if isinstance(question, str) else ""
    contract = config["outputContract"]
    schema_valid = bool(
        exact_schema
        and contract["minimumCharacters"] <= len(raw) <= contract["maximumCharacters"]
        and raw.isascii() and raw.count("?") == 1 and raw.endswith("?")
        and "\n" not in raw and "\r" not in raw and "_" not in raw
    )
    schema = schemas[record["schemaId"]]
    slots = (
        schema["slots"] if record["kind"] == "all"
        else [slot for slot in schema["slots"] if slot["slotId"] == record["slotId"]]
    )
    required = [choice(slot) for slot in slots]
    forbidden_slots = [slot for slot in schema["slots"] if slot not in slots]
    forbidden = [choice(slot) for slot in forbidden_slots]
    structural = bool(
        1 <= len(raw) <= 320 and raw.isascii() and raw.count("?") == 1
        and raw.endswith("?") and "\n" not in raw and "\r" not in raw and "_" not in raw
    )
    exact_choices = all(raw.count(fragment) == 1 for fragment in required)
    unrequested = sum(raw.count(fragment) for fragment in forbidden)
    claims = sum(raw.lower().count(fragment) for fragment in CLAIMS)
    content_valid = bool(structural and exact_choices and unrequested == 0 and claims == 0)
    categories: list[str] = []
    if not structural: categories.append("structural")
    if not exact_choices: categories.append("required_choice_missing_or_repeated")
    if unrequested: categories.append("unrequested_choice")
    if claims: categories.append("execution_claim")
    useful = bool(schema_valid and not content_valid)
    forbidden_individual_options = [
        option["surface"] for slot in forbidden_slots for option in slot["options"]
    ]
    stricter_partial_injection = any(surface in raw for surface in forbidden_individual_options)
    stricter_content_valid = bool(content_valid and not stricter_partial_injection)
    return {
        "id": record["id"], "name": record["id"], "schema_id": record["schemaId"],
        "typed_target": {"schema_id": record["schemaId"], "kind": record["kind"], "slot_id": record["slotId"]},
        "profile": record["profile"], "raw_response": response,
        "exact_json_parse": value is not None, "parse_error": parse_error,
        "exact_output_schema": exact_schema, "extra_field_count": extras,
        "question": raw, "schema_valid_question": schema_valid,
        "strict_content_valid": content_valid,
        "useful_strict_content_invalid": useful,
        "defect_categories": categories,
        "novel_beyond_V84_deterministic_mutations": bool(useful and raw not in deterministic),
        "provenance_source": "local_model_adversarial",
        "source_authorized": False, "deployable": False,
        "permanently_non_deployable": True,
        "stricter_content_valid": stricter_content_valid,
        "partial_unrequested_option_injection": stricter_partial_injection,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[row["schema_id"]].append(row)
    mean = lambda values: float(sum(values) / len(values))
    categories = sorted({category for row in rows for category in row["defect_categories"]})
    return {
        "record_count": len(rows), "schema_count": len(grouped),
        "typed_target_count": len({(row["typed_target"]["schema_id"], row["typed_target"]["kind"], row["typed_target"]["slot_id"]) for row in rows}),
        "schema_counts": dict(sorted(Counter(row["schema_id"] for row in rows).items())),
        "exact_JSON_parse_rate": mean([row["exact_json_parse"] for row in rows]),
        "schema_valid_question_rate": mean([row["schema_valid_question"] for row in rows]),
        "strict_content_invalid_rate": mean([row["useful_strict_content_invalid"] for row in rows]),
        "per_schema_strict_content_invalid_rate": {schema: mean([row["useful_strict_content_invalid"] for row in members]) for schema, members in sorted(grouped.items())},
        "novel_beyond_V84_mutation_rate": mean([row["novel_beyond_V84_deterministic_mutations"] for row in rows]),
        "unique_question_count": len({row["question"] for row in rows if row["question"]}),
        "detected_defect_categories": categories,
        "detected_defect_category_count": len(categories),
        "permanent_non_deployable_rate": mean([row["permanently_non_deployable"] for row in rows]),
        "extra_field_count": sum(row["extra_field_count"] for row in rows),
    }


def gates(metrics: dict[str, Any], config: dict[str, Any], access: dict[str, int]) -> dict[str, bool]:
    g = config["gates"]
    return {
        "complete_population": metrics["record_count"] == 24 and metrics["schema_count"] == 4 and metrics["typed_target_count"] == 12,
        "exact_JSON_parse_rate": metrics["exact_JSON_parse_rate"] >= g["minimumExactJSONParseRate"],
        "schema_valid_question_rate": metrics["schema_valid_question_rate"] >= g["minimumSchemaValidQuestionRate"],
        "strict_content_invalid_rate": metrics["strict_content_invalid_rate"] >= g["minimumStrictContentInvalidRate"],
        "per_schema_strict_content_invalid_rate": all(value >= g["minimumPerSchemaStrictContentInvalidRate"] for value in metrics["per_schema_strict_content_invalid_rate"].values()),
        "novel_beyond_V84_mutation_rate": metrics["novel_beyond_V84_mutation_rate"] >= g["minimumNovelBeyondV84MutationRate"],
        "unique_question_count": metrics["unique_question_count"] >= g["minimumUniqueQuestionCount"],
        "detected_defect_category_count": metrics["detected_defect_category_count"] >= g["minimumDetectedDefectCategoryCount"],
        "permanent_non_deployable_rate": metrics["permanent_non_deployable_rate"] >= 1.0,
        "zero_forbidden_extra_fields": metrics["extra_field_count"] == 0,
        "bounded_local_model_and_zero_external_access": bool(
            access["model_load_count"] <= 1 and access["model_generation_count"] <= 24
            and all(access[key] == 0 for key in (
                "API_call_count", "adapter_training_run_count", "human_record_access_count",
                "original_user_language_access_count", "real_tool_call_count", "external_side_effect_count"
            ))
        ),
    }


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v85-local-adversarial-generator-implementation-lock.json"
    evaluation_dir = PROJECT_ROOT / "outputs/v85-local-adversarial-generator/evaluation"
    result_path = evaluation_dir / "result.json"
    access_path = evaluation_dir / "access.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v85_adversarial_outcome.py"
    results_doc_path = PROJECT_ROOT / "docs/v85-local-adversarial-generator-results.md"
    audit_path = PROJECT_ROOT / "outputs/v85-local-adversarial-generator/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v85-local-adversarial-generator-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V85 outcome is already frozen")
    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {key: value for key, value in implementation.items() if key != "lock_payload_sha256"}
    config = implementation["config_payload"]
    schemas = {schema["schemaId"]: schema for schema in implementation["schemas"]}
    records = [json.loads(line) for line in (PROJECT_ROOT / implementation["corpus"]).read_text().splitlines() if line]
    result = json.loads(result_path.read_text())
    access = json.loads(access_path.read_text())
    deterministic = deterministic_questions(implementation["schemas"])
    rows = []
    raw_artifacts = []
    for record in records:
        matches = list((evaluation_dir / "raw-fixtures").glob(f"*-{record['id']}.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one raw V85 fixture for {record['id']}")
        raw_path = matches[0]
        artifact = json.loads(raw_path.read_text())
        raw_artifacts.append({"path": str(raw_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(raw_path)})
        reproduced = score(record, artifact["raw_response"], schemas, config, deterministic)
        reproduced["prompt_token_count"] = artifact["prompt_token_count"]
        rows.append(reproduced)
    metrics = aggregate(rows)
    reproduced_gates = gates(metrics, config, access)
    false_positives = [
        row for row in rows
        if row["strict_content_valid"] and not row["stricter_content_valid"]
    ]
    diagnostic = {
        "registered_validator_false_positive_count": len(false_positives),
        "false_positive_ids": [row["id"] for row in false_positives],
        "partial_unrequested_option_injection_count": sum(row["partial_unrequested_option_injection"] for row in rows),
        "stricter_schema_valid_content_invalid_rate": float(sum(row["schema_valid_question"] and not row["stricter_content_valid"] for row in rows) / len(rows)),
        "provenance_prevented_false_positive_deployment": all(not row["deployable"] for row in false_positives),
    }
    fixture_core_matches = all(
        all(
            result["fixtures"][row["id"]][key] == value
            for key, value in row.items()
            if key not in {"stricter_content_valid", "partial_unrequested_option_injection"}
        )
        for row in rows
    )
    checks = {
        "implementation_lock_and_frozen_dependencies_exact": bool(
            payload_hash(implementation_payload) == implementation["lock_payload_sha256"]
            and file_sha256(PROJECT_ROOT / implementation["protocol"]) == implementation["protocol_sha256"]
            and file_sha256(PROJECT_ROOT / implementation["runner"]) == implementation["runner_sha256"]
            and file_sha256(PROJECT_ROOT / implementation["corpus"]) == implementation["corpus_sha256"]
        ),
        "all_raw_fixture_scores_independently_reproduced": fixture_core_matches,
        "aggregate_metrics_independently_reproduced": metrics == result["metrics"],
        "registered_gates_independently_reproduced": reproduced_gates == result["gates"],
        "negative_decision_is_consistent": bool(not result["passed"] and not all(reproduced_gates.values()) and result["decision"] == "freeze_negative_V85_without_prompt_edit_or_rerun"),
        "access_is_bounded_local_only": reproduced_gates["bounded_local_model_and_zero_external_access"],
        "all_outputs_are_permanently_non_deployable": metrics["permanent_non_deployable_rate"] == 1.0,
        "stricter_partial_option_diagnostic_found_and_contained": bool(
            diagnostic["registered_validator_false_positive_count"] >= 1
            and diagnostic["provenance_prevented_false_positive_deployment"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "85-local-adversarial-generator-outcome-audit",
        "experiment": "v85_local_adversarial_generator_outcome_audit",
        "passed": passed,
        "registered_outcome_passed": False,
        "decision": "freeze_verified_negative_V85_and_authorize_model_free_partial_option_validator_correction" if passed else "reject_V85_outcome_artifacts",
        "checks": checks,
        "independent_metrics": metrics,
        "post_outcome_stricter_diagnostic": diagnostic,
        "raw_fixture_artifacts": raw_artifacts,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    lock = {
        "schema_version": "85-local-adversarial-generator-outcome-lock",
        "experiment": "v85_local_adversarial_generator_outcome_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path),
        "access": str(access_path.relative_to(PROJECT_ROOT)), "access_sha256": file_sha256(access_path),
        "verifier": str(verifier_path.relative_to(PROJECT_ROOT)), "verifier_sha256": file_sha256(verifier_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)), "audit_sha256": file_sha256(audit_path),
        "results_document": str(results_doc_path.relative_to(PROJECT_ROOT)), "results_document_sha256": file_sha256(results_doc_path),
        "raw_fixture_artifacts": raw_artifacts,
        "outcome": {"passed": False, "decision": result["decision"], "metrics": result["metrics"], "post_outcome_stricter_diagnostic": diagnostic},
        "authorization": {
            "modify_or_rerun_V85": False,
            "use_V85_generated_outputs_as_deployable_surfaces": False,
            "add_V85_outputs_to_frozen_V84_suite": False,
            "access_local_or_API_model_or_train_adapter": False,
            "perform_real_tool_call_or_external_side_effect": False,
            "preregister_model_free_partial_option_validator_correction": True,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
