from __future__ import annotations

from collections import Counter
from io import BytesIO
import hashlib
import json
import re
import tarfile
import unicodedata
from typing import Any


def normalize_name(value: str) -> str:
    return "".join(character for character in unicodedata.normalize("NFKC", value).casefold() if character.isalnum())


def normalize_description(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def definition_record(service: str, domain: str, intent: dict[str, Any]) -> dict[str, Any]:
    name = normalize_name(intent["name"]); description = normalize_description(intent["description"])
    required = tuple(sorted(intent.get("required_slots", []))); optional = tuple(sorted(intent.get("optional_slots", [])))
    signature = {"name": name, "description": description, "required_slots": required, "optional_slots": optional}
    return {
        "intent_id": f"{service}::{intent['name']}", "service": service, "domain": domain,
        "normalized_name": name, "normalized_description_sha256": hashlib.sha256(description.encode()).hexdigest(),
        "required_slots": required, "optional_slots": optional,
        "full_signature_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
        "_description": description,
    }


def _domain(service: str) -> str:
    match = re.fullmatch(r"(.*)_\d+", service)
    if not match: raise ValueError(f"invalid SGD service name: {service}")
    return match.group(1).lower()


def read_schema_definitions(archive_bytes: bytes, config: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    root = f"dstc8-schema-guided-dialogue-{config['schemaAudit']['archiveRevision']}"
    output: dict[str, dict[str, dict[str, Any]]] = {}
    with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as archive:
        files = [member for member in archive.getmembers() if member.isfile()]
        for partition in config["schemaAudit"]["schemaPartitions"]:
            matches = [member for member in files if member.name == f"{root}/{partition}/schema.json"]
            if len(matches) != 1: raise ValueError(f"expected one {partition} schema")
            handle = archive.extractfile(matches[0])
            if handle is None: raise ValueError("unreadable schema")
            services = json.loads(handle.read())
            definitions: dict[str, dict[str, Any]] = {}
            for service in services:
                name = service["service_name"]
                for intent in service["intents"]:
                    record = definition_record(name, _domain(name), intent)
                    if record["intent_id"] in definitions: raise ValueError("duplicate schema intent")
                    definitions[record["intent_id"]] = record
            output[partition] = definitions
    return output


def compare_definitions(novel: dict[str, Any], known: dict[str, Any]) -> dict[str, bool]:
    return {
        "exact_name": novel["normalized_name"] == known["normalized_name"],
        "exact_description": novel["_description"] == known["_description"],
        "exact_slot_signature": novel["required_slots"] == known["required_slots"] and novel["optional_slots"] == known["optional_slots"],
        "exact_full_signature": novel["full_signature_sha256"] == known["full_signature_sha256"],
    }


def run_audit(
    archive_bytes: bytes, catalog: dict[str, Any], population: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    schemas = read_schema_definitions(archive_bytes, config)
    known_choices = [row for row in catalog["choices"] if row["kind"] == "KNOWN"]
    novel_choices = [row for row in catalog["choices"] if row["kind"] == "NOVEL_COMPOSITE"]
    known_defs = []
    for choice in known_choices:
        matches = [schemas["train"].get(choice["intent_id"])]
        if matches[0] is None: raise ValueError("declared definition missing")
        known_defs.append(matches[0])
    selected_novel = [row for row in population["fixtures"] if row["class_label"] == "novel_valid"]
    record_counts = Counter(f"{row['service']}::{row['intent']}" for row in selected_novel)
    choice_by_domain = {row["domain"]: row["choice_id"] for row in novel_choices}
    pair_rows = []
    for intent_id, record_count in sorted(record_counts.items()):
        novel = schemas["test"].get(intent_id)
        if novel is None: raise ValueError("novel definition missing")
        comparisons = [(known["intent_id"], compare_definitions(novel, known)) for known in known_defs]
        collision = {key: any(values[key] for _, values in comparisons) for key in ("exact_name", "exact_description", "exact_slot_signature", "exact_full_signature")}
        pair_rows.append({
            "novel_intent_id": intent_id, "novel_choice_id": choice_by_domain[novel["domain"]],
            "selected_record_count": record_count, "normalized_name": novel["normalized_name"],
            "normalized_description_sha256": novel["normalized_description_sha256"],
            "full_signature_sha256": novel["full_signature_sha256"],
            "collision": collision,
            "colliding_known_intent_ids": {
                key: sorted(known_id for known_id, values in comparisons if values[key])
                for key in collision
            },
        })
    total = sum(row["selected_record_count"] for row in pair_rows)
    def weighted(key: str) -> float:
        return sum(row["selected_record_count"] for row in pair_rows if row["collision"][key]) / total
    choice_rows = {}
    for choice in novel_choices:
        members = [row for row in pair_rows if row["novel_choice_id"] == choice["choice_id"]]
        choice_rows[choice["choice_id"]] = {
            "member_definition_count": len(members),
            "selected_record_count": sum(row["selected_record_count"] for row in members),
            "has_any_exact_name_collision": any(row["collision"]["exact_name"] for row in members),
            "all_members_exact_name_collide": bool(members) and all(row["collision"]["exact_name"] for row in members),
        }
    summary = {
        "declared_known_definition_count": len(known_defs), "novel_definition_count": len(pair_rows),
        "selected_novel_record_count": total, "novel_composite_choice_count": len(novel_choices),
        "selected_record_collision_fractions": {
            key: weighted(key) for key in ("exact_name", "exact_description", "exact_slot_signature", "exact_full_signature")
        },
        "novel_choices_with_no_exact_name_collision": sum(not row["has_any_exact_name_collision"] for row in choice_rows.values()),
        "entirely_name_colliding_novel_choice_count": sum(row["all_members_exact_name_collide"] for row in choice_rows.values()),
        "choice_summary": choice_rows, "pair_summary": pair_rows,
        "raw_description_emission_count": 0, "utterance_or_slot_value_read_count": 0,
        "manual_semantic_judgment_count": 0, "model_load_count": 0, "model_generation_count": 0,
        "actual_execution_count": 0,
    }
    gates = config["identifiabilityGates"]
    checks = {
        "selected_novel_record_count": total == gates["requiredSelectedNovelRecordCount"],
        "novel_composite_choice_count": len(novel_choices) == gates["requiredNovelCompositeChoiceCount"],
        "novel_choices_have_no_exact_name_collision": summary["novel_choices_with_no_exact_name_collision"] >= gates["minimumNovelChoicesWithNoExactNameCollision"],
        "selected_name_collision_fraction": summary["selected_record_collision_fractions"]["exact_name"] <= gates["maximumSelectedNovelRecordExactNameCollisionFraction"],
        "selected_full_signature_collision_fraction": summary["selected_record_collision_fractions"]["exact_full_signature"] <= gates["maximumSelectedNovelRecordFullSignatureCollisionFraction"],
        "no_entirely_name_colliding_novel_choice": summary["entirely_name_colliding_novel_choice_count"] <= gates["maximumEntirelyNameCollidingNovelChoiceCount"],
        "every_novel_definition_present": gates["requireEveryNovelDefinitionPresentExactlyOnce"] and len({row["novel_intent_id"] for row in pair_rows}) == len(pair_rows),
        "declared_known_definition_count": len(known_defs) == gates["requiredDeclaredKnownDefinitionCount"],
        "zero_manual_judgment": summary["manual_semantic_judgment_count"] <= gates["maximumManualSemanticJudgmentCount"],
        "zero_raw_description_emission": summary["raw_description_emission_count"] <= gates["maximumRawDescriptionEmissionCount"],
        "zero_utterance_or_slot_value_read": summary["utterance_or_slot_value_read_count"] <= gates["maximumUtteranceOrSlotValueReadCount"],
        "zero_model_and_execution": summary["model_load_count"] == gates["maximumModelLoadCount"] == 0 and summary["model_generation_count"] == gates["maximumModelGenerationCount"] == 0 and summary["actual_execution_count"] == gates["maximumActualExecutionCount"] == 0,
    }
    passed = all(checks.values())
    return {"summary": summary, "identifiability_gates": checks, "identifiability_pass": passed, "decision": config["decisionRule"]["ifEveryIdentifiabilityAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"]}


__all__ = ["compare_definitions", "definition_record", "normalize_description", "normalize_name", "read_schema_definitions", "run_audit"]
