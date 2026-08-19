#!/usr/bin/env python3
"""Hash-select, extract, and seal the frozen V88 external-language corpus once."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def selection_hash(salt: str, service: str, intent: str, record_id: str) -> str:
    return hashlib.sha256(f"{salt}\0{service}\0{intent}\0{record_id}".encode()).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-design-lock.json"
    corpus_path = PROJECT_ROOT / "data/v88-external-intent-candidate/records.jsonl"
    seal_path = PROJECT_ROOT / "data/v88-external-intent-candidate/corpus-seal.json"
    if corpus_path.exists() or seal_path.exists():
        raise RuntimeError("V88 external corpus is already materialized or sealed")
    design = json.loads(design_path.read_text())
    design_payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    if payload_hash(design_payload) != design["lock_payload_sha256"]:
        raise RuntimeError("V88 design lock payload mismatch")
    if not design["authorization"]["select_extract_and_seal_corpus_once_by_frozen_code"]:
        raise RuntimeError("V88 design does not authorize corpus construction")

    parent = json.loads((PROJECT_ROOT / design["parent_V87_outcome_lock"]).read_text())
    if file_sha256(PROJECT_ROOT / parent["inventory"]) != design["source_inventory_sha256"]:
        raise RuntimeError("V87 structural inventory drifted")
    for source in parent["source_files"].values():
        if file_sha256(PROJECT_ROOT / source["local_path"]) != source["local_sha256"]:
            raise RuntimeError("V87 pinned source file drifted")

    config = design["config_payload"]
    inventory = json.loads((PROJECT_ROOT / design["source_inventory"]).read_text())
    schema_payload = json.loads((PROJECT_ROOT / parent["source_files"]["dev/schema.json"]["local_path"]).read_text())
    dialogue_payload = json.loads((PROJECT_ROOT / parent["source_files"]["dev/dialogues_001.json"]["local_path"]).read_text())
    schema_by_service = {service["service_name"]: service for service in schema_payload}
    dialogue_by_id = {dialogue["dialogue_id"]: dialogue for dialogue in dialogue_payload}
    structural_by_id = {row["record_id"]: row for row in inventory["record_index"]}

    selected_rows = []
    salt = config["population"]["selectionSalt"]
    for stratum in config["population"]["strata"]:
        candidates = [
            row for row in inventory["record_index"]
            if row["service"] == stratum["service"] and row["active_intent"] == stratum["activeIntent"]
        ]
        candidates.sort(key=lambda row: (
            selection_hash(salt, stratum["service"], stratum["activeIntent"], row["record_id"]),
            row["record_id"],
        ))
        if len(candidates) < stratum["count"]:
            raise RuntimeError(f"underfilled V88 stratum: {stratum}")
        selected_rows.extend(candidates[:stratum["count"]])
    if len(selected_rows) != config["population"]["recordCount"]:
        raise RuntimeError("V88 selected record count mismatch")
    if len({row["record_id"] for row in selected_rows}) != len(selected_rows):
        raise RuntimeError("V88 selection contains duplicate records")

    records = []
    for sequence, structural in enumerate(selected_rows):
        if structural_by_id[structural["record_id"]] != structural:
            raise RuntimeError("V88 structural record identity mismatch")
        dialogue = dialogue_by_id[structural["dialogue_id"]]
        turn = dialogue["turns"][structural["turn_index"]]
        if turn["speaker"] != "USER" or len(turn["frames"]) != 1:
            raise RuntimeError("V88 selected source turn no longer satisfies the frozen contract")
        frame = turn["frames"][0]
        state = frame["state"]
        if frame["service"] != structural["service"] or state["active_intent"] != structural["active_intent"]:
            raise RuntimeError("V88 source target differs from the frozen structural index")
        if sorted(state["slot_values"]) != structural["state_slot_keys"]:
            raise RuntimeError("V88 source state-slot keys differ from the frozen structural index")
        service_schema = schema_by_service[structural["service"]]
        allowed_intent_ids = [intent["name"] for intent in service_schema["intents"]]
        allowed_slot_ids = [slot["name"] for slot in service_schema["slots"]]
        if state["active_intent"] != "NONE" and state["active_intent"] not in allowed_intent_ids:
            raise RuntimeError("V88 active intent is outside the pinned service schema")
        gold_intents = ["NONE"] if state["active_intent"] == "NONE" else sorted([state["active_intent"], "NONE"])
        history = [
            {"speaker": source_turn["speaker"], "utterance": source_turn["utterance"]}
            for source_turn in dialogue["turns"][: structural["turn_index"] + 1]
        ]
        records.append({
            "id": f"v88-{sequence:03d}-{structural['record_id']}",
            "source_record_id": structural["record_id"],
            "selection_hash": selection_hash(salt, structural["service"], structural["active_intent"], structural["record_id"]),
            "service": structural["service"],
            "dialogue_history": history,
            "schema_context": {
                "service_name": service_schema["service_name"],
                "service_description": service_schema["description"],
                "intents": [
                    {"id": intent["name"], "description": intent["description"]}
                    for intent in service_schema["intents"]
                ],
                "slots": [
                    {"id": slot["name"], "description": slot["description"]}
                    for slot in service_schema["slots"]
                ],
            },
            "allowed_intent_ids": allowed_intent_ids + ["NONE"],
            "allowed_slot_ids": allowed_slot_ids,
            "gold": {
                "active_intent": state["active_intent"],
                "intent_candidates": gold_intents,
                "state_slot_key_candidates": sorted(state["slot_values"]),
            },
            "provenance": {
                "dataset": "Schema-Guided Dialogue Dataset",
                "revision": config["source"]["revision"],
                "license": "CC-BY-SA-4.0",
                "deployable": False,
                "executable": False,
                "manually_inspected": False,
            },
        })

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text("".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
    ))
    stratum_counts = Counter((record["service"], record["gold"]["active_intent"]) for record in records)
    seal = {
        "schema_version": "88-external-intent-candidate-corpus-seal",
        "experiment": "v88_external_intent_candidate_corpus_seal",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "source_outcome_lock": design["parent_V87_outcome_lock"],
        "source_outcome_lock_sha256": file_sha256(PROJECT_ROOT / design["parent_V87_outcome_lock"]),
        "source_inventory": design["source_inventory"],
        "source_inventory_sha256": design["source_inventory_sha256"],
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "record_count": len(records),
        "record_id_sha256": payload_hash({"ids": [record["source_record_id"] for record in records]}),
        "stratum_counts": {f"{key[0]}::{key[1]}": count for key, count in sorted(stratum_counts.items())},
        "license": "CC-BY-SA-4.0",
        "contains_human_language": True,
        "builder_output_contains_human_language": False,
        "manual_utterance_inspection_count": 0,
        "authorization": {
            "modify_rebuild_replace_or_inspect_corpus": False,
            "implement_and_audit_local_runner": True,
            "run_local_model": False,
            "run_API_model_or_train_adapter": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "record_count": len(records),
        "stratum_counts": seal["stratum_counts"],
        "manual_utterance_inspection_count": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
