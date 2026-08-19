#!/usr/bin/env python3
"""Select, extract, and seal the fresh V91 rank-only population exactly once."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def selection_hash(salt: str, service: str, intent: str, record_id: str) -> str:
    return hashlib.sha256(
        f"{salt}\0{service}\0{intent}\0{record_id}".encode()
    ).hexdigest()


def structurally_select(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_dialogues: set[str] = set()
    salt = config["population"]["selectionSalt"]
    for stratum in config["population"]["strata"]:
        candidates = [
            row
            for row in rows
            if row["service"] == stratum["service"]
            and row["active_intent"] == stratum["activeIntent"]
        ]
        candidates.sort(
            key=lambda row: (
                selection_hash(
                    salt,
                    stratum["service"],
                    stratum["activeIntent"],
                    row["record_id"],
                ),
                row["record_id"],
            )
        )
        chosen = []
        for row in candidates:
            if row["dialogue_id"] in used_dialogues:
                continue
            chosen.append(row)
            used_dialogues.add(row["dialogue_id"])
            if len(chosen) == stratum["count"]:
                break
        if len(chosen) != stratum["count"]:
            raise RuntimeError(f"underfilled unique-dialogue V91 stratum: {stratum}")
        selected.extend(chosen)
    return selected


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v91-rank-only-design-lock.json"
    corpus_path = PROJECT_ROOT / "data/v91-rank-only/records.jsonl"
    seal_path = PROJECT_ROOT / "data/v91-rank-only/corpus-seal.json"
    if corpus_path.exists() or seal_path.exists():
        raise RuntimeError("V91 corpus is already materialized or sealed")
    design = json.loads(design_path.read_text())
    design_payload = {
        key: value for key, value in design.items() if key != "lock_payload_sha256"
    }
    if payload_hash(design_payload) != design["lock_payload_sha256"]:
        raise RuntimeError("V91 design lock payload mismatch")
    if not design["authorization"]["select_extract_and_seal_corpus_once"]:
        raise RuntimeError("V91 design does not authorize corpus construction")
    for key in (
        "design",
        "source_outcome_lock",
        "source_inventory",
        "parent_model_decision_lock",
        "planner_outcome_lock",
        "plan",
        "protocol",
        "tests",
        "auditor",
        "builder",
    ):
        if file_sha256(PROJECT_ROOT / design[key]) != design[f"{key}_sha256"]:
            raise RuntimeError(f"V91 locked dependency drifted: {key}")

    source_lock = json.loads((PROJECT_ROOT / design["source_outcome_lock"]).read_text())
    source_file = PROJECT_ROOT / source_lock["source_file"]
    schema_lock = json.loads(
        (PROJECT_ROOT / "configs/v87-external-language-source-outcome-lock.json").read_text()
    )
    schema_path = PROJECT_ROOT / schema_lock["source_files"]["dev/schema.json"][
        "local_path"
    ]
    if file_sha256(source_file) != source_lock["source_file_sha256"]:
        raise RuntimeError("V91 source shard drifted")
    if (
        file_sha256(schema_path)
        != schema_lock["source_files"]["dev/schema.json"]["local_sha256"]
    ):
        raise RuntimeError("V91 schema dependency drifted")

    config = design["config_payload"]
    inventory = json.loads((PROJECT_ROOT / design["source_inventory"]).read_text())
    schema_payload = json.loads(schema_path.read_text())
    dialogue_payload = json.loads(source_file.read_text())
    schema_by_service = {item["service_name"]: item for item in schema_payload}
    dialogue_by_id = {item["dialogue_id"]: item for item in dialogue_payload}
    structural_by_id = {
        item["record_id"]: item for item in inventory["record_index"]
    }
    selected = structurally_select(config, inventory["record_index"])
    expected_count = config["population"]["recordCount"]
    if (
        len(selected) != expected_count
        or len({row["dialogue_id"] for row in selected}) != expected_count
    ):
        raise RuntimeError("V91 selected population cardinality or uniqueness drifted")

    records = []
    salt = config["population"]["selectionSalt"]
    for sequence, structural in enumerate(selected):
        if structural_by_id[structural["record_id"]] != structural:
            raise RuntimeError("V91 structural identity mismatch")
        dialogue = dialogue_by_id[structural["dialogue_id"]]
        turn = dialogue["turns"][structural["turn_index"]]
        if turn["speaker"] != "USER" or len(turn["frames"]) != 1:
            raise RuntimeError("V91 selected turn violates frozen structure")
        frame = turn["frames"][0]
        state = frame["state"]
        if (
            frame["service"] != structural["service"]
            or state["active_intent"] != structural["active_intent"]
            or sorted(state["slot_values"]) != structural["state_slot_keys"]
        ):
            raise RuntimeError("V91 source target drifted")
        service_schema = schema_by_service[structural["service"]]
        allowed = [item["name"] for item in service_schema["intents"]] + ["NONE"]
        history = [
            {"speaker": source_turn["speaker"], "utterance": source_turn["utterance"]}
            for source_turn in dialogue["turns"][: structural["turn_index"] + 1]
        ]
        records.append(
            {
                "id": f"v91-{sequence:03d}-{structural['record_id']}",
                "source_record_id": structural["record_id"],
                "source_dialogue_id": structural["dialogue_id"],
                "selection_hash": selection_hash(
                    salt,
                    structural["service"],
                    structural["active_intent"],
                    structural["record_id"],
                ),
                "service": structural["service"],
                "dialogue_history": history,
                "schema_context": {
                    "service_name": service_schema["service_name"],
                    "service_description": service_schema["description"],
                    "intents": [
                        {"id": item["name"], "description": item["description"]}
                        for item in service_schema["intents"]
                    ],
                },
                "allowed_intent_ids": allowed,
                "gold_intent": state["active_intent"],
                "authoritative_state_fingerprint": payload_hash(state["slot_values"]),
                "provenance": {
                    "dataset": config["source"]["dataset"],
                    "revision": config["source"]["revision"],
                    "source_shard": config["source"]["shard"],
                    "license": config["source"]["license"],
                    "deployable": False,
                    "executable": False,
                    "manually_inspected": False,
                },
            }
        )

    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        )
    )
    strata = Counter((record["service"], record["gold_intent"]) for record in records)
    seal = {
        "schema_version": "91-rank-only-corpus-seal",
        "experiment": "v91_rank_only_corpus_seal",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "source_outcome_lock": design["source_outcome_lock"],
        "source_outcome_lock_sha256": file_sha256(
            PROJECT_ROOT / design["source_outcome_lock"]
        ),
        "source_inventory": design["source_inventory"],
        "source_inventory_sha256": design["source_inventory_sha256"],
        "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "record_count": len(records),
        "dialogue_count": len({record["source_dialogue_id"] for record in records}),
        "record_id_sha256": payload_hash(
            {"ids": [record["source_record_id"] for record in records]}
        ),
        "stratum_counts": {
            f"{key[0]}::{key[1]}": value for key, value in sorted(strata.items())
        },
        "service_allowed_intent_ids": {
            service: next(
                record["allowed_intent_ids"]
                for record in records
                if record["service"] == service
            )
            for service in sorted({record["service"] for record in records})
        },
        "license": "CC-BY-SA-4.0",
        "contains_human_language": True,
        "builder_output_contains_human_language": False,
        "manual_utterance_inspection_count": 0,
        "new_model_weight_download_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "authorization": {
            "modify_rebuild_replace_or_inspect_corpus": False,
            "implement_and_audit_ranker_runner_and_invariance_harness": True,
            "load_or_run_local_model_before_implementation_lock": False,
            "run_API_model_or_train_adapter": False,
            "prune_or_early_stop_search": False,
            "grant_model_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False
        },
    }
    seal["lock_payload_sha256"] = payload_hash(seal)
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "corpus": str(corpus_path.relative_to(PROJECT_ROOT)),
                "corpus_sha256": file_sha256(corpus_path),
                "seal": str(seal_path.relative_to(PROJECT_ROOT)),
                "record_count": len(records),
                "dialogue_count": seal["dialogue_count"],
                "stratum_counts": seal["stratum_counts"],
                "manual_utterance_inspection_count": 0,
                "new_model_weight_download_count": 0,
                "model_load_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
