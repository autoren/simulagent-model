#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import tarfile

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v127_sgd_typed_constraint_feasibility import run_evaluation


def _slot_set(value) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value}
    if isinstance(value, list):
        return {str(item) for item in value}
    raise ValueError("unexpected SGD schema slot collection")


def extract_annotations_and_signatures(lock):
    config = lock["config_payload"]
    population = json.loads((PROJECT_ROOT / lock["fresh_population"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    wanted = {row["candidate_id"]: row for row in population["records"]}
    root = f"dstc8-schema-guided-dialogue-{config['sourceArchiveRevision']}"
    signatures = {}
    evidence = {}
    with tarfile.open(PROJECT_ROOT / lock["source_archive"], mode="r:gz") as archive:
        schema_member = archive.getmember(f"{root}/train/schema.json")
        handle = archive.extractfile(schema_member)
        if handle is None: raise ValueError("unreadable SGD train schema")
        services = json.loads(handle.read())
        schemas = {service["service_name"]: service for service in services}
        for choice in catalog["choices"]:
            if choice["kind"] != "KNOWN": continue
            service = schemas[choice["service"]]
            intents = [item for item in service["intents"] if item["name"] == choice["intent"]]
            if len(intents) != 1: raise ValueError("known intent missing from pinned train schema")
            intent = intents[0]
            required = _slot_set(intent.get("required_slots", []))
            optional = _slot_set(intent.get("optional_slots", {}))
            signatures[choice["intent_id"]] = {"required": required, "allowed": required | optional}
        members = [
            member for member in archive.getmembers()
            if member.isfile() and re.fullmatch(rf"{re.escape(root)}/test/dialogues_\d+\.json", member.name)
        ]
        for member in sorted(members, key=lambda item: item.name):
            handle = archive.extractfile(member)
            if handle is None: raise ValueError("unreadable SGD test dialogue member")
            for dialogue in json.loads(handle.read()):
                dialogue_id = str(dialogue["dialogue_id"])
                for turn_index, turn in enumerate(dialogue["turns"]):
                    if turn.get("speaker") != "USER": continue
                    matches = []
                    for frame in turn.get("frames", []):
                        service = frame.get("service")
                        active = frame.get("state", {}).get("active_intent")
                        actions = [action for action in frame.get("actions", []) if action.get("act") == "INFORM_INTENT" and action.get("slot") == "intent"]
                        if len(actions) == 1 and active != "NONE":
                            matches.append((service, active, frame))
                    if len(matches) != 1: continue
                    service, active, frame = matches[0]
                    candidate_id = f"sgd::test::{dialogue_id}::{turn_index:03d}::{service}::{active}"
                    if candidate_id not in wanted: continue
                    slots = set(frame.get("state", {}).get("slot_values", {}).keys())
                    slots.update(
                        str(action["slot"]) for action in frame.get("actions", [])
                        if action.get("slot") not in (None, "", "intent")
                    )
                    evidence[wanted[candidate_id]["record_id"]] = slots
    if set(evidence) != {row["record_id"] for row in population["records"]}:
        raise RuntimeError("V127 selected annotation extraction mismatch")
    if len(signatures) != 6:
        raise RuntimeError("V127 known schema signature mismatch")
    return population["records"], evidence, signatures


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v127-sgd-typed-constraint-feasibility-lock.json"
    output_path = PROJECT_ROOT / "outputs/v127-sgd-typed-constraint-feasibility/evaluation/result.json"
    if output_path.exists(): raise RuntimeError("V127 may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V127 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V127 dependency drifted: {key}")
    records, evidence, signatures = extract_annotations_and_signatures(lock)
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    baseline = json.loads((PROJECT_ROOT / lock["baseline_config"]).read_text())
    v119 = json.loads((PROJECT_ROOT / lock["V119_config"]).read_text())
    summary = run_evaluation(records, evidence, signatures, catalog, baseline, v119, lock["config_payload"])
    if run_evaluation(records, evidence, signatures, catalog, baseline, v119, lock["config_payload"]) != summary:
        raise RuntimeError("V127 deterministic recomputation mismatch")
    access = {
        "source_archive_read_count": 1, "automatic_selected_annotation_parse_count": 1,
        "utterance_field_access_count": 0, "slot_value_access_count": 0,
        "persisted_individual_evidence_count": 0, "manual_language_or_raw_response_inspection_count": 0,
        "protected_test_language_read_count": 0, "model_load_count": 0, "model_generation_count": 0,
        "LLM_API_call_count": 0, "adapter_training_run_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    first = next(iter(summary["conditions"].values()))
    result = {
        "schema_version": "127-sgd-typed-constraint-feasibility-result",
        "experiment": lock["config_payload"]["experiment"], "passed": summary["outcome_pass"],
        "decision": summary["decision"], "summary": summary,
        "extraction_summary": {
            "selected_annotation_record_count": len(records), "schema_signature_count": len(signatures),
            "typed_evidence_present_count": round(first["typed_evidence_presence_fraction"] * len(records)),
            "utterance_field_access_count": 0, "slot_value_access_count": 0,
            "individual_evidence_emission_count": 0,
        },
        "deterministic_in_memory_recomputation_exact": True, "access": access,
        "claim_boundary": lock["config_payload"]["claimBoundary"],
    }
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
