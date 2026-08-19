#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import tarfile

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v128_sgd_typed_relation_feasibility import build_support_signatures, relation_tokens, run_evaluation


def extract_relations(lock):
    config = lock["config_payload"]
    training_population = json.loads((PROJECT_ROOT / lock["training_population"]).read_text())["training_population"]
    evaluation = json.loads((PROJECT_ROOT / lock["fresh_population"]).read_text())["records"]
    train_wanted = {row["candidate_id"]: f"{row['service']}::{row['intent']}" for row in training_population}
    eval_wanted = {row["candidate_id"]: row["record_id"] for row in evaluation}
    training_relations = []
    evidence = {}
    root = f"dstc8-schema-guided-dialogue-{config['sourceArchiveRevision']}"
    with tarfile.open(PROJECT_ROOT / lock["source_archive"], mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and re.fullmatch(rf"{re.escape(root)}/(train|test)/dialogues_\d+\.json", member.name)]
        for member in sorted(members, key=lambda item: item.name):
            partition = member.name.split("/")[-2]
            handle = archive.extractfile(member)
            if handle is None: raise ValueError("unreadable SGD dialogue member")
            for dialogue in json.loads(handle.read()):
                dialogue_id = str(dialogue["dialogue_id"])
                for turn_index, turn in enumerate(dialogue["turns"]):
                    if turn.get("speaker") != "USER": continue
                    matches = []
                    for frame in turn.get("frames", []):
                        service = frame.get("service"); active = frame.get("state", {}).get("active_intent")
                        intent_actions = [action for action in frame.get("actions", []) if action.get("act") == "INFORM_INTENT" and action.get("slot") == "intent"]
                        if len(intent_actions) == 1 and active != "NONE": matches.append((service, active, frame))
                    if len(matches) != 1: continue
                    service, active, frame = matches[0]
                    candidate_id = f"sgd::{partition}::{dialogue_id}::{turn_index:03d}::{service}::{active}"
                    if candidate_id in train_wanted:
                        training_relations.append((train_wanted[candidate_id], relation_tokens(frame)))
                    if candidate_id in eval_wanted:
                        evidence[eval_wanted[candidate_id]] = relation_tokens(frame)
    if len(training_relations) != len(train_wanted): raise RuntimeError("V128 training annotation extraction mismatch")
    if set(evidence) != {row["record_id"] for row in evaluation}: raise RuntimeError("V128 evaluation annotation extraction mismatch")
    signatures = build_support_signatures(training_relations)
    if len(signatures) != 6: raise RuntimeError("V128 support signature count mismatch")
    return evaluation, evidence, signatures, len(training_relations)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v128-sgd-typed-relation-feasibility-lock.json"
    output_path = PROJECT_ROOT / "outputs/v128-sgd-typed-relation-feasibility/evaluation/result.json"
    if output_path.exists(): raise RuntimeError("V128 may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]: raise RuntimeError("V128 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V128 dependency drifted: {key}")
    records, evidence, signatures, training_count = extract_relations(lock)
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    baseline = json.loads((PROJECT_ROOT / lock["baseline_config"]).read_text())
    v119 = json.loads((PROJECT_ROOT / lock["V119_config"]).read_text())
    summary = run_evaluation(records, evidence, signatures, catalog, baseline, v119, lock["config_payload"])
    if run_evaluation(records, evidence, signatures, catalog, baseline, v119, lock["config_payload"]) != summary: raise RuntimeError("V128 deterministic recomputation mismatch")
    access = {
        "source_archive_read_count": 1, "automatic_annotation_parse_pass_count": 1,
        "utterance_field_access_count": 0, "slot_value_access_count": 0,
        "persisted_individual_evidence_count": 0, "manual_language_or_raw_response_inspection_count": 0,
        "protected_test_language_read_count": 0, "model_load_count": 0, "model_generation_count": 0,
        "LLM_API_call_count": 0, "adapter_training_run_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    first = next(iter(summary["conditions"].values()))
    support_sizes = [len(row["allowed"]) for row in signatures.values()]
    result = {
        "schema_version": "128-sgd-typed-relation-feasibility-result", "experiment": lock["config_payload"]["experiment"],
        "passed": summary["outcome_pass"], "decision": summary["decision"], "summary": summary,
        "extraction_summary": {
            "training_annotation_record_count": training_count, "evaluation_annotation_record_count": len(records),
            "signature_count": len(signatures), "minimum_signature_token_count": min(support_sizes),
            "maximum_signature_token_count": max(support_sizes),
            "typed_evidence_present_count": round(first["typed_evidence_presence_fraction"] * len(records)),
            "utterance_field_access_count": 0, "slot_value_access_count": 0, "individual_evidence_emission_count": 0,
        },
        "deterministic_in_memory_recomputation_exact": True, "access": access,
        "claim_boundary": lock["config_payload"]["claimBoundary"],
    }
    write_json(output_path, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
