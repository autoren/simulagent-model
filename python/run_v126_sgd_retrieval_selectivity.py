#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import tarfile

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v106_open_world_benchmark import character_retrieval_observations, fit_character_retrieval
from v126_sgd_retrieval_selectivity import run_evaluation


def extract_selected(lock):
    config = lock["config_payload"]
    populations = json.loads((PROJECT_ROOT / lock["selected_populations"]).read_text())
    wanted = {row["candidate_id"]: ("train", row) for row in populations["training_population"]}
    wanted.update({row["candidate_id"]: ("evaluation", row) for row in populations["evaluation_population"]})
    found = {}
    root = f"dstc8-schema-guided-dialogue-{config['extraction']['archiveRevision']}"
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
                        actions = [action for action in frame.get("actions", []) if action.get("act") == "INFORM_INTENT" and action.get("slot") == "intent"]
                        values = [value for action in actions for value in action.get("values", [])]
                        if len(actions) == 1 and len(values) == 1 and active == values[0] and active != "NONE": matches.append((service, active))
                    if len(matches) != 1: continue
                    service, intent = matches[0]
                    identifier = f"sgd::{partition}::{dialogue_id}::{turn_index:03d}::{service}::{intent}"
                    if identifier in wanted:
                        role, structural = wanted[identifier]
                        found[identifier] = {**structural, "record_id": structural["population_id"], "utterance": turn["utterance"], "role": role}
    if set(found) != set(wanted): raise RuntimeError("V126 selected identifier extraction mismatch")
    training = [found[row["candidate_id"]] for row in populations["training_population"]]
    evaluation = [found[row["candidate_id"]] for row in populations["evaluation_population"]]
    return training, evaluation


def main():
    lock_path = PROJECT_ROOT / "configs/v126-sgd-retrieval-selectivity-lock.json"
    output_path = PROJECT_ROOT / "outputs/v126-sgd-retrieval-selectivity/evaluation/result.json"
    if output_path.exists(): raise RuntimeError("V126 may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]: raise RuntimeError("V126 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V126 dependency drifted: {key}")
    training_rows, evaluation = extract_selected(lock)
    training = [{"source_id": row["candidate_id"], "intent_id": f"{row['service']}::{row['intent']}", "scenario": row["domain"], "utterance": row["utterance"]} for row in training_rows]
    baseline = json.loads((PROJECT_ROOT / lock["baseline_config"]).read_text())
    vectorizer_spec = baseline["deterministicBaselines"]["character_ngram_retrieval"]["vectorizer"]
    fitted = fit_character_retrieval(training, vectorizer_spec)
    retrieval = character_retrieval_observations(fitted, evaluation)
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    v119 = json.loads((PROJECT_ROOT / lock["V119_config"]).read_text())
    summary = run_evaluation(evaluation, retrieval, catalog, baseline, v119, lock["config_payload"])
    if run_evaluation(evaluation, retrieval, catalog, baseline, v119, lock["config_payload"]) != summary: raise RuntimeError("V126 deterministic recomputation mismatch")
    access = {"source_archive_read_count": 1, "automatic_selected_language_parse_count": 1, "persisted_selected_language_record_count": 0, "manual_language_or_raw_response_inspection_count": 0, "protected_test_language_read_count": 0, "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0}
    result = {"schema_version": "126-sgd-retrieval-selectivity-result", "experiment": lock["config_payload"]["experiment"], "passed": summary["outcome_pass"], "decision": summary["decision"], "summary": summary, "extraction_summary": {"training_record_count": len(training), "evaluation_record_count": len(evaluation), "individual_language_record_emission_count": 0}, "retrieval_summary": {"training_record_count": len(training), "declared_intent_count": fitted["declared_intent_count"], "evaluation_record_count": len(retrieval), "individual_retrieval_record_emission_count": 0}, "deterministic_in_memory_recomputation_exact": True, "access": access, "claim_boundary": lock["config_payload"]["claimBoundary"]}
    write_json(output_path, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
