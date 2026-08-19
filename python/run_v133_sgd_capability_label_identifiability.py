#!/usr/bin/env python3
import json

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v133_sgd_capability_label_identifiability import run_audit


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v133-sgd-capability-label-identifiability-lock.json"
    result_path = PROJECT_ROOT / "outputs/v133-sgd-capability-label-identifiability/evaluation/result.json"
    if result_path.exists(): raise RuntimeError("V133 may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]: raise RuntimeError("V133 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V133 dependency drifted: {key}")
    archive_bytes = (PROJECT_ROOT / lock["source_archive"]).read_bytes()
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    population = json.loads((PROJECT_ROOT / lock["fixture_population"]).read_text())
    audit = run_audit(archive_bytes, catalog, population, lock["config_payload"])
    access = {"source_archive_read_count": 1, "schema_file_read_count": 2, "dialogue_file_read_count": 0, "language_record_read_count": 0, "manual_language_or_raw_response_inspection_count": 0, "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0}
    ag = lock["config_payload"]["accessGates"]
    access_checks = {"source_archive_read_budget": access["source_archive_read_count"] <= ag["maximumSourceArchiveReadCount"], "schema_file_read_budget": access["schema_file_read_count"] <= ag["maximumSchemaFileReadCount"], "zero_dialogue_files": access["dialogue_file_read_count"] <= ag["maximumDialogueFileReadCount"], "zero_language_records": access["language_record_read_count"] <= ag["maximumLanguageRecordReadCount"], "zero_manual_inspection": access["manual_language_or_raw_response_inspection_count"] <= ag["maximumManualLanguageOrRawResponseInspectionCount"], "zero_model": access["model_load_count"] <= ag["maximumModelLoadCount"] and access["model_generation_count"] <= ag["maximumModelGenerationCount"], "zero_API_training_service_side_effect": access["LLM_API_call_count"] <= ag["maximumLLMAPICallCount"] and access["adapter_training_run_count"] <= ag["maximumAdapterTrainingRunCount"] and access["real_service_call_count"] <= ag["maximumRealServiceCallCount"] and access["external_side_effect_count"] <= ag["maximumExternalSideEffectCount"]}
    audit.update({"access": access, "access_gates": access_checks, "access_pass": all(access_checks.values())})
    write_json(result_path, audit); print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__": main()
