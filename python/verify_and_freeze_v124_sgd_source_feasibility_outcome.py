#!/usr/bin/env python3
import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v124-sgd-source-feasibility-lock.json"
    archive_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/source/sgd-pinned.tar.gz"
    inventory_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/source-inventory/sgd-open-set-inventory.json"
    access_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/source-inventory/access.json"
    doc_path = PROJECT_ROOT / "docs/v124-sgd-source-feasibility-results.md"
    audit_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v124-sgd-source-feasibility-outcome-lock.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v124_sgd_source_feasibility_outcome.py"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V124 outcome already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V124 result document first")
    lock = json.loads(lock_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    access = json.loads(access_path.read_text())
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    candidates = inventory["candidate_index"]
    identifiers = [row["candidate_id"] for row in candidates]
    forbidden = {"utterance", "text", "slot_values", "values", "tokens", "dialogue"}
    candidate_keys = set().union(*(row.keys() for row in candidates)) if candidates else set()
    config = lock["config_payload"]
    access_gates = config["accessGates"]
    checks = {
        "lock_and_dependencies_exact": bool(payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies)),
        "archive_hash_and_revision_root_exact": inventory["archive_sha256"] == file_sha256(archive_path) and inventory["archive_root"].endswith(config["revision"]),
        "source_gates_pass": inventory["source_pass"] and all(inventory["source_gates"].values()),
        "inventory_is_text_free_structural_and_unique": not (candidate_keys & forbidden) and not inventory["contains_language_or_slot_values"] and len(identifiers) == len(set(identifiers)) == inventory["candidate_count"],
        "access_gates_exact": bool(
            access["archive_payload_download_count"] <= access_gates["maximumArchivePayloadDownloadCount"]
            and access["source_archive_read_count"] <= access_gates["maximumSourceArchiveReadCount"]
            and access["automatic_language_parse_count"] <= access_gates["maximumAutomaticLanguageParseCount"]
            and access["manual_language_or_raw_response_inspection_count"] == 0
            and access["protected_test_language_read_count"] == 0
            and access["model_load_count"] == 0 and access["model_generation_count"] == 0
            and access["LLM_API_call_count"] == 0 and access["adapter_training_run_count"] == 0
            and access["real_service_call_count"] == 0 and access["external_side_effect_count"] == 0
            and access["actual_execution_count"] == 0
        ),
    }
    passed = all(checks.values())
    audit = {"schema_version": "124-sgd-source-feasibility-outcome-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "inventory_summary": {key: inventory[key] for key in ("dialogue_count", "domain_count", "service_count", "intent_count", "candidate_count", "candidate_counts", "test_open_set_class_counts", "test_open_set_domain_coverage", "candidate_index_sha256")}}
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit(1)
    paths = {"analysis_lock": lock_path, "source_archive": archive_path, "inventory": inventory_path, "access": access_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {
        "schema_version": "124-sgd-source-feasibility-outcome-lock",
        "experiment": "v124_sgd_source_feasibility_outcome_lock",
        "outcome": {"passed": True, "audit_pass": True, "source_pass": inventory["source_pass"], "decision": inventory["decision"], "inventory_summary": audit["inventory_summary"]},
        "authorization": {
            "modify_rerun_or_reclassify_V124": False,
            "preregister_text_free_SGD_catalog_and_population": bool(inventory["source_pass"]),
            "extract_selected_language_or_evaluate_signal_trigger_model": False,
            "open_protected_or_begin_induction_or_richer_planning": False,
            "run_API_training_action_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
