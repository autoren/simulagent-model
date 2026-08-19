#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_lock_path = PROJECT_ROOT / "configs/v102-presto-context-source-lock.json"
    runner_path = PROJECT_ROOT / "python/run_v102_presto_source_inventory.py"
    freezer_path = PROJECT_ROOT / "python/freeze_v102_presto_technical_failure.py"
    doc_path = PROJECT_ROOT / "docs/v102-presto-context-source-technical-failure.md"
    failure_path = PROJECT_ROOT / "outputs/v102-presto-context-source/technical-failure.json"
    outcome_path = PROJECT_ROOT / "configs/v102-presto-context-source-technical-outcome-lock.json"
    source_root = PROJECT_ROOT / "outputs/v102-presto-context-source/source"
    inventory_root = PROJECT_ROOT / "outputs/v102-presto-context-source/source-inventory"
    if failure_path.exists() or outcome_path.exists():
        raise RuntimeError("V102 technical failure is already frozen")
    if source_root.exists() or inventory_root.exists():
        raise RuntimeError("V102 unexpectedly persisted a source or inventory artifact")
    lock = json.loads(design_lock_path.read_text())
    if payload_hash(
        {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    ) != lock["lock_payload_sha256"]:
        raise RuntimeError("V102 design lock mismatch")
    dependency_keys = (
        "config", "parent_source_selection_lock", "massive_population_outcome", "plan",
        "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    if not all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
        for key in dependency_keys
    ):
        raise RuntimeError("V102 design dependency drifted")

    failure = {
        "schema_version": "102-presto-context-source-technical-failure",
        "experiment": "v102_presto_context_source_inventory",
        "passed": False,
        "scientific_source_feasibility_evaluated": False,
        "failure_class": "technical_parser_schema_validation_failure",
        "exception_type": "ValueError",
        "exception_message": "seeded note is invalid",
        "failed_operation": "automatic_text_free_context_surface_normalization",
        "archive_identity_checks_completed_before_failure": True,
        "locked_dev_test_JSONL_parse_completed_before_failure": True,
        "source_or_inventory_artifact_persisted": False,
        "access": {
            "pinned_HTTP_download_count": 1,
            "downloaded_byte_count": 415990813,
            "archive_payload_parse_count": 1,
            "emitted_language_record_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0
        },
        "decision": "freeze_technical_V102_and_preregister_parser_repair_without_gate_changes",
        "claim_boundary": "technical parser outcome only; no PRESTO feasibility or model conclusion",
    }
    failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
    dependencies = {
        "design_lock": design_lock_path,
        "runner": runner_path,
        "freezer": freezer_path,
        "failure": failure_path,
        "failure_document": doc_path,
    }
    outcome: dict[str, Any] = {
        "schema_version": "102-presto-context-source-technical-outcome-lock",
        "experiment": "v102_presto_context_source_technical_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_source_feasibility_evaluated": False,
            "decision": failure["decision"],
            "failure_class": failure["failure_class"],
        },
        "authorization": {
            "modify_or_rerun_V102": False,
            "preregister_V102r1_parser_repair": True,
            "retain_exact_source_dependency_rule_and_scientific_gates": True,
            "authorize_one_fresh_archive_retrieval_because_no_artifact_persisted": True,
            "emit_or_manually_inspect_language": False,
            "select_population_or_extract_selected_language": False,
            "load_local_or_API_model": False,
            "train_adapter_or_learn_likelihood": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(failure, indent=2, sort_keys=True))
    print(json.dumps({
        "lock": str(outcome_path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(outcome_path),
    }, indent=2))


if __name__ == "__main__":
    main()
