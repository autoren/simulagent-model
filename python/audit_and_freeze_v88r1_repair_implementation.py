#!/usr/bin/env python3
"""Audit and freeze the V88r1 name-preserving runner before its terminal retry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from run_v88r1_external_candidate_mlx import score_named_record
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v88r1-name-preservation-repair-design-lock.json"
    runner_path = PROJECT_ROOT / "python/run_v88r1_external_candidate_mlx.py"
    tests_path = PROJECT_ROOT / "python/test_v88r1_name_preservation.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v88r1_repair_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v88r1-name-preservation-repair-implementation-lock.json"
    outcome_dir = PROJECT_ROOT / "outputs/v88r1-external-intent-candidate/evaluation"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V88r1 repair implementation is already frozen")
    if outcome_dir.exists():
        raise RuntimeError("V88r1 outcome exists before implementation lock")

    design = json.loads(design_path.read_text())
    design_payload = {key: value for key, value in design.items() if key != "lock_payload_sha256"}
    original_path = PROJECT_ROOT / design["original_implementation_lock"]
    original = json.loads(original_path.read_text())
    original_payload = {key: value for key, value in original.items() if key != "lock_payload_sha256"}
    execution_path = PROJECT_ROOT / design["parent_execution_lock"]
    execution = json.loads(execution_path.read_text())
    runner_source = runner_path.read_text()

    fixture = {
        "name": "registered-name", "id": "fixture-id", "source_record_id": "source-id",
        "service": "Service_1", "allowed_intent_ids": ["Book", "NONE"],
        "allowed_slot_ids": ["city"],
        "gold": {"active_intent": "Book", "intent_candidates": ["Book", "NONE"], "state_slot_key_candidates": ["city"]},
    }
    response = json.dumps({"intent_candidates": ["Book", "NONE"], "state_slot_key_candidates": ["city"]})
    named = score_named_record(fixture, response)
    config = original["config_payload"]
    retry = design["config_payload"]["retry"]
    checks = {
        "repair_design_original_implementation_and_failure_locks_exact": bool(
            payload_hash(design_payload) == design["lock_payload_sha256"]
            and design["authorization"]["implement_and_audit_name_preserving_runner"]
            and not design["authorization"]["run_local_model"]
            and payload_hash(original_payload) == original["lock_payload_sha256"]
            and file_sha256(original_path) == design["original_implementation_lock_sha256"]
            and file_sha256(execution_path) == design["parent_execution_lock_sha256"]
        ),
        "all_scientific_dependencies_are_inherited_byte_exact": bool(
            all(file_sha256(PROJECT_ROOT / original[key]) == original[f"{key}_sha256"] for key in (
                "design_lock", "corpus_seal", "corpus", "protocol", "census_harness", "tests", "builder"
            ))
            and config["decoding"]["retryOnMalformedOutput"] is False
        ),
        "only_registered_fixture_name_is_preserved_before_harness": bool(
            named["name"] == fixture["name"]
            and named["id"] == fixture["id"]
            and named["intent_candidate_exact"]
            and named["state_slot_key_exact"]
            and 'row["name"] = record["name"]' in runner_source
        ),
        "runner_reuses_frozen_prompt_model_decoder_protocol_and_harness": bool(
            "format_user_prompt(record, config)" in runner_source
            and "score_response(record, response)" in runner_source
            and "load(str(snapshot))" in runner_source
            and "enable_thinking=False" in runner_source
            and "make_sampler(temp=config[\"decoding\"][\"temperature\"])" in runner_source
            and "run_locked_census_once" in runner_source
        ),
        "retry_and_cumulative_budgets_match_failure_lock": bool(
            execution["access"]["model_load_count"] == retry["priorFailedModelLoadCount"] == 1
            and execution["access"]["model_generation_count"] == retry["priorFailedModelGenerationCount"] == 1
            and retry["maximumRetryModelLoadCount"] == 1
            and retry["maximumRetryModelGenerationCount"] == 48
            and retry["maximumCumulativeModelLoadCount"] == 2
            and retry["maximumCumulativeModelGenerationCount"] == 49
            and "cumulative_model_load_budget" in runner_source
            and "cumulative_model_generation_budget" in runner_source
        ),
        "runner_forbids_API_training_manual_inspection_execution_and_further_retry": bool(
            "LLM_API_call_count" in runner_source
            and "manual_utterance_inspection_count" in runner_source
            and "real_service_call_count" in runner_source
            and not retry["anyFurtherRetryAuthorized"]
        ),
        "zero_repair_preinference_access": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "88r1-name-preservation-repair-implementation-audit",
        "experiment": "v88r1_name_preservation_repair_implementation_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_single_terminal_48_record_local_retry" if passed else "reject_V88r1_runner",
        "checks": checks,
        "access": {"model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "manual_utterance_inspection_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    snapshot = Path(original["local_snapshot_path"])
    lock = {
        "schema_version": "88r1-name-preservation-repair-implementation-lock",
        "experiment": "v88r1_name_preservation_repair_implementation_lock",
        "repair_design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "repair_design_lock_sha256": file_sha256(design_path),
        "original_implementation_lock": str(original_path.relative_to(PROJECT_ROOT)),
        "original_implementation_lock_sha256": file_sha256(original_path),
        "parent_execution_lock": str(execution_path.relative_to(PROJECT_ROOT)),
        "parent_execution_lock_sha256": file_sha256(execution_path),
        "config_payload": config,
        "corpus_seal": original["corpus_seal"], "corpus_seal_sha256": original["corpus_seal_sha256"],
        "corpus": original["corpus"], "corpus_sha256": original["corpus_sha256"],
        "protocol": original["protocol"], "protocol_sha256": original["protocol_sha256"],
        "census_harness": original["census_harness"], "census_harness_sha256": original["census_harness_sha256"],
        "runner": str(runner_path.relative_to(PROJECT_ROOT)), "runner_sha256": file_sha256(runner_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)), "tests_sha256": file_sha256(tests_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)), "implementation_auditor_sha256": file_sha256(auditor_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)), "implementation_audit_sha256": file_sha256(audit_path),
        "local_snapshot_path": str(snapshot),
        "local_config_sha256": original["local_config_sha256"],
        "local_model_index_sha256": original["local_model_index_sha256"],
        "local_tokenizer_config_sha256": original["local_tokenizer_config_sha256"],
        "prior_failed_attempt_access": execution["access"],
        "cumulative_resource_budget": {
            "maximum_model_load_count": retry["maximumCumulativeModelLoadCount"],
            "maximum_model_generation_count": retry["maximumCumulativeModelGenerationCount"],
        },
        "authorization": {
            "modify_any_scientific_dependency_or_name_repair": False,
            "run_single_local_repair_once": True,
            "deploy_or_execute_any_model_output": False,
            "run_API_model_or_train_adapter": False,
            "manually_inspect_source_language_or_prompts": False,
            "perform_real_service_call_or_external_side_effect": False,
            "rerun_after_any_outcome_or_execution_failure": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
