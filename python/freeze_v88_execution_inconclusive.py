#!/usr/bin/env python3
"""Freeze the V88 harness-identity execution failure without a scientific outcome."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    impl_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-implementation-lock.json"
    failure_path = PROJECT_ROOT / "outputs/v88-external-intent-candidate/evaluation/failure.json"
    progress_path = PROJECT_ROOT / "outputs/v88-external-intent-candidate/evaluation/access-progress.json"
    note_path = PROJECT_ROOT / "docs/v88-external-intent-candidate-execution-note.md"
    freezer_path = PROJECT_ROOT / "python/freeze_v88_execution_inconclusive.py"
    lock_path = PROJECT_ROOT / "configs/v88-external-intent-candidate-execution-lock.json"
    if lock_path.exists():
        raise RuntimeError("V88 execution failure is already frozen")
    result_path = PROJECT_ROOT / "outputs/v88-external-intent-candidate/evaluation/result.json"
    if result_path.exists():
        raise RuntimeError("V88 result exists; execution may not be frozen as inconclusive")
    impl = json.loads(impl_path.read_text())
    impl_payload = {key: value for key, value in impl.items() if key != "lock_payload_sha256"}
    failure = json.loads(failure_path.read_text())
    progress = json.loads(progress_path.read_text())
    checks = {
        "implementation_lock_exact": payload_hash(impl_payload) == impl["lock_payload_sha256"],
        "failure_is_exact_registered_name_contract_error": bool(
            failure["status"] == "execution_failure"
            and failure["stage"] == "fixture_evaluation"
            and failure["exception_type"] == "ValueError"
            and failure["exception_message"] == "fixture evaluator changed or omitted the registered name"
        ),
        "no_fixture_or_result_was_preserved": bool(
            failure["completed_fixture_count"] == 0
            and failure["raw_fixture_artifacts_preserved"] == 0
            and not failure["result_artifact_written"]
            and not result_path.exists()
        ),
        "exact_partial_access_is_one_local_generation_only": bool(
            progress["model_load_count"] == failure["attempt"]["model_load_count"] == 1
            and progress["model_generation_count"] == failure["attempt"]["model_generation_count"] == 1
            and progress["LLM_API_call_count"] == 0
            and progress["adapter_training_run_count"] == 0
            and progress["manual_utterance_inspection_count"] == 0
            and progress["real_service_call_count"] == 0
            and progress["external_side_effect_count"] == 0
        ),
        "failed_response_is_unobserved_and_unscored": failure["raw_fixture_artifacts_preserved"] == 0,
    }
    if not all(checks.values()):
        print(json.dumps(checks, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "88-external-intent-candidate-execution-lock",
        "experiment": "v88_external_intent_candidate_execution_inconclusive",
        "status": "execution_inconclusive",
        "scientific_outcome": None,
        "implementation_lock": str(impl_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(impl_path),
        "failure": str(failure_path.relative_to(PROJECT_ROOT)),
        "failure_sha256": file_sha256(failure_path),
        "access_progress": str(progress_path.relative_to(PROJECT_ROOT)),
        "access_progress_sha256": file_sha256(progress_path),
        "note": str(note_path.relative_to(PROJECT_ROOT)),
        "note_sha256": file_sha256(note_path),
        "freezer": str(freezer_path.relative_to(PROJECT_ROOT)),
        "freezer_sha256": file_sha256(freezer_path),
        "checks": checks,
        "access": progress,
        "authorization": {
            "modify_or_resume_V88_attempt": False,
            "interpret_as_positive_or_negative_model_evidence": False,
            "preregister_one_mechanical_name_preservation_retry": True,
            "retry_may_change_corpus_prompt_model_decoding_parser_scoring_controls_or_quality_gates": False,
            "retry_may_only_preserve_registered_fixture_name": True,
            "maximum_cumulative_model_load_count_after_retry": 2,
            "maximum_cumulative_model_generation_count_after_retry": 49,
            "run_API_model_or_train_adapter": False,
            "perform_real_service_call_or_external_side_effect": False,
            "authorize_any_further_retry": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks, "lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
