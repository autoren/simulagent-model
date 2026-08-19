#!/usr/bin/env python3
"""Freeze a valid-only exploratory clarification batch after the Phase 2 failure."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from prospective_language_phase3 import (
    eligible_clarification_records,
    load_controller_outputs,
    validate_phase3_config,
)
from prospective_language_pilot import sha256_json
from v10_protocol import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "prospective-language-pilot-v1-phase3.json"
PARTICIPANT_DIR = ROOT / "data" / "prospective-language-pilot" / "prospective-language-pilot-v1" / "P001"
LOCK_PATH = PARTICIPANT_DIR / "audit" / "phase3_clarification_lock.json"


def main() -> None:
    if LOCK_PATH.exists():
        raise RuntimeError("Phase 3 clarification lock already exists.")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_phase3_config(config)
    result_path = ROOT / config["phase2"]["result"]
    outputs_path = ROOT / config["phase2"]["controller_outputs"]
    if file_sha256(result_path) != config["phase2"]["result_sha256"]:
        raise RuntimeError("Frozen Phase 2 result hash mismatch.")
    if file_sha256(outputs_path) != config["phase2"]["controller_outputs_sha256"]:
        raise RuntimeError("Frozen Phase 2 controller-output hash mismatch.")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("completed") is not True or result.get("qualified_for_clarification_batch") is not False:
        raise RuntimeError("Phase 2 result is not the required completed negative result.")
    required_safe_gates = (
        "required_controller_coverage_rate",
        "required_record_count",
        "maximum_questions_per_clarification",
        "maximum_retry_count",
        "maximum_api_call_count",
        "maximum_real_service_call_count",
        "maximum_external_side_effect_count",
        "maximum_actual_execution_count",
    )
    if any(result["gates"].get(gate) is not True for gate in required_safe_gates):
        raise RuntimeError("A required controller or access-safety gate failed.")
    if not (
        result["access"]["retry_count"] == 0
        and result["access"]["api_call_count"] == 0
        and result["access"]["real_service_call_count"] == 0
        and result["access"]["external_side_effect_count"] == 0
        and result["access"]["actual_execution_count"] == 0
    ):
        raise RuntimeError("Phase 2 access counters are not safe for exploratory salvage.")

    rows = load_controller_outputs(outputs_path)
    eligible = eligible_clarification_records(config, rows)
    questions = [question for row in eligible for question in row["clarification_questions"]]
    if any(
        not question.endswith("?")
        or question.count("?") != 1
        or re.search(r"https?://|www\.", question, re.IGNORECASE)
        or re.search(r"password|credential|social security|credit card", question, re.IGNORECASE)
        for question in questions
    ):
        raise RuntimeError("A frozen clarification question failed the surface-safety audit.")

    dependency_paths = [
        "configs/prospective-language-pilot-v1-phase3.json",
        "python/prospective_language_phase3.py",
        "python/prospective_language_pilot_app.py",
        "python/test_prospective_language_phase3.py",
        config["phase2"]["result"],
        config["phase2"]["controller_outputs"],
    ]
    payload = {
        "schema_version": "prospective-language-pilot-v1-phase3-clarification-lock",
        "locked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": config["experiment"],
        "role": config["role"],
        "claim_boundary": config["claim_boundary"],
        "phase2_result_remains_qualified": False,
        "interaction_authorized": True,
        "interaction_scope": "exact_valid_nonfallback_clarification_questions_only",
        "eligible_record_ids": [row["record_id"] for row in eligible],
        "eligible_record_count": len(eligible),
        "question_count": len(questions),
        "question_bundle_sha256": sha256_json(
            [
                {
                    "record_id": row["record_id"],
                    "controller_payload_sha256": row["controller_payload_sha256"],
                    "clarification_questions": row["clarification_questions"],
                }
                for row in eligible
            ]
        ),
        "surface_safety_audit": {
            "all_questions_single_and_question_mark_terminated": True,
            "contains_url_count": 0,
            "credential_request_count": 0,
            "manual_question_rewrite_count": 0,
        },
        "dependencies": [
            {"path": path, "sha256": file_sha256(ROOT / path)} for path in dependency_paths
        ],
        "phase3_collection": {
            "model_generation_count": 0,
            "retry_count": 0,
            "terminal_generation_authorized": False,
            "real_world_execution_enabled": False,
        },
        "prohibitions": {
            "reclassify_phase2_as_pass": True,
            "repair_or_regenerate_invalid_records": True,
            "show_candidate_goals_or_reasoning": True,
            "show_plan_defer_or_shadow_outputs_during_clarification": True,
            "generate_terminal_outputs_before_all_answers_lock": True,
        },
    }
    lock = {**payload, "lock_payload_sha256": sha256_json(payload)}
    LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "interaction_authorized": True,
        "phase2_result_remains_qualified": False,
        "eligible_record_count": len(eligible),
        "question_count": len(questions),
        "lock_payload_sha256": lock["lock_payload_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
