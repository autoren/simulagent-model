"""Immutable clarification storage for the prospective language pilot Phase 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from prospective_language_pilot import (
    PilotProtocolError,
    _atomic_write_json,
    _atomic_write_text,
    canonical_json,
    scenario_index,
    sha256_bytes,
    sha256_json,
    utc_now,
)


PHASE3_SCHEMA_VERSION = "prospective-language-pilot-v1-phase3-design-1.0.0"
ALLOWED_UNABLE_REASONS = {"do_not_know", "question_unclear", "prefer_not_to_answer", "other"}


def validate_phase3_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != PHASE3_SCHEMA_VERSION:
        raise PilotProtocolError("Unsupported Phase 3 config schema.")
    phase2 = config.get("phase2", {})
    eligibility = config.get("eligibility", {})
    collection = config.get("collection", {})
    if phase2.get("required_qualified_for_clarification_batch") is not False:
        raise PilotProtocolError("Phase 3 must preserve the failed Phase 2 qualification status.")
    if phase2.get("required_valid_nonfallback_clarification_records") != 11:
        raise PilotProtocolError("Phase 3 requires the frozen 11-record clarification subset.")
    if not (
        eligibility.get("include_only_route") == "CLARIFY"
        and eligibility.get("exclude_safe_fallback") is True
        and eligibility.get("allow_question_rewrite") is False
        and eligibility.get("allow_model_rerun_or_reprompt") is False
    ):
        raise PilotProtocolError("Phase 3 valid-only eligibility rules changed.")
    if not (
        collection.get("all_answers_before_terminal_generation") is True
        and collection.get("responses_immutable_after_lock") is True
        and collection.get("assistant_generation_enabled") is False
        and collection.get("terminal_run_authorized") is False
    ):
        raise PilotProtocolError("Phase 3 collection boundary changed.")


def load_controller_outputs(path: Path | str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def eligible_clarification_records(
    phase3_config: Mapping[str, Any], controller_rows: list[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    validate_phase3_config(phase3_config)
    rows = [
        row
        for row in controller_rows
        if row.get("route") == "CLARIFY" and row.get("used_safe_fallback") is False
    ]
    expected = phase3_config["phase2"]["required_valid_nonfallback_clarification_records"]
    if len(rows) != expected:
        raise PilotProtocolError(f"Expected {expected} eligible clarifications but found {len(rows)}.")
    if sum(len(row["clarification_questions"]) for row in rows) != phase3_config["phase2"]["required_question_count"]:
        raise PilotProtocolError("Frozen clarification question count mismatch.")
    return sorted(rows, key=lambda row: row["display_position"])


def phase3_state_path(participant_dir: Path | str) -> Path:
    return Path(participant_dir) / "private" / "phase3_clarification_state.json"


def initialize_or_load_phase3(
    phase3_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
    participant_dir: Path | str,
    controller_rows: list[Mapping[str, Any]],
    phase3_lock: Mapping[str, Any],
) -> dict[str, Any]:
    validate_phase3_config(phase3_config)
    participant_path = Path(participant_dir)
    eligible = eligible_clarification_records(phase3_config, controller_rows)
    state_path = phase3_state_path(participant_path)
    config_hash = sha256_json(phase3_config)
    eligible_hash = sha256_json(eligible)
    lock_hash = phase3_lock.get("lock_payload_sha256")
    if not isinstance(lock_hash, str):
        raise PilotProtocolError("Phase 3 clarification lock is missing its payload hash.")

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("phase3_config_sha256") != config_hash:
            raise PilotProtocolError("Frozen Phase 3 config changed after collection began.")
        if state.get("eligible_records_sha256") != eligible_hash:
            raise PilotProtocolError("Eligible clarification records changed after collection began.")
        if state.get("phase3_lock_payload_sha256") != lock_hash:
            raise PilotProtocolError("Phase 3 clarification lock changed after collection began.")
        export_phase3_bundles(phase3_config, study_config, participant_path, controller_rows, state)
        return state

    now = utc_now()
    state = {
        "schema_version": "prospective-language-pilot-v1-phase3-state-1.0.0",
        "study_id": study_config["study_id"],
        "participant_code": phase3_config["participant_code"],
        "phase3_config_sha256": config_hash,
        "phase3_lock_payload_sha256": lock_hash,
        "eligible_records_sha256": eligible_hash,
        "clarification_order": [row["record_id"] for row in eligible],
        "phase": "phase_3_clarification_collection",
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
        "responses": {},
    }
    _atomic_write_json(state_path, state)
    export_phase3_bundles(phase3_config, study_config, participant_path, controller_rows, state)
    return state


def next_clarification_record_id(state: Mapping[str, Any]) -> str | None:
    responses = state.get("responses", {})
    return next((record_id for record_id in state["clarification_order"] if record_id not in responses), None)


def lock_clarification_response(
    phase3_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
    participant_dir: Path | str,
    controller_rows: list[Mapping[str, Any]],
    state: dict[str, Any],
    *,
    record_id: str,
    answer: str,
    unable_reason: str | None,
    unable_note: str,
    participant_attestation: bool,
) -> dict[str, Any]:
    if state.get("phase") != "phase_3_clarification_collection":
        raise PilotProtocolError("The clarification batch is already complete.")
    expected = next_clarification_record_id(state)
    if record_id != expected:
        raise PilotProtocolError(f"Expected clarification {expected}, received {record_id}.")
    if not participant_attestation:
        raise PilotProtocolError("Confirm that this is your own response before locking it.")

    response = answer.strip()
    note = unable_note.strip()
    if unable_reason is None:
        if len(response) < phase3_config["collection"]["minimum_answer_characters"]:
            raise PilotProtocolError("Write an answer or use the unable-to-answer option.")
        status = "answered"
        unable = None
    else:
        if unable_reason not in ALLOWED_UNABLE_REASONS:
            raise PilotProtocolError("Unsupported clarification unable reason.")
        if response:
            raise PilotProtocolError("Leave the answer blank when using unable-to-answer.")
        status = "unable_to_answer"
        unable = {"reason": unable_reason, "note": note or None}

    row = next(row for row in controller_rows if row["record_id"] == record_id)
    locked_at = utc_now()
    public_payload = {
        "schema_version": "1.0.0",
        "study_id": study_config["study_id"],
        "participant_code": state["participant_code"],
        "record_id": record_id,
        "display_position": row["display_position"],
        "clarification_questions": row["clarification_questions"],
        "response_status": status,
        "clarification_response": response or None,
        "locked_at": locked_at,
    }
    record = {
        **public_payload,
        "unable": unable,
        "participant_attestation": True,
        "public_payload_sha256": sha256_json(public_payload),
    }
    state["responses"][record_id] = record
    state["updated_at"] = locked_at
    if len(state["responses"]) == len(state["clarification_order"]):
        state["phase"] = "phase_3_complete_waiting_for_terminal_run"
        state["completed_at"] = locked_at
    participant_path = Path(participant_dir)
    _atomic_write_json(phase3_state_path(participant_path), state)
    export_phase3_bundles(phase3_config, study_config, participant_path, controller_rows, state)
    return record


def export_phase3_bundles(
    phase3_config: Mapping[str, Any],
    study_config: Mapping[str, Any],
    participant_dir: Path,
    controller_rows: list[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> dict[str, Path]:
    controller = {row["record_id"]: row for row in controller_rows}
    scenarios = scenario_index(study_config)
    public_records: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []
    for record_id in state["clarification_order"]:
        if record_id not in state["responses"]:
            continue
        response = state["responses"][record_id]
        public_payload = {
            key: response[key]
            for key in (
                "schema_version",
                "study_id",
                "participant_code",
                "record_id",
                "display_position",
                "clarification_questions",
                "response_status",
                "clarification_response",
                "locked_at",
            )
        }
        if sha256_json(public_payload) != response["public_payload_sha256"]:
            raise PilotProtocolError(f"Clarification payload hash mismatch for {record_id}.")
        public_records.append({**public_payload, "public_payload_sha256": response["public_payload_sha256"]})
        private_records.append(
            {
                **public_payload,
                "initial_request": next(
                    row["initial_request"]
                    for row in load_controller_source_requests(phase3_config)
                    if row["record_id"] == record_id
                ),
                "participant_card": scenarios[record_id]["participant_card"],
                "controller_payload_sha256": controller[record_id]["controller_payload_sha256"],
                "unable": response["unable"],
                "participant_attestation": response["participant_attestation"],
                "public_payload_sha256": response["public_payload_sha256"],
            }
        )

    def as_bytes(records: list[Mapping[str, Any]]) -> bytes:
        if not records:
            return b""
        return ("\n".join(canonical_json(record) for record in records) + "\n").encode("utf-8")

    public_path = participant_dir / "public" / "phase3_clarification_answers.jsonl"
    private_path = participant_dir / "private" / "phase3_private_records.jsonl"
    audit_path = participant_dir / "audit" / "phase3_manifest.json"
    public_bytes = as_bytes(public_records)
    private_bytes = as_bytes(private_records)
    _atomic_write_text(public_path, public_bytes.decode("utf-8"))
    _atomic_write_text(private_path, private_bytes.decode("utf-8"))
    manifest = {
        "schema_version": "prospective-language-pilot-v1-phase3-manifest-1.0.0",
        "study_id": study_config["study_id"],
        "participant_code": state["participant_code"],
        "phase": state["phase"],
        "phase2_qualified_for_clarification_batch": False,
        "exploratory_valid_only_salvage": True,
        "eligible_record_count": len(state["clarification_order"]),
        "locked_response_count": len(public_records),
        "assistant_generation_count_during_phase3": 0,
        "terminal_run_authorized": False,
        "started_at": state["started_at"],
        "updated_at": state["updated_at"],
        "completed_at": state["completed_at"],
        "files": {
            "public_clarification_answers": {
                "relative_path": str(public_path.relative_to(participant_dir)),
                "sha256": sha256_bytes(public_bytes),
                "record_count": len(public_records),
            },
            "private_phase3_records": {
                "relative_path": str(private_path.relative_to(participant_dir)),
                "sha256": sha256_bytes(private_bytes),
                "record_count": len(private_records),
            },
        },
    }
    _atomic_write_json(audit_path, manifest)
    return {"public": public_path, "private": private_path, "audit": audit_path}


def load_controller_source_requests(phase3_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[1]
    phase2_result_path = project_root / phase3_config["phase2"]["result"]
    participant_dir = phase2_result_path.parents[2]
    public_path = participant_dir / "public" / "phase1_initial_requests.jsonl"
    return [json.loads(line) for line in public_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_phase3_bundle(
    phase3_config: Mapping[str, Any], participant_dir: Path | str
) -> dict[str, Any]:
    """Read-only acceptance audit for a completed clarification batch."""

    validate_phase3_config(phase3_config)
    participant_path = Path(participant_dir)
    state_path = phase3_state_path(participant_path)
    public_path = participant_path / "public" / "phase3_clarification_answers.jsonl"
    private_path = participant_path / "private" / "phase3_private_records.jsonl"
    audit_path = participant_path / "audit" / "phase3_manifest.json"
    lock_path = participant_path / "audit" / "phase3_clarification_lock.json"
    missing = [
        str(path)
        for path in (state_path, public_path, private_path, audit_path, lock_path)
        if not path.is_file()
    ]
    if missing:
        raise PilotProtocolError(f"Phase 3 bundle is missing required files: {', '.join(missing)}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest = json.loads(audit_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if lock.get("lock_payload_sha256") != sha256_json(lock_payload):
        raise PilotProtocolError("Phase 3 clarification lock hash mismatch.")
    if state.get("phase3_config_sha256") != sha256_json(phase3_config):
        raise PilotProtocolError("Phase 3 state config hash mismatch.")
    if state.get("phase3_lock_payload_sha256") != lock["lock_payload_sha256"]:
        raise PilotProtocolError("Phase 3 state does not match the clarification lock.")
    if state.get("phase") != "phase_3_complete_waiting_for_terminal_run":
        raise PilotProtocolError("Phase 3 is not complete.")
    expected = phase3_config["phase2"]["required_valid_nonfallback_clarification_records"]
    if len(state.get("responses", {})) != expected:
        raise PilotProtocolError(f"Phase 3 has {len(state.get('responses', {}))} responses; expected {expected}.")

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    public_records = read_jsonl(public_path)
    private_records = read_jsonl(private_path)
    if len(public_records) != expected or len(private_records) != expected:
        raise PilotProtocolError("Phase 3 public/private projection count mismatch.")
    if [row["record_id"] for row in public_records] != state["clarification_order"]:
        raise PilotProtocolError("Phase 3 public record order differs from the frozen order.")
    forbidden_public_keys = {"participant_card", "private_goal", "known_facts", "unable", "research_metadata"}
    if any(forbidden_public_keys & set(record) for record in public_records):
        raise PilotProtocolError("Phase 3 public projection leaks private fields.")
    for record in public_records:
        payload = {key: value for key, value in record.items() if key != "public_payload_sha256"}
        if record.get("public_payload_sha256") != sha256_json(payload):
            raise PilotProtocolError(f"Phase 3 public payload hash mismatch for {record.get('record_id')}.")
        state_record = state["responses"].get(record["record_id"])
        if not state_record or state_record["public_payload_sha256"] != record["public_payload_sha256"]:
            raise PilotProtocolError(f"Phase 3 public record differs from locked state for {record['record_id']}.")

    public_bytes = public_path.read_bytes()
    private_bytes = private_path.read_bytes()
    if not (
        manifest.get("phase") == state["phase"]
        and manifest.get("phase2_qualified_for_clarification_batch") is False
        and manifest.get("exploratory_valid_only_salvage") is True
        and manifest.get("eligible_record_count") == expected
        and manifest.get("locked_response_count") == expected
        and manifest.get("assistant_generation_count_during_phase3") == 0
        and manifest.get("terminal_run_authorized") is False
    ):
        raise PilotProtocolError("Phase 3 manifest boundary or counts are invalid.")
    expected_files = {
        "public_clarification_answers": (public_path, public_bytes),
        "private_phase3_records": (private_path, private_bytes),
    }
    for label, (path, content) in expected_files.items():
        entry = manifest.get("files", {}).get(label, {})
        if (
            entry.get("relative_path") != str(path.relative_to(participant_path))
            or entry.get("sha256") != sha256_bytes(content)
            or entry.get("record_count") != expected
        ):
            raise PilotProtocolError(f"Phase 3 manifest file integrity mismatch for {label}.")

    return {
        "verification": "pass",
        "participant_code": state["participant_code"],
        "locked_response_count": expected,
        "unable_response_count": sum(
            row["response_status"] == "unable_to_answer" for row in public_records
        ),
        "assistant_generation_count_during_phase3": 0,
        "phase2_result_remains_qualified": False,
        "public_projection_sha256": sha256_bytes(public_bytes),
        "private_projection_sha256": sha256_bytes(private_bytes),
        "completed_at": state["completed_at"],
    }
