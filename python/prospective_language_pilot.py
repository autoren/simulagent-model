"""Core storage and validation for the prospective language pilot.

The Streamlit UI imports this module, but the module itself has no Streamlit
dependency.  That keeps the protocol, hashing, and public/private projections
unit-testable with the standard library.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


STUDY_SCHEMA_VERSION = "1.0.0"
SESSION_SCHEMA_VERSION = "1.0.0"
ALLOWED_UNABLE_REASONS = {
    "scenario_unclear",
    "would_not_ask_an_assistant",
    "cannot_form_natural_request",
    "other",
}
PARTICIPANT_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,31}$")


class PilotProtocolError(ValueError):
    """Raised when a protocol or immutable-session invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def normalize_participant_code(value: str) -> str:
    normalized = value.strip().upper()
    if not PARTICIPANT_CODE_PATTERN.fullmatch(normalized):
        raise PilotProtocolError(
            "Participant code must be 2–32 characters using only letters, numbers, '-' or '_'."
        )
    return normalized


def load_study_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PilotProtocolError(f"Study config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise PilotProtocolError(f"Study config is not valid JSON: {exc}") from exc
    validate_study_config(config)
    return config


def validate_study_config(config: Mapping[str, Any]) -> None:
    required_top_level = {
        "study_id",
        "schema_version",
        "phase_1",
        "participant_instructions",
        "assistant_policy_contract",
        "scenarios",
    }
    missing = sorted(required_top_level - set(config))
    if missing:
        raise PilotProtocolError(f"Study config missing fields: {', '.join(missing)}")
    if config["schema_version"] != STUDY_SCHEMA_VERSION:
        raise PilotProtocolError(
            f"Unsupported study schema {config['schema_version']!r}; expected {STUDY_SCHEMA_VERSION!r}."
        )

    phase_1 = config["phase_1"]
    if phase_1.get("assistant_generation_enabled") is not False:
        raise PilotProtocolError("Phase 1 must disable assistant generation.")
    if phase_1.get("assistant_run_authorized") is not False:
        raise PilotProtocolError("Phase 1 must not authorize an assistant run.")
    if phase_1.get("show_assistant_outputs_before_all_initial_requests_lock") is not False:
        raise PilotProtocolError("Phase 1 must hide all assistant outputs.")

    scenarios = config["scenarios"]
    required_count = int(phase_1["required_scenario_count"])
    if len(scenarios) != required_count:
        raise PilotProtocolError(
            f"Expected {required_count} scenarios but found {len(scenarios)}."
        )

    seen_ids: set[str] = set()
    for scenario in scenarios:
        required_scenario = {
            "record_id",
            "title",
            "participant_card",
            "assistant_visible_context",
            "research_metadata",
        }
        scenario_missing = sorted(required_scenario - set(scenario))
        if scenario_missing:
            raise PilotProtocolError(
                f"Scenario missing fields {scenario_missing}: {scenario.get('record_id', '<unknown>')}"
            )
        record_id = scenario["record_id"]
        if record_id in seen_ids:
            raise PilotProtocolError(f"Duplicate record ID: {record_id}")
        if not re.fullmatch(r"H\d{3}", record_id):
            raise PilotProtocolError(f"Record ID must be opaque H### form: {record_id}")
        seen_ids.add(record_id)

        card = scenario["participant_card"]
        if not all(key in card for key in ("setting", "private_goal", "known_facts")):
            raise PilotProtocolError(f"Incomplete participant card: {record_id}")
        if not isinstance(card["known_facts"], list) or not card["known_facts"]:
            raise PilotProtocolError(f"Scenario requires known facts: {record_id}")


def study_config_hash(config: Mapping[str, Any]) -> str:
    validate_study_config(config)
    return sha256_json(config)


def scenario_index(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {scenario["record_id"]: scenario for scenario in config["scenarios"]}


def deterministic_scenario_order(config: Mapping[str, Any], participant_code: str) -> list[str]:
    participant = normalize_participant_code(participant_code)
    salt = str(config["randomization_salt"])
    study_id = str(config["study_id"])

    def order_key(record_id: str) -> str:
        return sha256_text(f"{study_id}:{salt}:{participant}:{record_id}")

    return sorted((scenario["record_id"] for scenario in config["scenarios"]), key=order_key)


def participant_directory(
    storage_root: Path | str, config: Mapping[str, Any], participant_code: str
) -> Path:
    participant = normalize_participant_code(participant_code)
    return Path(storage_root) / str(config["study_id"]) / participant


def session_state_path(participant_dir: Path | str) -> Path:
    return Path(participant_dir) / "private" / "session_state.json"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def initialize_or_load_session(
    config: Mapping[str, Any], participant_code: str, storage_root: Path | str
) -> tuple[Path, dict[str, Any]]:
    validate_study_config(config)
    participant = normalize_participant_code(participant_code)
    participant_dir = participant_directory(storage_root, config, participant)
    state_path = session_state_path(participant_dir)
    config_hash = study_config_hash(config)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _validate_session_compatibility(state, config, participant, config_hash)
        export_phase_1_bundles(config, participant_dir, state)
        return participant_dir, state

    now = utc_now()
    state = {
        "session_schema_version": SESSION_SCHEMA_VERSION,
        "study_id": config["study_id"],
        "study_config_sha256": config_hash,
        "participant_code": participant,
        "scenario_order": deterministic_scenario_order(config, participant),
        "phase": "phase_1_initial_request_collection",
        "started_at": now,
        "updated_at": now,
        "phase_1_completed_at": None,
        "initial_responses": {},
    }
    _atomic_write_json(state_path, state)
    export_phase_1_bundles(config, participant_dir, state)
    return participant_dir, state


def _validate_session_compatibility(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    participant_code: str,
    config_hash: str,
) -> None:
    if state.get("session_schema_version") != SESSION_SCHEMA_VERSION:
        raise PilotProtocolError("Existing session uses an unsupported schema version.")
    if state.get("study_id") != config["study_id"]:
        raise PilotProtocolError("Existing session belongs to a different study.")
    if state.get("participant_code") != participant_code:
        raise PilotProtocolError("Existing session belongs to a different participant code.")
    if state.get("study_config_sha256") != config_hash:
        raise PilotProtocolError(
            "The frozen study config changed after this session began. Preserve the session and use a new study ID."
        )
    expected_ids = {scenario["record_id"] for scenario in config["scenarios"]}
    if set(state.get("scenario_order", [])) != expected_ids:
        raise PilotProtocolError("Existing session scenario order does not match the frozen population.")
    if not set(state.get("initial_responses", {})).issubset(expected_ids):
        raise PilotProtocolError("Existing session contains an unknown record ID.")


def completed_count(state: Mapping[str, Any]) -> int:
    return len(state.get("initial_responses", {}))


def next_incomplete_record_id(state: Mapping[str, Any]) -> str | None:
    completed = state.get("initial_responses", {})
    return next((record_id for record_id in state["scenario_order"] if record_id not in completed), None)


def phase_1_is_complete(config: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    return completed_count(state) == len(config["scenarios"])


def lock_initial_response(
    config: Mapping[str, Any],
    participant_dir: Path | str,
    state: dict[str, Any],
    *,
    record_id: str,
    initial_request: str,
    unable_reason: str | None,
    unable_note: str,
    participant_attestation: bool,
) -> dict[str, Any]:
    validate_study_config(config)
    if state.get("phase") != "phase_1_initial_request_collection":
        raise PilotProtocolError("Phase 1 is already complete; no further requests can be added.")
    expected_record_id = next_incomplete_record_id(state)
    if record_id != expected_record_id:
        raise PilotProtocolError(
            f"Records must be locked in frozen order; expected {expected_record_id}, received {record_id}."
        )
    if record_id in state["initial_responses"]:
        raise PilotProtocolError(f"Record {record_id} is already locked and cannot be edited.")
    if not participant_attestation:
        raise PilotProtocolError("Confirm that the response uses your own wording before locking it.")

    request = initial_request.strip()
    note = unable_note.strip()
    minimum_characters = int(config["phase_1"]["minimum_request_characters"])
    if unable_reason is None:
        if len(request) < minimum_characters:
            raise PilotProtocolError(
                f"Write at least {minimum_characters} characters, or use the unable-to-respond option."
            )
        response_status = "submitted"
        unable_payload = None
    else:
        if unable_reason not in ALLOWED_UNABLE_REASONS:
            raise PilotProtocolError(f"Unsupported unable reason: {unable_reason}")
        if request:
            raise PilotProtocolError("Leave the request blank when using unable-to-respond.")
        response_status = "unable_to_respond"
        unable_payload = {"reason": unable_reason, "note": note or None}

    scenario = scenario_index(config)[record_id]
    locked_at = utc_now()
    position = state["scenario_order"].index(record_id) + 1
    public_payload = {
        "schema_version": "1.0.0",
        "study_id": config["study_id"],
        "participant_code": state["participant_code"],
        "record_id": record_id,
        "display_position": position,
        "scenario_title": scenario["title"],
        "assistant_visible_context": scenario["assistant_visible_context"],
        "response_status": response_status,
        "initial_request": request or None,
        "locked_at": locked_at,
    }
    public_payload_hash = sha256_json(public_payload)
    response_record = {
        "record_id": record_id,
        "display_position": position,
        "response_status": response_status,
        "initial_request": request or None,
        "unable": unable_payload,
        "participant_attestation": True,
        "locked_at": locked_at,
        "public_payload_sha256": public_payload_hash,
    }
    state["initial_responses"][record_id] = response_record
    state["updated_at"] = locked_at

    if phase_1_is_complete(config, state):
        state["phase"] = "phase_1_complete_waiting_for_assistant_run"
        state["phase_1_completed_at"] = locked_at

    participant_path = Path(participant_dir)
    _atomic_write_json(session_state_path(participant_path), state)
    export_phase_1_bundles(config, participant_path, state)
    return response_record


def _jsonl_bytes(records: list[Mapping[str, Any]]) -> bytes:
    if not records:
        return b""
    return ("\n".join(canonical_json(record) for record in records) + "\n").encode("utf-8")


def export_phase_1_bundles(
    config: Mapping[str, Any], participant_dir: Path | str, state: Mapping[str, Any]
) -> dict[str, Path]:
    participant_path = Path(participant_dir)
    index = scenario_index(config)
    public_records: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []

    for record_id in state["scenario_order"]:
        if record_id not in state["initial_responses"]:
            continue
        response = state["initial_responses"][record_id]
        scenario = index[record_id]
        public_payload = {
            "schema_version": "1.0.0",
            "study_id": config["study_id"],
            "participant_code": state["participant_code"],
            "record_id": record_id,
            "display_position": response["display_position"],
            "scenario_title": scenario["title"],
            "assistant_visible_context": scenario["assistant_visible_context"],
            "response_status": response["response_status"],
            "initial_request": response["initial_request"],
            "locked_at": response["locked_at"],
        }
        payload_hash = sha256_json(public_payload)
        if payload_hash != response["public_payload_sha256"]:
            raise PilotProtocolError(f"Locked public payload hash mismatch for {record_id}.")
        public_records.append({**public_payload, "public_payload_sha256": payload_hash})
        private_records.append(
            {
                "schema_version": "1.0.0",
                "study_id": config["study_id"],
                "study_config_sha256": state["study_config_sha256"],
                "participant_code": state["participant_code"],
                "record_id": record_id,
                "display_position": response["display_position"],
                "scenario_title": scenario["title"],
                "participant_card": scenario["participant_card"],
                "research_metadata": scenario["research_metadata"],
                "response_status": response["response_status"],
                "initial_request": response["initial_request"],
                "unable": response["unable"],
                "participant_attestation": response["participant_attestation"],
                "locked_at": response["locked_at"],
                "public_payload_sha256": payload_hash,
            }
        )

    public_path = participant_path / "public" / "phase1_initial_requests.jsonl"
    private_path = participant_path / "private" / "phase1_private_records.jsonl"
    audit_path = participant_path / "audit" / "phase1_manifest.json"
    public_bytes = _jsonl_bytes(public_records)
    private_bytes = _jsonl_bytes(private_records)
    _atomic_write_text(public_path, public_bytes.decode("utf-8"))
    _atomic_write_text(private_path, private_bytes.decode("utf-8"))

    complete = phase_1_is_complete(config, state)
    manifest = {
        "schema_version": "1.0.0",
        "study_id": config["study_id"],
        "study_config_sha256": state["study_config_sha256"],
        "participant_code": state["participant_code"],
        "phase": state["phase"],
        "phase_1_complete": complete,
        "assistant_generation_count": 0,
        "assistant_run_authorized": False,
        "scenario_count": len(config["scenarios"]),
        "locked_record_count": len(public_records),
        "scenario_order": state["scenario_order"],
        "started_at": state["started_at"],
        "updated_at": state["updated_at"],
        "phase_1_completed_at": state["phase_1_completed_at"],
        "files": {
            "public_initial_requests": {
                "relative_path": str(public_path.relative_to(participant_path)),
                "sha256": sha256_bytes(public_bytes),
                "record_count": len(public_records),
            },
            "private_phase_1_records": {
                "relative_path": str(private_path.relative_to(participant_path)),
                "sha256": sha256_bytes(private_bytes),
                "record_count": len(private_records),
            },
        },
        "locked_public_payloads": {
            record["record_id"]: record["public_payload_sha256"] for record in public_records
        },
    }
    _atomic_write_json(audit_path, manifest)
    return {"public": public_path, "private": private_path, "audit": audit_path}


def load_export_bytes(participant_dir: Path | str) -> dict[str, bytes]:
    participant_path = Path(participant_dir)
    paths = {
        "public": participant_path / "public" / "phase1_initial_requests.jsonl",
        "private": participant_path / "private" / "phase1_private_records.jsonl",
        "audit": participant_path / "audit" / "phase1_manifest.json",
    }
    return {name: path.read_bytes() for name, path in paths.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PilotProtocolError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise PilotProtocolError(f"Expected an object at {path}:{line_number}.")
        records.append(record)
    return records


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_nested_keys(child))
    return keys


def verify_phase_1_bundle(
    config: Mapping[str, Any], participant_dir: Path | str
) -> dict[str, Any]:
    """Independently verify a completed Phase 1 bundle without rewriting it."""

    validate_study_config(config)
    participant_path = Path(participant_dir)
    state_path = session_state_path(participant_path)
    public_path = participant_path / "public" / "phase1_initial_requests.jsonl"
    private_path = participant_path / "private" / "phase1_private_records.jsonl"
    audit_path = participant_path / "audit" / "phase1_manifest.json"
    required_paths = (state_path, public_path, private_path, audit_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise PilotProtocolError(f"Phase 1 bundle is missing required files: {', '.join(missing)}")

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PilotProtocolError(f"Phase 1 state or manifest is invalid JSON: {exc}") from exc

    participant = normalize_participant_code(str(state.get("participant_code", "")))
    _validate_session_compatibility(state, config, participant, study_config_hash(config))
    required_count = len(config["scenarios"])
    if state.get("phase") != "phase_1_complete_waiting_for_assistant_run":
        raise PilotProtocolError("Phase 1 is not in its completed waiting state.")
    if completed_count(state) != required_count or not phase_1_is_complete(config, state):
        raise PilotProtocolError(
            f"Phase 1 has {completed_count(state)} locked records; expected {required_count}."
        )

    public_bytes = public_path.read_bytes()
    private_bytes = private_path.read_bytes()
    public_records = _read_jsonl(public_path)
    private_records = _read_jsonl(private_path)
    if len(public_records) != required_count or len(private_records) != required_count:
        raise PilotProtocolError("Public and private projections must each contain every scenario.")

    forbidden_public_keys = {
        "participant_card",
        "private_goal",
        "known_facts",
        "research_metadata",
        "unable",
        "participant_attestation",
    }
    leaked_keys = sorted(forbidden_public_keys & _nested_keys(public_records))
    if leaked_keys:
        raise PilotProtocolError(f"Public projection leaks private keys: {', '.join(leaked_keys)}")

    observed_order = [record.get("record_id") for record in public_records]
    if observed_order != state["scenario_order"]:
        raise PilotProtocolError("Public record order differs from the frozen scenario order.")
    for record in public_records:
        record_id = record.get("record_id")
        stored_hash = record.get("public_payload_sha256")
        payload = {key: value for key, value in record.items() if key != "public_payload_sha256"}
        if stored_hash != sha256_json(payload):
            raise PilotProtocolError(f"Public payload hash mismatch for {record_id}.")
        state_record = state["initial_responses"].get(record_id)
        if not state_record or state_record.get("public_payload_sha256") != stored_hash:
            raise PilotProtocolError(f"Public payload does not match locked state for {record_id}.")

    if manifest.get("study_config_sha256") != study_config_hash(config):
        raise PilotProtocolError("Audit manifest config hash does not match the frozen config.")
    if manifest.get("participant_code") != participant:
        raise PilotProtocolError("Audit manifest participant code does not match the session.")
    if manifest.get("phase") != state["phase"] or manifest.get("phase_1_complete") is not True:
        raise PilotProtocolError("Audit manifest does not certify a completed Phase 1.")
    if manifest.get("locked_record_count") != required_count:
        raise PilotProtocolError("Audit manifest locked-record count is incorrect.")
    if manifest.get("assistant_generation_count") != 0:
        raise PilotProtocolError("Assistant-generation count must be exactly zero in Phase 1.")
    if manifest.get("assistant_run_authorized") is not False:
        raise PilotProtocolError("Audit manifest must not authorize an assistant run.")

    expected_files = {
        "public_initial_requests": (public_path, public_bytes, len(public_records)),
        "private_phase_1_records": (private_path, private_bytes, len(private_records)),
    }
    manifest_files = manifest.get("files", {})
    for label, (path, content, record_count) in expected_files.items():
        entry = manifest_files.get(label, {})
        if entry.get("relative_path") != str(path.relative_to(participant_path)):
            raise PilotProtocolError(f"Audit manifest path mismatch for {label}.")
        if entry.get("sha256") != sha256_bytes(content):
            raise PilotProtocolError(f"Audit manifest file hash mismatch for {label}.")
        if entry.get("record_count") != record_count:
            raise PilotProtocolError(f"Audit manifest record count mismatch for {label}.")

    return {
        "verification": "pass",
        "study_id": config["study_id"],
        "participant_code": participant,
        "study_config_sha256": study_config_hash(config),
        "locked_record_count": required_count,
        "assistant_generation_count": 0,
        "public_projection_sha256": sha256_bytes(public_bytes),
        "private_projection_sha256": sha256_bytes(private_bytes),
        "phase_1_completed_at": state["phase_1_completed_at"],
    }
