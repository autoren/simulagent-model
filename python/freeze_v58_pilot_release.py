#!/usr/bin/env python3
"""Validate an external coordinator declaration and freeze pilot-only release."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import re
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


DECLARATION_FIELDS = {
    "schema_version", "experiment", "declaration_id", "timestamp",
    "coordinator_role_id", "pilot_writer_assignments", "validator_role_ids",
    "adjudicator_role_ids", "candidate_developer_role_ids",
    "external_identity_and_consent_record_sha256",
    "external_collection_storage_descriptor_sha256", "tooling_lock",
    "tooling_lock_sha256", "packet_seal", "packet_seal_sha256",
    "protocol_sha256", "attestations",
}
ATTESTATION_FIELDS = {
    "all_named_roles_are_real_humans",
    "all_roles_are_disjoint_as_required",
    "pilot_writers_are_adults_or_valid_local_consent_has_been_obtained",
    "pilot_writers_received_no_generative_assistance_briefing",
    "identity_map_and_consent_records_remain_external",
    "collection_storage_is_external_to_candidate_development",
    "evaluation_packets_remain_unreleased",
    "candidate_development_remains_frozen",
    "protocol_and_stop_conditions_acknowledged",
    "project_owner_authorizes_pilot_release_only",
}


def declaration_errors(
    declaration: dict[str, Any],
    tooling: dict[str, Any],
    packet_seal: dict[str, Any],
) -> list[str]:
    errors = []
    if set(declaration) != DECLARATION_FIELDS:
        errors.append("declaration_fields")
    if declaration.get("schema_version") != 58 or declaration.get(
        "experiment"
    ) != "v58_pilot_coordinator_declaration":
        errors.append("declaration_identity")
    timestamp = declaration.get("timestamp")
    try:
        timestamp_ok = (
            isinstance(timestamp, str)
            and timestamp.endswith("Z")
            and datetime.fromisoformat(timestamp[:-1] + "+00:00") is not None
        )
    except ValueError:
        timestamp_ok = False
    if not timestamp_ok:
        errors.append("timestamp")
    if not isinstance(declaration.get("declaration_id"), str) or len(
        declaration["declaration_id"]
    ) < 12:
        errors.append("declaration_id")
    digest_fields = (
        "external_identity_and_consent_record_sha256",
        "external_collection_storage_descriptor_sha256",
        "tooling_lock_sha256",
        "packet_seal_sha256",
        "protocol_sha256",
    )
    if any(
        not isinstance(declaration.get(field), str)
        or re.fullmatch(r"[a-f0-9]{64}", declaration[field]) is None
        for field in digest_fields
    ):
        errors.append("digest_format")
    if (
        declaration.get("tooling_lock") != "configs/v58-pilot-tooling-lock.json"
        or declaration.get("tooling_lock_sha256")
        != file_sha256(PROJECT_ROOT / declaration.get("tooling_lock", "missing"))
        or declaration.get("tooling_lock_sha256")
        != file_sha256(PROJECT_ROOT / "configs/v58-pilot-tooling-lock.json")
        or declaration.get("packet_seal") != "configs/v58-author-packet-seal.json"
        or declaration.get("packet_seal_sha256")
        != file_sha256(PROJECT_ROOT / declaration.get("packet_seal", "missing"))
        or declaration.get("packet_seal_sha256")
        != file_sha256(PROJECT_ROOT / "configs/v58-author-packet-seal.json")
        or declaration.get("protocol_sha256") != packet_seal["protocol_sha256"]
    ):
        errors.append("frozen_binding")
    assignments = declaration.get("pilot_writer_assignments")
    pilot_artifacts = [
        row for row in packet_seal["artifacts"]
        if row["anonymous_writer_id"].startswith("pilot_writer_slot_")
    ]
    expected_pairs = {
        (row["anonymous_writer_id"], row["packet_id"])
        for row in pilot_artifacts
    }
    if (
        not isinstance(assignments, list)
        or len(assignments) != 2
        or any(
            set(row) != {
                "anonymous_writer_slot", "packet_id", "external_human_role_id"
            }
            for row in assignments if isinstance(row, dict)
        )
        or any(not isinstance(row, dict) for row in assignments)
        or {
            (row["anonymous_writer_slot"], row["packet_id"])
            for row in assignments
        } != expected_pairs
        or len({row["external_human_role_id"] for row in assignments}) != 2
    ):
        errors.append("pilot_assignment")
    role_lists = [
        [declaration.get("coordinator_role_id")],
        [row.get("external_human_role_id") for row in assignments]
        if isinstance(assignments, list) else [],
        declaration.get("validator_role_ids"),
        declaration.get("adjudicator_role_ids"),
        declaration.get("candidate_developer_role_ids"),
    ]
    if (
        any(not isinstance(rows, list) or not rows for rows in role_lists)
        or len(role_lists[2]) < 2
        or any(
            not isinstance(role, str) or len(role) < 8
            for rows in role_lists for role in rows
        )
        or len({role for rows in role_lists for role in rows})
        != sum(len(rows) for rows in role_lists)
    ):
        errors.append("role_census_or_separation")
    attestations = declaration.get("attestations")
    if (
        not isinstance(attestations, dict)
        or set(attestations) != ATTESTATION_FIELDS
        or not all(value is True for value in attestations.values())
    ):
        errors.append("attestations")
    if (
        tooling["decision"]
        != "tooling_ready_await_separate_external_pilot_release_lock"
        or not tooling["authorization"][
            "use_form_renderer_after_valid_pilot_release_lock"
        ]
        or not tooling["authorization"][
            "use_intake_after_valid_pilot_release_lock"
        ]
        or tooling["authorization"]["release_pilot_packets"]
        or tooling["authorization"]["collect_pilot_language"]
    ):
        errors.append("tooling_boundary")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-lock", default="configs/v58-pilot-release-gate-lock.json")
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--output", default="configs/v58-pilot-release-lock.json")
    args = parser.parse_args()
    gate_path = (PROJECT_ROOT / args.gate_lock).resolve()
    declaration_path = (PROJECT_ROOT / args.declaration).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V58 pilot release already frozen")
    gate = json.loads(gate_path.read_text())
    if (
        not gate["authorization"][
            "validate_external_declaration_and_freeze_pilot_release"
        ]
        or gate["authorization"]["release_pilot_packets"]
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in gate["release_gate_files_sha256"].items()
        )
    ):
        raise RuntimeError("V58 release gate lock is invalid")
    declaration = json.loads(declaration_path.read_text())
    tooling_path = PROJECT_ROOT / declaration["tooling_lock"]
    seal_path = PROJECT_ROOT / declaration["packet_seal"]
    tooling = json.loads(tooling_path.read_text())
    seal = json.loads(seal_path.read_text())
    errors = declaration_errors(declaration, tooling, seal)
    if errors:
        raise RuntimeError(f"V58 coordinator declaration failed: {errors}")
    pilot_artifacts = [
        row for row in seal["artifacts"]
        if row["anonymous_writer_id"].startswith("pilot_writer_slot_")
    ]
    lock = {
        "schema_version": 58,
        "experiment": "v58_pilot_release_lock",
        "release_gate_lock": str(gate_path.relative_to(PROJECT_ROOT)),
        "release_gate_lock_sha256": file_sha256(gate_path),
        "coordinator_declaration": str(declaration_path.relative_to(PROJECT_ROOT)),
        "coordinator_declaration_sha256": file_sha256(declaration_path),
        "tooling_lock": declaration["tooling_lock"],
        "tooling_lock_sha256": declaration["tooling_lock_sha256"],
        "packet_seal": declaration["packet_seal"],
        "packet_seal_sha256": declaration["packet_seal_sha256"],
        "protocol_sha256": declaration["protocol_sha256"],
        "pilot_packet_artifacts": pilot_artifacts,
        "external_identity_and_consent_record_sha256": declaration[
            "external_identity_and_consent_record_sha256"
        ],
        "external_collection_storage_descriptor_sha256": declaration[
            "external_collection_storage_descriptor_sha256"
        ],
        "authorization": {
            "release_pilot_packets": True,
            "collect_pilot_language": True,
            "use_frozen_offline_renderer": True,
            "use_frozen_pilot_intake": True,
            "release_evaluation_packets": False,
            "collect_evaluation_language": False,
            "write_candidate_parser": False,
            "run_candidate_evaluation": False,
            "model_generated_writing_assistance": False,
            "human_authored_language_claim": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
