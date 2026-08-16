#!/usr/bin/env python3
"""Audit the V58 pilot release gate without any real coordinator declaration."""
from __future__ import annotations

import argparse
import copy
import inspect
import json

from freeze_v58_pilot_release import ATTESTATION_FIELDS, declaration_errors
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


RELEASE_GATE_FILES = (
    "configs/v58-pilot-coordinator-declaration.schema.json",
    "docs/v58-pilot-release-gate.md",
    "python/freeze_v58_pilot_release.py",
    "python/audit_v58_pilot_release_gate.py",
    "python/freeze_v58_pilot_release_gate.py",
)


def _synthetic_declaration(tooling: dict, seal: dict) -> dict:
    pilot_artifacts = [
        row for row in seal["artifacts"]
        if row["anonymous_writer_id"].startswith("pilot_writer_slot_")
    ]
    return {
        "schema_version": 58,
        "experiment": "v58_pilot_coordinator_declaration",
        "declaration_id": "synthetic_release_gate_fixture",
        "timestamp": "2026-08-15T17:00:00Z",
        "coordinator_role_id": "fixture_coordinator",
        "pilot_writer_assignments": [
            {
                "anonymous_writer_slot": row["anonymous_writer_id"],
                "packet_id": row["packet_id"],
                "external_human_role_id": f"fixture_human_writer_{index}",
            }
            for index, row in enumerate(pilot_artifacts)
        ],
        "validator_role_ids": ["fixture_validator_0", "fixture_validator_1"],
        "adjudicator_role_ids": ["fixture_adjudicator_0"],
        "candidate_developer_role_ids": ["fixture_candidate_developer_0"],
        "external_identity_and_consent_record_sha256": "a" * 64,
        "external_collection_storage_descriptor_sha256": "b" * 64,
        "tooling_lock": "configs/v58-pilot-tooling-lock.json",
        "tooling_lock_sha256": file_sha256(
            PROJECT_ROOT / "configs/v58-pilot-tooling-lock.json"
        ),
        "packet_seal": "configs/v58-author-packet-seal.json",
        "packet_seal_sha256": file_sha256(
            PROJECT_ROOT / "configs/v58-author-packet-seal.json"
        ),
        "protocol_sha256": seal["protocol_sha256"],
        "attestations": {key: True for key in ATTESTATION_FIELDS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tooling-lock", default="configs/v58-pilot-tooling-lock.json"
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/v58-human-authored-known-ontology-language/"
            "pilot-release-gate-audit.json"
        ),
    )
    args = parser.parse_args()
    tooling_path = (PROJECT_ROOT / args.tooling_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    tooling = json.loads(tooling_path.read_text())
    seal_path = PROJECT_ROOT / tooling["packet_seal"]
    seal = json.loads(seal_path.read_text())
    errors: list[str] = []

    boundary_ok = (
        tooling["decision"]
        == "tooling_ready_await_separate_external_pilot_release_lock"
        and tooling["authorization"][
            "use_form_renderer_after_valid_pilot_release_lock"
        ]
        and tooling["authorization"][
            "use_intake_after_valid_pilot_release_lock"
        ]
        and not tooling["authorization"]["release_pilot_packets"]
        and not tooling["authorization"]["collect_pilot_language"]
        and not tooling["authorization"]["write_candidate_parser"]
        and file_sha256(seal_path) == tooling["packet_seal_sha256"]
    )
    if not boundary_ok:
        errors.append("V58 frozen tooling boundary is invalid")

    schema = json.loads(
        (PROJECT_ROOT / "configs/v58-pilot-coordinator-declaration.schema.json").read_text()
    )
    schema_ok = (
        set(schema["required"]) == set(schema["properties"])
        and schema["properties"]["pilot_writer_assignments"]["minItems"] == 2
        and schema["properties"]["pilot_writer_assignments"]["maxItems"] == 2
        and schema["properties"]["validator_role_ids"]["minItems"] == 2
        and schema["properties"]["adjudicator_role_ids"]["minItems"] == 1
        and schema["properties"]["candidate_developer_role_ids"]["minItems"] == 1
        and set(schema["properties"]["attestations"]["required"])
        == ATTESTATION_FIELDS
        and all(
            row.get("const") is True
            for row in schema["properties"]["attestations"]["properties"].values()
        )
    )
    if not schema_ok:
        errors.append("V58 coordinator declaration schema is incomplete")

    declaration = _synthetic_declaration(tooling, seal)
    valid_fixture_ok = declaration_errors(declaration, tooling, seal) == []
    if not valid_fixture_ok:
        errors.append("V58 valid synthetic coordinator declaration failed")

    role_overlap = copy.deepcopy(declaration)
    role_overlap["validator_role_ids"][0] = role_overlap[
        "pilot_writer_assignments"
    ][0]["external_human_role_id"]
    false_attestation = copy.deepcopy(declaration)
    false_attestation["attestations"][
        "project_owner_authorizes_pilot_release_only"
    ] = False
    wrong_packet = copy.deepcopy(declaration)
    wrong_packet["pilot_writer_assignments"][0]["packet_id"] = "wrong_packet"
    bad_digest = copy.deepcopy(declaration)
    bad_digest["external_identity_and_consent_record_sha256"] = "not-a-digest"
    no_candidate_role = copy.deepcopy(declaration)
    no_candidate_role["candidate_developer_role_ids"] = []
    evaluation_not_sealed = copy.deepcopy(declaration)
    evaluation_not_sealed["attestations"][
        "evaluation_packets_remain_unreleased"
    ] = False
    attacks = {
        "role_overlap": role_overlap,
        "false_owner_authorization": false_attestation,
        "wrong_packet_assignment": wrong_packet,
        "bad_external_record_digest": bad_digest,
        "missing_candidate_role": no_candidate_role,
        "evaluation_release_attestation_false": evaluation_not_sealed,
    }
    attack_errors = {
        name: declaration_errors(row, tooling, seal)
        for name, row in attacks.items()
    }
    adversarial_ok = all(attack_errors.values()) and {
        "role_census_or_separation",
        "attestations",
        "pilot_assignment",
        "digest_format",
    }.issubset({error for rows in attack_errors.values() for error in rows})
    if not adversarial_ok:
        errors.append("V58 invalid release declarations were not all rejected")

    source = inspect.getsource(declaration_errors)
    output_source = inspect.getsource(
        __import__("freeze_v58_pilot_release").main
    )
    narrow_output_ok = (
        '"release_pilot_packets": True' in output_source
        and '"collect_pilot_language": True' in output_source
        and '"release_evaluation_packets": False' in output_source
        and '"collect_evaluation_language": False' in output_source
        and '"write_candidate_parser": False' in output_source
        and '"model_generated_writing_assistance": False' in output_source
        and "candidate_output" not in source
        and "submitted_text" not in source
    )
    if not narrow_output_ok:
        errors.append("V58 release-lock output is not pilot-only and text-free")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v58-pilot-release-gate-lock.json",
            "configs/v58-pilot-release-lock.json",
            "data/v58-human-authored-known-ontology-language/pilot-submissions",
            "configs/v58-pilot-population-seal.json",
            "configs/v58-candidate-lock.json",
        )
    )
    if not downstream_absent:
        errors.append("V58 release, human text, or candidate artifact already exists")

    audit = {
        "schema_version": 58,
        "experiment": "v58_pilot_release_gate_audit",
        "passed": not errors,
        "decision": (
            "authorize_v58_pilot_release_gate_lock"
            if not errors else "repair_v58_pilot_release_gate"
        ),
        "errors": errors,
        "tooling_lock": str(tooling_path.relative_to(PROJECT_ROOT)),
        "tooling_lock_sha256": file_sha256(tooling_path),
        "packet_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "packet_seal_sha256": file_sha256(seal_path),
        "release_gate_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path)
            for path in RELEASE_GATE_FILES
        },
        "checks": {
            "frozen_tooling_and_unreleased_boundary": boundary_ok,
            "coordinator_declaration_schema": schema_ok,
            "valid_synthetic_declaration": valid_fixture_ok,
            "role_attestation_assignment_and_digest_attacks_rejected": adversarial_ok,
            "pilot_only_text_free_release_lock_output": narrow_output_ok,
            "release_human_text_and_candidate_downstream_absent": downstream_absent,
        },
        "attack_errors": attack_errors,
        "data_access": {
            "real_coordinator_declarations_accessed": 0,
            "human_authored_records_collected": 0,
            "human_authored_text_accessed": 0,
            "pilot_packets_released": 0,
            "evaluation_packets_released": 0,
            "candidate_evaluation_runs": 0,
            "model_forward_passes": 0,
        },
    }
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
