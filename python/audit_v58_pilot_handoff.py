#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v58_collection_protocol import (
    ADJUDICATION_FIELDS,
    SUBMISSION_FIELDS,
    VALIDATION_FIELDS,
)


HANDOFF_FILES = (
    "docs/v58-pilot-collection-handoff.md",
    "configs/v58-human-submission.schema.json",
    "configs/v58-human-validation.schema.json",
    "configs/v58-human-adjudication.schema.json",
    "python/audit_v58_pilot_handoff.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-seal", default="configs/v58-author-packet-seal.json"
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/v58-human-authored-known-ontology-language/"
            "pilot-handoff-audit.json"
        ),
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.packet_seal).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    seal = json.loads(seal_path.read_text())
    protocol = json.loads((PROJECT_ROOT / seal["protocol"]).read_text())
    errors: list[str] = []

    seal_ok = (
        seal["authorization"]["prepare_pilot_collection_handoff"]
        and not seal["authorization"]["release_pilot_packets"]
        and not seal["authorization"]["collect_pilot_language"]
        and not seal["authorization"]["write_candidate_parser"]
        and not seal["authorization"]["release_evaluation_packets"]
        and seal["release"] == {
            "pilot_packets_released": 0,
            "evaluation_packets_released": 0,
            "reserve_packets_activated": 0,
        }
        and seal["human_text"] == {
            "collected": 0,
            "accessed": 0,
            "fields_present": 0,
        }
        and all(
            file_sha256(PROJECT_ROOT / row["path"]) == row["sha256"]
            for row in seal["artifacts"]
        )
    )
    if not seal_ok:
        errors.append("V58 packet seal is not intact and handoff-only")

    submission_schema = json.loads(
        (PROJECT_ROOT / "configs/v58-human-submission.schema.json").read_text()
    )
    validation_schema = json.loads(
        (PROJECT_ROOT / "configs/v58-human-validation.schema.json").read_text()
    )
    adjudication_schema = json.loads(
        (PROJECT_ROOT / "configs/v58-human-adjudication.schema.json").read_text()
    )
    schemas_ok = (
        set(submission_schema["required"]) == SUBMISSION_FIELDS
        and set(submission_schema["properties"]) == SUBMISSION_FIELDS
        and set(validation_schema["required"]) == VALIDATION_FIELDS
        and set(validation_schema["properties"]) == VALIDATION_FIELDS
        and set(adjudication_schema["required"]) == ADJUDICATION_FIELDS
        and set(adjudication_schema["properties"]) == ADJUDICATION_FIELDS
        and submission_schema["properties"][
            "consent_and_license_attestation"
        ]["properties"]["datasetLicense"]["const"]
        == protocol["submissionSchema"]["attestation"]["datasetLicense"]
        and set(validation_schema["properties"]["verdict"]["enum"])
        == set(protocol["validation"]["verdicts"])
        and set(adjudication_schema["properties"]["final_verdict"]["enum"])
        == set(protocol["validation"]["verdicts"])
    )
    if not schemas_ok:
        errors.append("V58 handoff schemas do not match the frozen protocol")

    pilot_artifacts = [
        row for row in seal["artifacts"]
        if row["anonymous_writer_id"].startswith("pilot_writer_slot_")
    ]
    packet_content_ok = len(pilot_artifacts) == 2
    for artifact in pilot_artifacts:
        packet = json.loads((PROJECT_ROOT / artifact["path"]).read_text())
        packet_content_ok = packet_content_ok and (
            packet["writer_cohort"] == "pilot"
            and packet["slot_status"] == "active"
            and not packet["release_authorized"]
            and len(packet["prompts"]) == 60
            and {row["construction_family"] for row in packet["prompts"]}
            == set(protocol["constructionSplit"]["pilotExposedFamilies"])
            and all(row["stratum"] == "primary" for row in packet["prompts"])
        )
    if not packet_content_ok:
        errors.append("V58 pilot packet census or holdout boundary is invalid")

    human_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "data/v58-human-authored-known-ontology-language/pilot-submissions",
            "data/v58-human-authored-known-ontology-language/pilot-validations",
            "data/v58-human-authored-known-ontology-language/evaluation-submissions",
            "configs/v58-pilot-population-seal.json",
            "configs/v58-candidate-lock.json",
        )
    )
    if not human_absent:
        errors.append("V58 human text or candidate artifact already exists")

    audit = {
        "schema_version": 58,
        "experiment": "v58_pilot_collection_handoff_audit",
        "passed": not errors,
        "decision": (
            "pilot_handoff_ready_but_release_and_collection_remain_unauthorized"
            if not errors else "repair_v58_pilot_handoff"
        ),
        "errors": errors,
        "packet_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "packet_seal_sha256": file_sha256(seal_path),
        "handoff_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in HANDOFF_FILES
        },
        "pilot_packet_artifacts": pilot_artifacts,
        "checks": {
            "sealed_unreleased_packet_boundary": seal_ok,
            "submission_validation_and_adjudication_schemas": schemas_ok,
            "two_balanced_pilot_packets_and_construction_holdout": packet_content_ok,
            "human_text_and_candidate_absent": human_absent,
        },
        "data_access": {
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
