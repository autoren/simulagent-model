#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default=(
            "outputs/v58-human-authored-known-ontology-language/"
            "pilot-release-gate-audit.json"
        ),
    )
    parser.add_argument(
        "--output", default="configs/v58-pilot-release-gate-lock.json"
    )
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V58 pilot release gate already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["tooling_lock_sha256"]
        != file_sha256(PROJECT_ROOT / audit["tooling_lock"])
        or audit["packet_seal_sha256"]
        != file_sha256(PROJECT_ROOT / audit["packet_seal"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in audit["release_gate_files_sha256"].items()
        )
    ):
        raise RuntimeError("V58 pilot release gate audit is not intact and bound")
    lock = {
        "schema_version": 58,
        "experiment": "v58_pilot_release_gate_lock",
        "pilot_release_gate_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "pilot_release_gate_audit_sha256": file_sha256(audit_path),
        "tooling_lock": audit["tooling_lock"],
        "tooling_lock_sha256": audit["tooling_lock_sha256"],
        "packet_seal": audit["packet_seal"],
        "packet_seal_sha256": audit["packet_seal_sha256"],
        "release_gate_files_sha256": audit["release_gate_files_sha256"],
        "decision": "await_valid_external_coordinator_declaration",
        "authorization": {
            "validate_external_declaration_and_freeze_pilot_release": True,
            "release_pilot_packets": False,
            "collect_pilot_language": False,
            "write_candidate_parser": False,
            "release_evaluation_packets": False,
            "collect_evaluation_language": False,
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
