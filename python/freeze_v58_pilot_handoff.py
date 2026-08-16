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
            "pilot-handoff-audit.json"
        ),
    )
    parser.add_argument(
        "--output", default="configs/v58-pilot-collection-handoff-lock.json"
    )
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V58 pilot collection handoff already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["packet_seal_sha256"]
        != file_sha256(PROJECT_ROOT / audit["packet_seal"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in audit["handoff_files_sha256"].items()
        )
    ):
        raise RuntimeError("V58 pilot handoff audit is not intact and bound")
    lock = {
        "schema_version": 58,
        "experiment": "v58_pilot_collection_handoff_lock",
        "packet_seal": audit["packet_seal"],
        "packet_seal_sha256": audit["packet_seal_sha256"],
        "pilot_handoff_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "pilot_handoff_audit_sha256": file_sha256(audit_path),
        "handoff_files_sha256": audit["handoff_files_sha256"],
        "pilot_packet_artifacts": audit["pilot_packet_artifacts"],
        "decision": "await_external_human_pilot_coordinator_and_release_authorization",
        "authorization": {
            "show_readiness_to_user": True,
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
