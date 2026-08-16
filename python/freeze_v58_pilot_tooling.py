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
            "pilot-tooling-audit.json"
        ),
    )
    parser.add_argument(
        "--output", default="configs/v58-pilot-tooling-lock.json"
    )
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V58 pilot tooling already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["handoff_lock_sha256"]
        != file_sha256(PROJECT_ROOT / audit["handoff_lock"])
        or audit["packet_seal_sha256"]
        != file_sha256(PROJECT_ROOT / audit["packet_seal"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in audit["tooling_files_sha256"].items()
        )
    ):
        raise RuntimeError("V58 pilot tooling audit is not intact and bound")
    lock = {
        "schema_version": 58,
        "experiment": "v58_pilot_tooling_lock",
        "pilot_tooling_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "pilot_tooling_audit_sha256": file_sha256(audit_path),
        "handoff_lock": audit["handoff_lock"],
        "handoff_lock_sha256": audit["handoff_lock_sha256"],
        "packet_seal": audit["packet_seal"],
        "packet_seal_sha256": audit["packet_seal_sha256"],
        "tooling_files_sha256": audit["tooling_files_sha256"],
        "decision": "tooling_ready_await_separate_external_pilot_release_lock",
        "authorization": {
            "use_form_renderer_after_valid_pilot_release_lock": True,
            "use_intake_after_valid_pilot_release_lock": True,
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
