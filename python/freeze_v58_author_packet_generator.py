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
            "author-packet-generator-audit.json"
        ),
    )
    parser.add_argument(
        "--output", default="configs/v58-author-packet-generator-lock.json"
    )
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V58 author packet generator already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["protocol_lock_sha256"]
        != file_sha256(PROJECT_ROOT / audit["protocol_lock"])
        or audit["protocol_sha256"]
        != file_sha256(PROJECT_ROOT / audit["protocol"])
        or audit["v40_core_sha256"]
        != file_sha256(PROJECT_ROOT / audit["v40_core"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in audit["generator_files_sha256"].items()
        )
    ):
        raise RuntimeError("V58 packet generator audit is not intact and bound")
    seed = hashlib.sha256(
        (audit["protocol_lock_sha256"] + "|canonical-v58-author-packets").encode()
    ).hexdigest()
    lock = {
        "schema_version": 58,
        "experiment": "v58_author_packet_generator_lock",
        "protocol_lock": audit["protocol_lock"],
        "protocol_lock_sha256": audit["protocol_lock_sha256"],
        "protocol": audit["protocol"],
        "protocol_sha256": audit["protocol_sha256"],
        "v40_core": audit["v40_core"],
        "v40_core_sha256": audit["v40_core_sha256"],
        "generator_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "generator_audit_sha256": file_sha256(audit_path),
        "generator_files_sha256": audit["generator_files_sha256"],
        "packet_generation_seed": seed,
        "authorization": {
            "generate_v58_blinded_author_packets_once": True,
            "regenerate_v58_blinded_author_packets": False,
            "release_pilot_packets": False,
            "collect_pilot_language": False,
            "release_evaluation_packets": False,
            "collect_evaluation_language": False,
            "write_candidate_parser": False,
            "run_candidate_evaluation": False,
            "model_generated_writing_assistance": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
