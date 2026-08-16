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
            "collection-protocol-audit.json"
        ),
    )
    parser.add_argument(
        "--output", default="configs/v58-collection-protocol-lock.json"
    )
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V58 collection protocol already frozen")
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["protocol_sha256"]
        != file_sha256(PROJECT_ROOT / audit["protocol"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in audit["protocol_files_sha256"].items()
        )
        or audit["v57_outcome_lock_sha256"]
        != file_sha256(PROJECT_ROOT / audit["v57_outcome_lock"])
        or audit["v58_design_lock_sha256"]
        != file_sha256(PROJECT_ROOT / audit["v58_design_lock"])
        or audit["v40_corpus_seal_sha256"]
        != file_sha256(PROJECT_ROOT / audit["v40_corpus_seal"])
    ):
        raise RuntimeError("V58 collection protocol audit is not intact and bound")
    lock = {
        "schema_version": 58,
        "experiment": "v58_collection_protocol_lock",
        "protocol": audit["protocol"],
        "protocol_sha256": audit["protocol_sha256"],
        "collection_protocol_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "collection_protocol_audit_sha256": file_sha256(audit_path),
        "protocol_files_sha256": audit["protocol_files_sha256"],
        "v57_outcome_lock": audit["v57_outcome_lock"],
        "v57_outcome_lock_sha256": audit["v57_outcome_lock_sha256"],
        "v58_design_lock": audit["v58_design_lock"],
        "v58_design_lock_sha256": audit["v58_design_lock_sha256"],
        "v40_corpus_seal": audit["v40_corpus_seal"],
        "v40_corpus_seal_sha256": audit["v40_corpus_seal_sha256"],
        "authorization": {
            "write_and_audit_blinded_author_packet_generator": True,
            "generate_and_audit_blinded_author_packets": True,
            "release_pilot_packets": False,
            "collect_pilot_language": False,
            "write_candidate_parser": False,
            "release_evaluation_packets": False,
            "collect_evaluation_language": False,
            "run_candidate_evaluation": False,
            "model_generated_writing_assistance": False,
            "human_authored_language_claim": False,
            "joint_new_surface_new_concept_claim": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
