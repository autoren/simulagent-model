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
            "author-packet-audit.json"
        ),
    )
    parser.add_argument(
        "--output", default="configs/v58-author-packet-seal.json"
    )
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V58 author packets already sealed")
    audit = json.loads(audit_path.read_text())
    manifest_path = PROJECT_ROOT / audit["manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        not audit["passed"]
        or audit["manifest_sha256"] != file_sha256(manifest_path)
        or audit["generator_lock_sha256"]
        != file_sha256(PROJECT_ROOT / audit["generator_lock"])
        or audit["protocol_sha256"]
        != file_sha256(PROJECT_ROOT / audit["protocol"])
        or audit["v40_core_sha256"]
        != file_sha256(PROJECT_ROOT / audit["v40_core"])
        or any(
            file_sha256(PROJECT_ROOT / row["path"]) != row["sha256"]
            for row in manifest["artifacts"]
        )
    ):
        raise RuntimeError("V58 author packet audit is not intact and bound")
    seal = {
        "schema_version": 58,
        "experiment": "v58_author_packet_seal",
        "packet_directory": audit["packet_directory"],
        "manifest": audit["manifest"],
        "manifest_sha256": audit["manifest_sha256"],
        "author_packet_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "author_packet_audit_sha256": file_sha256(audit_path),
        "generator_lock": audit["generator_lock"],
        "generator_lock_sha256": audit["generator_lock_sha256"],
        "protocol": audit["protocol"],
        "protocol_sha256": audit["protocol_sha256"],
        "v40_core": audit["v40_core"],
        "v40_core_sha256": audit["v40_core_sha256"],
        "artifacts": manifest["artifacts"],
        "counts": manifest["counts"],
        "release": manifest["release"],
        "human_text": manifest["human_text"],
        "authorization": {
            "prepare_pilot_collection_handoff": True,
            "release_pilot_packets": False,
            "collect_pilot_language": False,
            "activate_evaluation_reserve_packet": False,
            "write_candidate_parser": False,
            "release_evaluation_packets": False,
            "collect_evaluation_language": False,
            "run_candidate_evaluation": False,
            "model_generated_writing_assistance": False,
            "human_authored_language_claim": False,
        },
    }
    seal["seal_payload_sha256"] = hashlib.sha256(
        json.dumps(seal, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(seal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
