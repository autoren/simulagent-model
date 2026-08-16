#!/usr/bin/env python3
"""Pre-generation audit of the deterministic V58 packet builder."""
from __future__ import annotations

import argparse
import inspect
import json

from audit_v58_author_packets import audit_packet_objects
from generate_v58_author_packets import build_packets, read_jsonl
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


GENERATOR_FILES = (
    "python/generate_v58_author_packets.py",
    "python/audit_v58_author_packet_generator.py",
    "python/audit_v58_author_packets.py",
    "python/seal_v58_author_packets.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-lock", default="configs/v58-collection-protocol-lock.json"
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/v58-human-authored-known-ontology-language/"
            "author-packet-generator-audit.json"
        ),
    )
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.protocol_lock).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    lock = json.loads(lock_path.read_text())
    protocol_path = PROJECT_ROOT / lock["protocol"]
    protocol = json.loads(protocol_path.read_text())
    source_path = PROJECT_ROOT / protocol["knownOntologySource"]["corePopulation"]
    source_rows = read_jsonl(source_path)
    errors: list[str] = []

    lock_ok = (
        lock["authorization"]["write_and_audit_blinded_author_packet_generator"]
        and lock["authorization"]["generate_and_audit_blinded_author_packets"]
        and not lock["authorization"]["release_pilot_packets"]
        and not lock["authorization"]["collect_pilot_language"]
        and not lock["authorization"]["release_evaluation_packets"]
        and not lock["authorization"]["collect_evaluation_language"]
        and file_sha256(protocol_path) == lock["protocol_sha256"]
        and file_sha256(PROJECT_ROOT / lock["collection_protocol_audit"])
        == lock["collection_protocol_audit_sha256"]
    )
    if not lock_ok:
        errors.append("V58 protocol lock is not intact or packet-authorized")

    source_code = inspect.getsource(build_packets)
    firewall_ok = all(token not in source_code for token in (
        "submitted_text", "candidate", "model", "evaluation_output"
    ))
    if not firewall_ok:
        errors.append("V58 packet builder crosses text or candidate firewall")

    audit_seed = lock["lock_payload_sha256"] + "|altered-generator-audit"
    packets = build_packets(protocol, source_rows, audit_seed)
    object_audit = audit_packet_objects(packets, protocol, source_rows)
    deterministic_ok = packets == build_packets(protocol, source_rows, audit_seed)
    altered_ok = object_audit["passed"] and deterministic_ok
    if not altered_ok:
        errors.append("V58 altered-seed packet fixture failed")

    leak_mutant = json.loads(json.dumps(packets))
    first_prompt = leak_mutant[0]["prompts"][0]
    first_source = next(
        row for row in source_rows
        if row["id"] == first_prompt["source_record_id"]
    )
    first_prompt["reference_surface_realization"] = first_source[
        "agent_input"
    ]["evidence_text"]
    duplicate_mutant = json.loads(json.dumps(packets))
    duplicate_mutant[0]["prompts"][1]["source_record_id"] = (
        duplicate_mutant[0]["prompts"][0]["source_record_id"]
    )
    release_mutant = json.loads(json.dumps(packets))
    release_mutant[0]["release_authorized"] = True
    mutants_ok = not any(
        audit_packet_objects(mutant, protocol, source_rows)["passed"]
        for mutant in (leak_mutant, duplicate_mutant, release_mutant)
    )
    if not mutants_ok:
        errors.append("V58 packet leak, duplicate, or release mutant survived")

    downstream_absent = not any(
        (PROJECT_ROOT / path).exists()
        for path in (
            "configs/v58-author-packet-generator-lock.json",
            "configs/v58-author-packet-seal.json",
            "data/v58-human-authored-known-ontology-language/author-packets",
            "data/v58-human-authored-known-ontology-language/pilot-submissions",
            "data/v58-human-authored-known-ontology-language/evaluation-submissions",
        )
    )
    if not downstream_absent:
        errors.append("V58 author packets or human submissions already exist")

    audit = {
        "schema_version": 58,
        "experiment": "v58_author_packet_generator_audit",
        "passed": not errors,
        "decision": (
            "authorize_v58_author_packet_generator_lock"
            if not errors else "repair_v58_author_packet_generator"
        ),
        "errors": errors,
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "v40_core": str(source_path.relative_to(PROJECT_ROOT)),
        "v40_core_sha256": file_sha256(source_path),
        "generator_files_sha256": {
            path: file_sha256(PROJECT_ROOT / path) for path in GENERATOR_FILES
        },
        "checks": {
            "protocol_lock_authorization_and_binding": lock_ok,
            "human_text_candidate_and_model_firewall": firewall_ok,
            "altered_seed_packet_census_balance_and_determinism": altered_ok,
            "leak_duplicate_and_release_mutants_rejected": mutants_ok,
            "packet_and_human_text_downstream_absent": downstream_absent,
        },
        "altered_seed_metrics": object_audit["metrics"],
        "data_access": {
            "human_authored_records_collected": 0,
            "human_authored_text_accessed": 0,
            "packets_released": 0,
            "synthetic_text_free_prompts_generated": object_audit["metrics"][
                "prompts"
            ],
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
