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
            "outputs/v57-definition-augmented-ontology-transfer/"
            "post-result-audit.json"
        ),
    )
    parser.add_argument("--summary", default="docs/v57-results.md")
    parser.add_argument("--output", default="configs/v57-outcome-lock.json")
    args = parser.parse_args()
    audit_path, summary_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.summary, args.output)
    )
    if output.exists():
        raise RuntimeError("V57 outcome already frozen")
    audit = json.loads(audit_path.read_text())
    result_path = PROJECT_ROOT / audit["result"]
    seal_path = PROJECT_ROOT / audit["population_seal"]
    v58_design_path = PROJECT_ROOT / "configs/v58-design-lock.json"
    v58_design = json.loads(v58_design_path.read_text())
    if (
        not audit["passed"]
        or audit["result_sha256"] != file_sha256(result_path)
        or audit["population_seal_sha256"] != file_sha256(seal_path)
        or not v58_design["authorization"][
            "write_and_audit_collection_protocol"
        ]
        or v58_design["authorization"]["collect_pilot_language"]
        or v58_design["authorization"]["collect_evaluation_language"]
    ):
        raise RuntimeError("V57 post-result audit or V58 boundary is invalid")
    qualified = audit["qualification"]["passed"]
    lock = {
        "schema_version": 57,
        "experiment": "v57_outcome_lock",
        "qualification_passed": qualified,
        "result": audit["result"],
        "result_sha256": audit["result_sha256"],
        "population_seal": audit["population_seal"],
        "population_seal_sha256": audit["population_seal_sha256"],
        "post_result_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "post_result_audit_sha256": file_sha256(audit_path),
        "summary": str(summary_path.relative_to(PROJECT_ROOT)),
        "summary_sha256": file_sha256(summary_path),
        "v58_design_lock": str(v58_design_path.relative_to(PROJECT_ROOT)),
        "v58_design_lock_sha256": file_sha256(v58_design_path),
        "qualification": audit["qualification"],
        "decision": (
            "qualify_controlled_definition_conditioned_new_concept_transfer_and_"
            "authorize_v58_collection_protocol_only"
            if qualified else
            "retain_v57_failure_and_localize_failed_gates"
        ),
        "authorization": {
            "write_and_audit_v58_collection_protocol": qualified,
            "freeze_v58_collection_protocol": qualified,
            "generate_blinded_v58_author_packets": False,
            "collect_v58_pilot_language": False,
            "collect_v58_evaluation_language": False,
            "run_v58_candidate_evaluation": False,
            "run_additional_v57_candidate_evaluation": False,
            "human_authored_language_claim": False,
            "open_language_claim": False,
            "joint_new_surface_new_concept_claim": False,
            "probabilistic_or_planning_claim": False,
            "model_access": False,
            "adapter_training": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
