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
            "evaluation-implementation-audit.json"
        ),
    )
    parser.add_argument(
        "--population-seal", default="configs/v57-population-seal.json"
    )
    parser.add_argument(
        "--output", default="configs/v57-evaluation-implementation-lock.json"
    )
    args = parser.parse_args()
    audit_path, seal_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.audit, args.population_seal, args.output)
    )
    if output.exists():
        raise RuntimeError("V57 evaluation implementation already frozen")
    audit = json.loads(audit_path.read_text())
    seal = json.loads(seal_path.read_text())
    if (
        not audit["passed"]
        or audit["population_seal_sha256"] != file_sha256(seal_path)
        or audit["manifest_sha256"]
        != file_sha256(PROJECT_ROOT / seal["manifest"])
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for section in (
                "evaluation_files_sha256",
                "frozen_dependencies_sha256",
            )
            for path, digest in audit[section].items()
        )
    ):
        raise RuntimeError("V57 evaluation audit is not intact and bound")
    lock = {
        "schema_version": 57,
        "experiment": "v57_evaluation_implementation_lock",
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "implementation_lock": audit["implementation_lock"],
        "implementation_lock_sha256": audit["implementation_lock_sha256"],
        "manifest": audit["manifest"],
        "manifest_sha256": audit["manifest_sha256"],
        "evaluation_implementation_audit": str(
            audit_path.relative_to(PROJECT_ROOT)
        ),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "evaluation_files_sha256": audit["evaluation_files_sha256"],
        "frozen_dependencies_sha256": audit["frozen_dependencies_sha256"],
        "authorization": {
            "run_one_v57_candidate_evaluation": True,
            "run_additional_v57_candidate_evaluation": False,
            "change_evaluation_implementation": False,
            "modify_v57_population": False,
            "collect_human_language": False,
            "joint_new_surface_new_concept_claim": False,
            "open_language_claim": False,
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
