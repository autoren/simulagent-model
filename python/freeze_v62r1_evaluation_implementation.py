#!/usr/bin/env python3
"""Freeze the V62r1 rescore evaluator before its one execution."""
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
        default="outputs/v62r1-terminal-residual-repair/evaluation-implementation-audit.json",
    )
    parser.add_argument(
        "--output", default="configs/v62r1-evaluation-implementation-lock.json"
    )
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V62r1 evaluation implementation already frozen")
    audit = json.loads(audit_path.read_text())
    implementation_lock_path = PROJECT_ROOT / audit["implementation_lock"]
    if (
        not audit["passed"]
        or not all(audit["checks"].values())
        or not all(audit["gate_mutants_killed"].values())
        or file_sha256(implementation_lock_path) != audit["implementation_lock_sha256"]
        or any(
            file_sha256(PROJECT_ROOT / path) != digest
            for path, digest in audit["evaluation_files_sha256"].items()
        )
    ):
        raise RuntimeError("V62r1 evaluation implementation is not passing and intact")
    implementation_lock = json.loads(implementation_lock_path.read_text())
    lock = {
        "schema_version": "62r1",
        "experiment": "v62r1_evaluation_implementation_lock",
        "implementation_lock": audit["implementation_lock"],
        "implementation_lock_sha256": audit["implementation_lock_sha256"],
        "evaluation_files_sha256": audit["evaluation_files_sha256"],
        "evaluation_implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "evaluation_implementation_audit_sha256": file_sha256(audit_path),
        "source_v62_outcome_lock": implementation_lock["source_v62_outcome_lock"],
        "source_v62_outcome_lock_sha256": implementation_lock[
            "source_v62_outcome_lock_sha256"
        ],
        "source_v62_external_bundle_seal": implementation_lock[
            "source_v62_external_bundle_seal"
        ],
        "source_v62_external_bundle_seal_sha256": implementation_lock[
            "source_v62_external_bundle_seal_sha256"
        ],
        "authorization": {
            "run_one_repair_rescore": True,
            "change_evaluation_implementation": False,
            "rerun_v62_candidate_evaluation": False,
            "rerun_external_rollouts": False,
            "modify_v62_or_v62r1_frozen_artifacts": False,
            "access_human_v58_records": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
