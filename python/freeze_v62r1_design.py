#!/usr/bin/env python3
"""Freeze the audited V62r1 measurement-repair design."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit", default="outputs/v62r1-terminal-residual-repair/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v62r1-design-lock.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V62r1 design already frozen")
    audit = json.loads(audit_path.read_text())
    bindings = (
        ("config", "config_sha256"),
        ("preregistration", "preregistration_sha256"),
        ("source_v62_outcome_lock", "source_v62_outcome_lock_sha256"),
        (
            "source_v62_evaluation_implementation_lock",
            "source_v62_evaluation_implementation_lock_sha256",
        ),
        ("source_v62_external_bundle_seal", "source_v62_external_bundle_seal_sha256"),
        ("source_post_hoc_diagnostic", "source_post_hoc_diagnostic_sha256"),
    )
    if (
        not audit["passed"]
        or not all(audit["checks"].values())
        or any(
            file_sha256(PROJECT_ROOT / audit[path]) != audit[digest]
            for path, digest in bindings
        )
    ):
        raise RuntimeError("V62r1 design audit is not passing and intact")
    lock = {
        "schema_version": "62r1",
        "experiment": "v62r1_design_lock",
        "config": audit["config"],
        "config_sha256": audit["config_sha256"],
        "config_payload": json.loads((PROJECT_ROOT / audit["config"]).read_text()),
        "preregistration": audit["preregistration"],
        "preregistration_sha256": audit["preregistration_sha256"],
        "source_v62_outcome_lock": audit["source_v62_outcome_lock"],
        "source_v62_outcome_lock_sha256": audit["source_v62_outcome_lock_sha256"],
        "source_v62_evaluation_implementation_lock": audit[
            "source_v62_evaluation_implementation_lock"
        ],
        "source_v62_evaluation_implementation_lock_sha256": audit[
            "source_v62_evaluation_implementation_lock_sha256"
        ],
        "source_v62_external_bundle_seal": audit["source_v62_external_bundle_seal"],
        "source_v62_external_bundle_seal_sha256": audit[
            "source_v62_external_bundle_seal_sha256"
        ],
        "source_post_hoc_diagnostic": audit["source_post_hoc_diagnostic"],
        "source_post_hoc_diagnostic_sha256": audit[
            "source_post_hoc_diagnostic_sha256"
        ],
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "write_and_audit_repair_implementation": True,
            "run_repair_rescore": False,
            "rerun_v62_candidate_evaluation": False,
            "rerun_external_rollouts": False,
            "modify_v62_artifacts": False,
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
