#!/usr/bin/env python3
"""Freeze the passing V62r1 repair implementation audit."""
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
        default="outputs/v62r1-terminal-residual-repair/implementation-audit.json",
    )
    parser.add_argument("--output", default="configs/v62r1-implementation-lock.json")
    args = parser.parse_args()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    if output.exists():
        raise RuntimeError("V62r1 implementation already frozen")
    audit = json.loads(audit_path.read_text())
    bindings = (
        ("design_lock", "design_lock_sha256"),
        ("implementation", "implementation_sha256"),
        ("tests", "tests_sha256"),
    )
    if (
        not audit["passed"]
        or not all(audit["checks"].values())
        or audit["analytic_fixture_pass_rate"] != 1.0
        or audit["mutant_kill_rate"] != 1.0
        or any(
            file_sha256(PROJECT_ROOT / audit[path]) != audit[digest]
            for path, digest in bindings
        )
    ):
        raise RuntimeError("V62r1 implementation audit is not passing and intact")
    design = json.loads((PROJECT_ROOT / audit["design_lock"]).read_text())
    lock = {
        "schema_version": "62r1",
        "experiment": "v62r1_implementation_lock",
        "design_lock": audit["design_lock"],
        "design_lock_sha256": audit["design_lock_sha256"],
        "implementation": audit["implementation"],
        "implementation_sha256": audit["implementation_sha256"],
        "tests": audit["tests"],
        "tests_sha256": audit["tests_sha256"],
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "source_v62_outcome_lock": design["source_v62_outcome_lock"],
        "source_v62_outcome_lock_sha256": design["source_v62_outcome_lock_sha256"],
        "source_v62_evaluation_implementation_lock": design[
            "source_v62_evaluation_implementation_lock"
        ],
        "source_v62_evaluation_implementation_lock_sha256": design[
            "source_v62_evaluation_implementation_lock_sha256"
        ],
        "source_v62_external_bundle_seal": design["source_v62_external_bundle_seal"],
        "source_v62_external_bundle_seal_sha256": design[
            "source_v62_external_bundle_seal_sha256"
        ],
        "source_post_hoc_diagnostic": design["source_post_hoc_diagnostic"],
        "source_post_hoc_diagnostic_sha256": design[
            "source_post_hoc_diagnostic_sha256"
        ],
        "audit_controls": {
            "analytic_fixture_pass_rate": audit["analytic_fixture_pass_rate"],
            "mutant_kill_rate": audit["mutant_kill_rate"],
            "mutants_killed": audit["mutants_killed"],
        },
        "authorization": {
            "run_one_repair_rescore": True,
            "change_repair_implementation": False,
            "modify_v62_artifacts": False,
            "rerun_v62_candidate_evaluation": False,
            "rerun_external_rollouts": False,
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
