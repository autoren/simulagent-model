#!/usr/bin/env python3
"""Freeze the audited V61 implementation."""
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v61-design-lock.json")
    parser.add_argument(
        "--audit", default="outputs/v61-long-horizon-policy-verification/implementation-audit.json"
    )
    parser.add_argument("--output", default="configs/v61-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.design_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V61 implementation is already frozen")
    design = json.loads(design_path.read_text())
    audit = json.loads(audit_path.read_text())
    implementation_path = PROJECT_ROOT / audit["implementation"]
    tests_path = PROJECT_ROOT / audit["unit_tests"]
    expected_versions = {
        "storm": design["config_payload"]["probabilisticVerification"]["version"],
        "z3": design["config_payload"]["independentVerification"]["z3SolverVersion"],
    }
    bound = (
        audit["passed"]
        and audit["design_lock_sha256"] == file_sha256(design_path)
        and audit["implementation_sha256"] == file_sha256(implementation_path)
        and audit["unit_tests_sha256"] == file_sha256(tests_path)
        and audit["analytic_fixture_pass_rate"] == 1.0
        and audit["mutation_kill_rate"] == 1.0
        and all(audit["independence_checks"].values())
        and audit["tool_versions"] == expected_versions
    )
    if not bound:
        raise RuntimeError("V61 implementation audit is not intact and bound")
    lock = {
        "schema_version": 61,
        "experiment": "v61_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation": audit["implementation"],
        "implementation_sha256": file_sha256(implementation_path),
        "unit_tests": audit["unit_tests"],
        "unit_tests_sha256": file_sha256(tests_path),
        "tool_versions": audit["tool_versions"],
        "analytic_fixture_pass_rate": audit["analytic_fixture_pass_rate"],
        "mutation_kill_rate": audit["mutation_kill_rate"],
        "authorization": {
            "reconstruct_v60_source_policies": True,
            "construct_and_audit_v61_verification_bundle": True,
            "write_and_audit_v61_candidate_runner": True,
            "run_v61_candidate_verification": False,
            "access_v59_audit_truth": False,
            "rerun_v60_evaluation": False,
            "formal_safety_claim": False,
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
