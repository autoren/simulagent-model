#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v56-design-lock.json")
    parser.add_argument(
        "--audit",
        default="outputs/v56-symbolic-probabilistic-policy-verification/implementation-audit.json",
    )
    parser.add_argument("--output", default="configs/v56-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.design_lock, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V56 implementation already frozen")
    design = json.loads(design_path.read_text())
    audit = json.loads(audit_path.read_text())
    implementation_path = PROJECT_ROOT / audit["implementation"]
    unit_test_path = PROJECT_ROOT / audit["unit_tests"]
    if (
        not audit["passed"]
        or audit["design_lock_sha256"] != file_sha256(design_path)
        or audit["implementation_sha256"] != file_sha256(implementation_path)
        or audit["unit_tests_sha256"] != file_sha256(unit_test_path)
        or audit["tool_versions"] != {
            "storm": design["config_payload"]["probabilisticVerification"]["version"],
            "z3": design["config_payload"]["symbolicVerification"]["solverVersion"],
        }
    ):
        raise RuntimeError("V56 implementation audit is not intact and bound")
    lock = {
        "schema_version": 56,
        "experiment": "v56_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "implementation": audit["implementation"],
        "implementation_sha256": file_sha256(implementation_path),
        "unit_tests": audit["unit_tests"],
        "unit_tests_sha256": file_sha256(unit_test_path),
        "tool_versions": audit["tool_versions"],
        "exhaustive_symbolic_cases": audit["exhaustive_symbolic_audit"]["cases"],
        "mutation_kill_rate": audit["mutation_controls"]["kill_rate"],
        "authorization": {
            "construct_and_audit_v56_verification_bundle": True,
            "write_and_audit_v56_candidate_runner": True,
            "run_v56_candidate_formal_verification": False,
            "modify_v55_or_v55r1_results": False,
            "formal_safety_claim": False,
            "language_grounding": False,
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
