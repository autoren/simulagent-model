#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/v56-symbolic-probabilistic-policy-verification.json",
    )
    parser.add_argument(
        "--plan",
        default="docs/v56-symbolic-probabilistic-policy-verification-plan.md",
    )
    parser.add_argument(
        "--audit",
        default="outputs/v56-symbolic-probabilistic-policy-verification/design-audit.json",
    )
    parser.add_argument("--output", default="configs/v56-design-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V56 design already frozen")
    audit = json.loads(audit_path.read_text())
    source_path = PROJECT_ROOT / audit["source_outcome_lock"]
    if (
        not audit["passed"]
        or audit["config_sha256"] != file_sha256(config_path)
        or audit["preregistration_sha256"] != file_sha256(plan_path)
        or audit["source_outcome_lock_sha256"] != file_sha256(source_path)
    ):
        raise RuntimeError("V56 design audit is not intact and bound")
    config = json.loads(config_path.read_text())
    lock = {
        "schema_version": 56,
        "experiment": "v56_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "config_payload": config,
        "authorization": {
            "install_pinned_verification_dependencies": True,
            "write_and_audit_independent_verifiers": True,
            "construct_v56_verification_bundle": False,
            "run_v56_candidate_formal_verification": False,
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
