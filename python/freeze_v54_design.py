#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v54-exact-one-step-eig.json")
    parser.add_argument("--plan", default="docs/v54-exact-one-step-eig-plan.md")
    parser.add_argument("--audit", default="outputs/v54-exact-one-step-eig/design-audit.json")
    parser.add_argument("--output", default="configs/v54-design-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V54 design already frozen")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    source_path = (PROJECT_ROOT / config["sourceV53r2OutcomeLock"]).resolve()
    if (
        not audit["passed"]
        or audit["config_sha256"] != file_sha256(config_path)
        or audit["preregistration_sha256"] != file_sha256(plan_path)
        or audit["source_outcome_lock_sha256"] != file_sha256(source_path)
    ):
        raise RuntimeError("V54 design audit is not bound to config, plan, and V53r2 outcome")
    lock = {
        "schema_version": 54,
        "experiment": "v54_design_lock",
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
            "write_and_audit_exact_eig_implementation": True,
            "construct_v54_active_populations": False,
            "run_v54_active_evaluation": False,
            "reward_or_planning": False,
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
