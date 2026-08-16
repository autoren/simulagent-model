#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v53-continuous-parameter-smc2.json")
    parser.add_argument("--plan", default="docs/v53-continuous-parameter-smc2-plan.md")
    parser.add_argument(
        "--audit", default="outputs/v53-continuous-parameter-smc2/design-audit.json"
    )
    parser.add_argument("--output", default="configs/v53-design-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V53 design already frozen")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (
        not audit["passed"]
        or audit["config_sha256"] != file_sha256(config_path)
        or audit["preregistration_sha256"] != file_sha256(plan_path)
    ):
        raise RuntimeError("V53 design audit is not bound to config and plan")
    lock = {
        "schema_version": 53,
        "experiment": "v53_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_outcome_lock": audit["source_outcome_lock"],
        "source_outcome_lock_sha256": audit["source_outcome_lock_sha256"],
        "config_payload": config,
        "authorization": {
            "write_and_audit_smc_squared_implementation": True,
            "construct_v53_populations": False,
            "run_v53_evaluation": False,
            "active_intervention_selection": False,
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
