#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v49-passive-partial-observation.json")
    parser.add_argument("--audit", default="outputs/v49-passive-partial-observation/design-audit.json")
    parser.add_argument("--output", default="configs/v49-design-lock.json")
    args = parser.parse_args()
    config_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.config, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V49 design already frozen")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V49 design audit failed")
    config = json.loads(config_path.read_text())
    source = PROJECT_ROOT / config["sourceV48OutcomeLock"]
    plan = PROJECT_ROOT / "docs/v49-passive-partial-observation-plan.md"
    lock = {
        "schema_version": 49,
        "experiment": "v49_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_v48_outcome_lock": str(source.relative_to(PROJECT_ROOT)),
        "source_v48_outcome_lock_sha256": file_sha256(source),
        "authorization": {
            "write_belief_inference_implementation": True,
            "construct_development_population": False,
            "run_development": False,
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
