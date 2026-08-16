#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v50-history-dependent-belief-filtering.json")
    parser.add_argument("--audit", default="outputs/v50-history-dependent-belief-filtering/design-audit.json")
    parser.add_argument("--output", default="configs/v50-design-lock.json")
    args = parser.parse_args()
    config_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve() for value in (args.config, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V50 design already frozen")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V50 design audit failed")
    config = json.loads(config_path.read_text())
    source = PROJECT_ROOT / config["sourceV49OutcomeLock"]
    plan = PROJECT_ROOT / "docs/v50-history-dependent-belief-filtering-plan.md"
    lock = {
        "schema_version": 50,
        "experiment": "v50_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_v49_outcome_lock": str(source.relative_to(PROJECT_ROOT)),
        "source_v49_outcome_lock_sha256": file_sha256(source),
        "authorization": {
            "write_history_dependent_implementation": True,
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
