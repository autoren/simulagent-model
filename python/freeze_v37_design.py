#!/usr/bin/env python3
"""Hash-freeze the audited V37 development design."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v37-semantic-invariance.json")
    parser.add_argument("--audit", default="outputs/v37-semantic-invariance/design-audit.json")
    parser.add_argument("--output", default="configs/v37-semantic-invariance-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V37 design is already frozen")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v37_design_lock":
        raise RuntimeError("V37 design audit did not pass")
    config = json.loads(config_path.read_text())
    lock = {
        "schema_version": 37,
        "experiment": "v37_semantic_invariance_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "plan": "docs/v37-semantic-invariance-plan.md",
        "plan_sha256": file_sha256(PROJECT_ROOT / "docs/v37-semantic-invariance-plan.md"),
        "authorization": {
            "write_implementation": True,
            "construct_corpus": False,
            "extract_features": False,
            "fit_interface": False,
            "score_validation": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
