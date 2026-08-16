#!/usr/bin/env python3
"""Hash-freeze the V38 focus-parser design without authorizing execution."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v38-ontology-anchored-focus-parser.json")
    parser.add_argument("--audit", default="outputs/v38-ontology-anchored-focus-parser/design-audit.json")
    parser.add_argument("--output", default="configs/v38-ontology-anchored-focus-parser-lock.json")
    args = parser.parse_args()
    config_path, audit_path, output_path = (
        (PROJECT_ROOT / value).resolve() for value in (args.config, args.audit, args.output)
    )
    if output_path.exists():
        raise RuntimeError("V38 design is already frozen")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v38_design_lock":
        raise RuntimeError("V38 design audit did not pass")
    config = json.loads(config_path.read_text())
    lock = {
        "schema_version": 38,
        "experiment": "v38_ontology_anchored_focus_parser_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": "docs/v38-ontology-anchored-focus-parser-plan.md",
        "preregistration_sha256": file_sha256(PROJECT_ROOT / "docs/v38-ontology-anchored-focus-parser-plan.md"),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "write_implementation": True,
            "construct_corpus": False,
            "model_access": False,
            "fit_parser": False,
            "score_validation": False,
            "v32_evaluation": False,
            "v28": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
