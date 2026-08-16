#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v58-human-authored-known-ontology-language.json")
    parser.add_argument("--plan", default="docs/v58-human-authored-known-ontology-language-plan.md")
    parser.add_argument("--audit", default="outputs/v58-human-authored-known-ontology-language/design-audit.json")
    parser.add_argument("--output", default="configs/v58-design-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, output = tuple(
        (PROJECT_ROOT / value).resolve()
        for value in (args.config, args.plan, args.audit, args.output)
    )
    if output.exists():
        raise RuntimeError("V58 design already frozen")
    audit = json.loads(audit_path.read_text())
    source_path = PROJECT_ROOT / audit["source_outcome_lock"]
    if (
        not audit["passed"]
        or audit["config_sha256"] != file_sha256(config_path)
        or audit["preregistration_sha256"] != file_sha256(plan_path)
        or audit["source_outcome_lock_sha256"] != file_sha256(source_path)
    ):
        raise RuntimeError("V58 design audit is not intact and bound")
    lock = {
        "schema_version": 58,
        "experiment": "v58_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_outcome_lock_sha256": file_sha256(source_path),
        "config_payload": json.loads(config_path.read_text()),
        "authorization": {
            "write_and_audit_collection_protocol": True,
            "generate_blinded_author_packets": False,
            "collect_pilot_language": False,
            "collect_evaluation_language": False,
            "write_candidate_parser": False,
            "run_candidate_evaluation": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
