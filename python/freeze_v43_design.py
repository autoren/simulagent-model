#!/usr/bin/env python3
"""Freeze the audited V43 sequential-language design."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v43-sequential-language-grounding.json")
    parser.add_argument("--audit", default="outputs/v43-sequential-language-grounding/design-audit.json")
    parser.add_argument("--output", default="configs/v43-design-lock.json")
    args = parser.parse_args()
    config_path, audit_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.config, args.audit, args.output))
    if output.exists():
        raise RuntimeError("V43 design already frozen")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V43 design audit did not pass")
    config = json.loads(config_path.read_text())
    source_path = PROJECT_ROOT / config["sourceV42OutcomeLock"]
    seal_path = PROJECT_ROOT / config["sourceV42CorpusSeal"]
    plan_path = PROJECT_ROOT / "docs/v43-sequential-language-grounding-plan.md"
    lock = {
        "schema_version": 43,
        "experiment": "v43_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_v42_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v42_outcome_lock_sha256": file_sha256(source_path),
        "source_v42_corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "source_v42_corpus_seal_sha256": file_sha256(seal_path),
        "authorization": {
            "write_and_audit_implementation": True,
            "construct_paired_language_corpus": False,
            "run_paired_development": False,
            "preregister_delayed_effects": False,
            "model_access": False,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
