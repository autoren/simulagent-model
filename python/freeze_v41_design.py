#!/usr/bin/env python3
"""Freeze the V41 design and immutable component hashes."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v41-relational-mechanic-confirmation.json")
    parser.add_argument("--audit", default="outputs/v41-relational-mechanic-confirmation/design-audit.json")
    parser.add_argument("--output", default="configs/v41-design-lock.json")
    args = parser.parse_args()
    config_path, audit_path, output = tuple((PROJECT_ROOT / value).resolve() for value in (args.config, args.audit, args.output))
    if output.exists():
        raise RuntimeError("V41 design already frozen")
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V41 design audit did not pass")
    config = json.loads(config_path.read_text())
    source_path = PROJECT_ROOT / config["sourceV40OutcomeLock"]
    compiler_path = PROJECT_ROOT / config["frozenCompiler"]
    kernel_path = PROJECT_ROOT / config["frozenSemanticKernel"]
    v22_config_path = PROJECT_ROOT / config["sourceV22Config"]
    lock = {
        "schema_version": 41,
        "experiment": "v41_design_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": "docs/v41-relational-mechanic-confirmation-plan.md",
        "preregistration_sha256": file_sha256(PROJECT_ROOT / "docs/v41-relational-mechanic-confirmation-plan.md"),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "source_v40_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v40_outcome_lock_sha256": file_sha256(source_path),
        "frozen_compiler": str(compiler_path.relative_to(PROJECT_ROOT)),
        "frozen_compiler_sha256": file_sha256(compiler_path),
        "frozen_semantic_kernel": str(kernel_path.relative_to(PROJECT_ROOT)),
        "frozen_semantic_kernel_sha256": file_sha256(kernel_path),
        "v22_config": str(v22_config_path.relative_to(PROJECT_ROOT)),
        "v22_config_sha256": file_sha256(v22_config_path),
        "old_v22_target_programs": audit["old_v22_target_programs"],
        "unseen_program_capacity": audit["unseen_program_capacity"],
        "authorization": {"write_implementation": True, "construct_confirmation": False, "score_confirmation": False, "model_access": False, "v22r2_evaluation": False, "v28": False},
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
