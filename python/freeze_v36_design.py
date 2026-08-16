#!/usr/bin/env python3
"""Freeze the V36 scientific design without authorizing construction or model access."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v36-independent-confirmation-design.json")
    parser.add_argument("--plan", default="docs/v36-independent-confirmation-plan.md")
    parser.add_argument("--audit", default="outputs/v36-independent-confirmation/design-audit.json")
    parser.add_argument("--output", default="configs/v36-independent-confirmation-design-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.config, args.plan, args.audit, args.output))
    if output_path.exists():
        raise RuntimeError("V36 design lock already exists")
    config, audit = json.loads(config_path.read_text()), json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v36_design_lock" or audit["source"]["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V36 design audit does not authorize this lock")
    lock = {
        "schema_version": 36, "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)), "config_sha256": file_sha256(config_path), "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)), "preregistration_sha256": file_sha256(plan_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)), "design_audit_sha256": file_sha256(audit_path),
        "audit_script_sha256": file_sha256(PROJECT_ROOT / "python/audit_v36_design.py"),
        "freeze_script_sha256": file_sha256(PROJECT_ROOT / "python/freeze_v36_design.py"),
        "source": audit["source"], "derived_population": audit["derived_population"],
        "authorization": {"write_implementation": True, "fit_interface": False, "construct_confirmation": False, "model_access": False, "reuse_v32_evaluation": False, "run_v28": False, "construct_final_suite": False},
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
