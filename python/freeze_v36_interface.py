#!/usr/bin/env python3
"""Freeze V36 parameters before confirmation construction."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v36-implementation-lock.json")
    parser.add_argument("--ledger", default="outputs/v36-independent-confirmation/interface/fit-ledger.json")
    parser.add_argument("--audit", default="outputs/v36-independent-confirmation/interface/audit.json")
    parser.add_argument("--output", default="configs/v36-interface-lock.json")
    args = parser.parse_args()
    implementation_path, ledger_path, audit_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.implementation_lock, args.ledger, args.audit, args.output))
    if output_path.exists():
        raise RuntimeError("V36 interface lock already exists")
    implementation, ledger, audit = (json.loads(path.read_text()) for path in (implementation_path, ledger_path, audit_path))
    if not audit["passed"] or audit["decision"] != "authorize_v36_interface_lock" or audit["source"]["fit_ledger_sha256"] != file_sha256(ledger_path):
        raise RuntimeError("V36 interface audit does not authorize freeze")
    artifact_path = PROJECT_ROOT / ledger["parameter_artifact"]
    lock = {
        "schema_version": 36, "experiment": "v36_frozen_interface",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)), "implementation_lock_sha256": file_sha256(implementation_path),
        "fit_ledger": str(ledger_path.relative_to(PROJECT_ROOT)), "fit_ledger_sha256": file_sha256(ledger_path),
        "interface_audit": str(audit_path.relative_to(PROJECT_ROOT)), "interface_audit_sha256": file_sha256(audit_path),
        "parameter_artifact": str(artifact_path.relative_to(PROJECT_ROOT)), "parameter_artifact_sha256": file_sha256(artifact_path),
        "fixed_components": implementation["config_payload"]["frozenInterface"],
        "authorization": {"construct_confirmation": True, "audit_and_seal_confirmation": True, "model_access": False, "evaluate_confirmation": False, "reuse_v32_evaluation": False, "run_v28": False, "construct_final_suite": False},
        "data_access_before_freeze": ledger["data_access"],
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
