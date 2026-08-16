#!/usr/bin/env python3
"""Hash-freeze the complete V36 registry and implementation before readout fitting."""

from __future__ import annotations

import argparse
import hashlib
import json

from audit_v36_implementation import IMPLEMENTATION
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v36-independent-confirmation-design-lock.json")
    parser.add_argument("--audit", default="outputs/v36-independent-confirmation/implementation-audit.json")
    parser.add_argument("--output", default="configs/v36-implementation-lock.json")
    args = parser.parse_args()
    design_path, audit_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.design_lock, args.audit, args.output))
    if output_path.exists():
        raise RuntimeError("V36 implementation lock already exists")
    design, audit = json.loads(design_path.read_text()), json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v36_implementation_lock" or audit["source"]["design_lock_sha256"] != file_sha256(design_path):
        raise RuntimeError("V36 implementation audit does not authorize freeze")
    config = design["config_payload"]
    v32_lock = json.loads((PROJECT_ROOT / config["sourceV32ProtocolLock"]).read_text())
    v34_lock = json.loads((PROJECT_ROOT / config["sourceV34ProtocolLock"]).read_text())
    v35_lock = json.loads((PROJECT_ROOT / config["sourceV35ProtocolLock"]).read_text())
    lock = {
        "schema_version": 36, "experiment": "v36_implementation_lock",
        "design_lock": str(design_path.relative_to(PROJECT_ROOT)), "design_lock_sha256": file_sha256(design_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)), "implementation_audit_sha256": file_sha256(audit_path),
        "config_payload": config, "v32_config_payload": v32_lock["config_payload"], "v34_config_payload": v34_lock["config_payload"], "v35_config_payload": v35_lock["config_payload"],
        "registry": audit["registry"], "registry_sha256": audit["registry_sha256"],
        "training_sources": audit["training_sources"],
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION},
        "authorization": {"fit_interface": True, "interface_fit_runs": 4, "selection_runs": 0, "construct_confirmation": False, "model_access": False, "evaluate_confirmation": False, "reuse_v32_evaluation": False, "run_v28": False, "construct_final_suite": False},
        "data_access_before_lock": audit["data_access"],
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
