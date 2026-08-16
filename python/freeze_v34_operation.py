#!/usr/bin/env python3
"""Hash-lock V34 before its single frozen-model extraction."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v34_operation.py", "python/audit_v34_operation.py",
    "python/freeze_v34_operation.py", "python/extract_v34_operation_features_mlx.py",
    "python/run_v34_operation.py", "python/audit_and_summarize_v34.py",
    "python/freeze_v34_outcome.py", "python/test_v34_operation.py",
    "python/audit_v32_factorized_semantics.py", "python/extract_v10_features_mlx.py",
    "python/v32_language.py", "python/v10_protocol.py", "python/v22r2_grounding.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v34-operation-interface.json")
    parser.add_argument("--plan", default="docs/v34-operation-interface-plan.md")
    parser.add_argument("--audit", default="outputs/v34-operation-interface/pre-run-audit.json")
    parser.add_argument("--output", default="configs/v34-operation-interface-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, output_path = map(
        lambda value: (PROJECT_ROOT / value).resolve(),
        (args.config, args.plan, args.audit, args.output),
    )
    if output_path.exists():
        raise RuntimeError("V34 protocol lock already exists")
    config, audit = json.loads(config_path.read_text()), json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v34_protocol_lock":
        raise RuntimeError("V34 pre-run audit does not authorize model access")
    if audit["source"]["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V34 config changed after audit")
    lock = {
        "schema_version": 34, "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)), "config_sha256": file_sha256(config_path),
        "config_payload": config, "v32_config_payload": audit["v32_config_payload"],
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)), "preregistration_sha256": file_sha256(plan_path),
        "pre_run_audit": str(audit_path.relative_to(PROJECT_ROOT)), "pre_run_audit_sha256": file_sha256(audit_path),
        "source": {key: value for key, value in audit["source"].items() if key != "config_sha256"},
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION},
        "limits": config["limits"],
        "data_access_before_lock": {"fit_records_read": 0, "calibration_records_read": 0, "v32_evaluation_records_read": 0, "backbone_forward_passes": 0, "ridge_training_fits": 0, "v28_integration_replays": 0},
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
