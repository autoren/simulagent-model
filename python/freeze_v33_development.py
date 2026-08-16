#!/usr/bin/env python3
"""Hash-lock the bounded V33 development study before head training."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v33_development.py", "python/run_v33_development.py",
    "python/audit_v33_development.py", "python/freeze_v33_development.py",
    "python/audit_and_summarize_v33.py", "python/freeze_v33_outcome.py",
    "python/test_v33_development.py", "python/v32_structured_model.py",
    "python/v32_language.py", "python/v30_language.py",
    "python/audit_v32_factorized_semantics.py", "python/train_v32_heads.py",
    "python/v10_protocol.py", "python/v22r2_grounding.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v33-development-adequacy.json")
    parser.add_argument("--plan", default="docs/v33-development-adequacy-plan.md")
    parser.add_argument("--audit", default="outputs/v33-development-adequacy/pre-run-audit.json")
    parser.add_argument("--output", default="configs/v33-development-adequacy-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, output_path = map(
        lambda value: (PROJECT_ROOT / value).resolve(), (args.config, args.plan, args.audit, args.output)
    )
    if output_path.exists(): raise RuntimeError("V33 development lock already exists")
    if (PROJECT_ROOT / "outputs/v33-development-adequacy/run-attempt.json").exists():
        raise RuntimeError("V33 training began before protocol lock")
    config, audit = json.loads(config_path.read_text()), json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v33_development_lock" or audit["source"]["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V33 pre-run audit does not authorize this config")
    v32_protocol_path = PROJECT_ROOT / config["sourceV32ProtocolLock"]
    v32_protocol = json.loads(v32_protocol_path.read_text())
    metadata_path = PROJECT_ROOT / config["sourceV32FeatureMetadata"]
    metadata = json.loads(metadata_path.read_text())
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    lock = {
        "schema_version": 33, "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)), "config_sha256": file_sha256(config_path),
        "config_payload": config, "v32_config_payload": v32_protocol["config_payload"],
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)), "preregistration_sha256": file_sha256(plan_path),
        "pre_run_audit": str(audit_path.relative_to(PROJECT_ROOT)), "pre_run_audit_sha256": file_sha256(audit_path),
        "source": {
            "v32_protocol_lock": config["sourceV32ProtocolLock"], "v32_protocol_lock_sha256": file_sha256(v32_protocol_path),
            "v32_trained_lock": config["sourceV32TrainedLock"], "v32_trained_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV32TrainedLock"]),
            "v32_post_audit": config["sourceV32PostAudit"], "v32_post_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV32PostAudit"]),
            "feature_metadata": config["sourceV32FeatureMetadata"], "feature_metadata_sha256": file_sha256(metadata_path),
            "feature_artifact": str(feature_path.relative_to(PROJECT_ROOT)), "feature_artifact_sha256": file_sha256(feature_path),
            "allowed_corpus_file_sha256": audit["source"]["allowed_corpus_file_sha256"],
        },
        "search": config["search"], "confirmation": config["confirmation"],
        "qualification": config["qualification"], "limits": config["limits"],
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION},
        "data_access_before_lock": {
            "v32_fit_records_read": 0, "v32_calibration_records_read": 0,
            "v32_evaluation_records_read": 0, "v32_evaluation_features_read": 0,
            "v32_evaluation_predictions_read": 0, "training_paths": 0,
            "backbone_forward_passes": 0, "v28_integration_replays": 0,
            "fresh_suite_constructions": 0,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__": main()
