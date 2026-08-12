#!/usr/bin/env python3
"""Freeze V14 corpus generation and pre-model auditing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


IMPLEMENTATION_PATHS = [
    "src/compile-v14.ts",
    "src/v14-contracts.ts",
    "src/v14-grounding.ts",
    "src/v14-validation.ts",
    "tests/v14-grounding.test.ts",
    "src/v9-symbolic.ts",
    "python/v14_protocol.py",
    "python/audit_v14_shortcuts.py",
    "python/test_v14_protocol.py",
    "python/v10_protocol.py",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/dataset.v14.json")
    plan_path = Path("docs/v14-supervision-redesign-plan.md")
    symbolic_lock_path = Path("configs/v9-symbolic-lock.json")
    symbolic_audit_path = Path("outputs/v9-symbolic/oracle-audit.json")
    source_manifest_path = Path("data/v8/manifest.json")
    output_path = Path("configs/v14-grounding-lock.json")
    config = json.loads(config_path.read_text())
    symbolic_audit = json.loads(symbolic_audit_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    if not symbolic_audit["passed"] or symbolic_audit["decision"] != "authorize_v9_grounding_generation":
        raise RuntimeError("V9 symbolic audit does not authorize V14")
    if symbolic_audit["protocol_lock_sha256"] != file_sha256(symbolic_lock_path):
        raise RuntimeError("V14 symbolic audit and lock differ")
    source_hashes = {}
    for source_text in config["sourceRecords"]:
        source_path = Path(source_text)
        relative = str(source_path.relative_to(source_manifest_path.parent))
        if file_sha256(source_path) != source_manifest["artifact_sha256"][relative]:
            raise RuntimeError(f"V14 source artifact changed: {source_text}")
        source_hashes[source_text] = file_sha256(source_path)
    lock = {
        "schema_version": 14,
        "experiment": "v14_locked_operator_supported_corpus_and_shortcut_audit",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config": config,
        "symbolic_lock": str(symbolic_lock_path),
        "symbolic_lock_sha256": file_sha256(symbolic_lock_path),
        "symbolic_audit": str(symbolic_audit_path),
        "symbolic_audit_sha256": file_sha256(symbolic_audit_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_dataset_sha256": source_manifest["dataset_sha256"],
        "source_artifact_sha256": source_hashes,
        "source_v13": {
            "result": "outputs/v13-token-local/evaluation/result.json",
            "result_sha256": file_sha256(Path("outputs/v13-token-local/evaluation/result.json")),
            "operator_support_audit": "outputs/v13-token-local/operator-support-audit.json",
            "operator_support_audit_sha256": file_sha256(Path("outputs/v13-token-local/operator-support-audit.json")),
        },
        "implementation": {path: file_sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "expected_manifest": "data/v14/manifest.json",
        "limits": {
            "development_corpus_generations_permitted": 1,
            "pre_model_shortcut_audits_permitted": 1,
            "frozen_feature_extractions_permitted": 0,
            "classifier_evaluations_permitted": 0,
            "adapter_training_runs_permitted": 0,
            "final_mechanic_evaluations_permitted": 0,
        },
        "data_access": {
            "v3_test_records_read": 0, "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0, "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0, "final_v9_mechanic_records_read": 0,
        },
        "lock_payload_sha256": "",
    }
    if json.loads(Path(lock["source_v13"]["result"]).read_text())["decision"] != "token_local_frozen_readout_insufficient_stop_probes_redesign_supervision":
        raise RuntimeError("V13 did not authorize V14 supervision redesign")
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V14 lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
