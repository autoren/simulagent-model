#!/usr/bin/env python3
"""Freeze the V13 token-local extraction and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


IMPLEMENTATION_PATHS = [
    "python/extract_v13_token_local_mlx.py",
    "python/evaluate_v13_token_local.py",
    "python/test_v13_token_local.py",
    "python/extract_v10_features_mlx.py",
    "python/extract_v11_scale_features_mlx.py",
    "python/evaluate_v12_joint_readout.py",
    "python/v10_protocol.py",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/v13-token-local.json")
    plan_path = Path("docs/v13-token-local-plan.md")
    output_path = Path("configs/v13-token-local-lock.json")
    config = json.loads(config_path.read_text())
    v10_lock_path = Path(config["sourceV10FrozenLock"])
    v10_feature_path = Path(config["sourceV10FeatureArtifact"])
    v11_lock_path = Path(config["sourceV11Lock"])
    v11_metadata_path = Path(config["sourceV11FeatureMetadata"])
    v12_lock_path = Path(config["sourceV12Lock"])
    v12_result_path = Path(config["sourceV12Result"])
    v11_lock = json.loads(v11_lock_path.read_text())
    v11_metadata = json.loads(v11_metadata_path.read_text())
    v12_result = json.loads(v12_result_path.read_text())
    if v12_result["decision"] != "frozen_final_state_joint_readout_insufficient_extract_token_span_interactions":
        raise RuntimeError("V12 did not authorize V13 token-local extraction")
    if v12_result["lora_authorized"] or v12_result["final_mechanic_authorized"]:
        raise RuntimeError("V12 firewall state is inconsistent with V13")
    model = v11_lock["models"][config["modelKey"]]
    expected = {
        "model": config["model"], "revision": config["revision"],
        "configSha256": config["configSha256"], "totalLayers": config["totalLayers"],
        "hiddenSize": config["hiddenSize"], "extractionLayer": config["extractionLayer"],
    }
    if model != expected:
        raise RuntimeError("V13 4B model differs from the frozen V11 specification")
    if v11_metadata["feature_artifact_sha256"] != file_sha256(Path(v11_metadata["feature_artifact"])):
        raise RuntimeError("V13 V11 reference metadata is inconsistent")

    lock = {
        "schema_version": 13,
        "experiment": "v13_locked_4b_token_local_relation_diagnostic",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model": {
            "model_key": config["modelKey"], "model": config["model"],
            "revision": config["revision"], "config_sha256": config["configSha256"],
            "total_layers": config["totalLayers"], "hidden_size": config["hiddenSize"],
            "extraction_layer": config["extractionLayer"],
        },
        "representations": config["representations"],
        "linear_heads": config["linearHeads"],
        "conditional_head": config["conditionalHead"],
        "seed": config["seed"],
        "max_sequence_length": config["maxSequenceLength"],
        "gates": config["gates"],
        "folds": json.loads(v10_lock_path.read_text())["folds"],
        "source_v10": {
            "frozen_lock": str(v10_lock_path),
            "frozen_lock_sha256": file_sha256(v10_lock_path),
            "feature_artifact": str(v10_feature_path),
            "feature_artifact_sha256": file_sha256(v10_feature_path),
        },
        "source_v11": {
            "frozen_lock": str(v11_lock_path),
            "frozen_lock_sha256": file_sha256(v11_lock_path),
            "feature_metadata": str(v11_metadata_path),
            "feature_metadata_sha256": file_sha256(v11_metadata_path),
        },
        "source_v12": {
            "frozen_lock": str(v12_lock_path),
            "frozen_lock_sha256": file_sha256(v12_lock_path),
            "result": str(v12_result_path),
            "result_sha256": file_sha256(v12_result_path),
        },
        "implementation": {path: file_sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "limits": {
            "frozen_4b_nli_extractions_permitted": 1,
            "alternative_models_or_layers_permitted": 0,
            "linear_24_fold_evaluations_permitted": 2,
            "conditional_mlp_24_fold_evaluations_permitted": 1,
            "hyperparameter_searches_permitted": 0,
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
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V13 lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
