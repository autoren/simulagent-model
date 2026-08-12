#!/usr/bin/env python3
"""Freeze V11 before either larger frozen backbone is accessed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from v10_protocol import file_sha256


IMPLEMENTATION_PATHS = [
    "python/extract_v11_scale_features_mlx.py",
    "python/evaluate_v11_frozen_scale.py",
    "python/test_v11_frozen_scale.py",
    "python/extract_v10_features_mlx.py",
    "python/evaluate_v10_frozen.py",
    "python/v10_protocol.py",
    "python/v9_symbolic.py",
    "python/binary_metrics.py",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_layer(total_layers: int, depth_rule: dict[str, Any]) -> int:
    if depth_rule["rounding"] != "nearest_integer":
        raise RuntimeError("V11 only supports its locked nearest-integer depth rule")
    return round(total_layers * depth_rule["referenceExtractionLayer"] / depth_rule["referenceLayers"])


def main() -> None:
    config_path = Path("configs/v11-frozen-scale.json")
    plan_path = Path("docs/v11-frozen-scale-plan.md")
    output_path = Path("configs/v11-frozen-scale-lock.json")
    config = json.loads(config_path.read_text())
    v10_lock_path = Path(config["sourceV10FrozenLock"])
    v10_manifest_path = Path(config["sourceV10Manifest"])
    v10_metadata_path = Path(config["sourceV10FeatureMetadata"])
    v10_feature_path = Path(config["sourceV10FeatureArtifact"])
    v10_result_path = Path(config["sourceV10Result"])
    v10_lock = json.loads(v10_lock_path.read_text())
    v10_manifest = json.loads(v10_manifest_path.read_text())
    v10_metadata = json.loads(v10_metadata_path.read_text())
    v10_result = json.loads(v10_result_path.read_text())
    if v10_result["decision"] != "authorize_separately_locked_larger_frozen_capacity_diagnostic":
        raise RuntimeError("V10 result did not authorize a frozen scale diagnostic")
    if v10_result["lora_authorized"] or v10_result["larger_frozen_model_run_in_v10"]:
        raise RuntimeError("V10 firewall state is inconsistent with V11")
    if v10_manifest["dataset_sha256"] != v10_lock["dataset_sha256"]:
        raise RuntimeError("V10 manifest and frozen lock identities differ")
    if v10_metadata["feature_artifact_sha256"] != file_sha256(v10_feature_path):
        raise RuntimeError("V10 reference feature artifact changed")
    if v10_result["feature_artifact_sha256"] != v10_metadata["feature_artifact_sha256"]:
        raise RuntimeError("V10 result and feature identities differ")
    if config["runOrder"] != ["qwen35_4b", "qwen35_9b"]:
        raise RuntimeError("V11 must run both larger models in locked order")
    if config["depthRule"]["fraction"] != 0.25:
        raise RuntimeError("V11 homologous depth must remain one quarter")

    models = {}
    for model_key in config["runOrder"]:
        specification = config["models"][model_key]
        if expected_layer(specification["totalLayers"], config["depthRule"]) != specification["extractionLayer"]:
            raise RuntimeError(f"V11 {model_key} extraction layer violates depth rule")
        snapshot = Path(snapshot_download(
            repo_id=specification["model"],
            revision=specification["revision"],
            allow_patterns=["config.json"],
            local_files_only=True,
        ))
        model_config_path = snapshot / "config.json"
        if file_sha256(model_config_path) != specification["configSha256"]:
            raise RuntimeError(f"V11 {model_key} model config hash differs")
        model_config = json.loads(model_config_path.read_text())
        text_config = model_config["text_config"]
        if (
            text_config["num_hidden_layers"] != specification["totalLayers"]
            or text_config["hidden_size"] != specification["hiddenSize"]
            or model_config["quantization"]["bits"] != 4
            or model_config["quantization"]["mode"] != "affine"
        ):
            raise RuntimeError(f"V11 {model_key} architecture differs from preregistration")
        models[model_key] = specification

    lock = {
        "schema_version": 11,
        "experiment": "v11_locked_frozen_scale_capacity_diagnostic",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "depth_rule": config["depthRule"],
        "models": models,
        "run_order": config["runOrder"],
        "protocol": v10_lock["protocol"],
        "folds": v10_lock["folds"],
        "source_v10": {
            "frozen_lock": str(v10_lock_path),
            "frozen_lock_sha256": file_sha256(v10_lock_path),
            "manifest": str(v10_manifest_path),
            "manifest_sha256": file_sha256(v10_manifest_path),
            "dataset_sha256": v10_manifest["dataset_sha256"],
            "feature_metadata": str(v10_metadata_path),
            "feature_metadata_sha256": file_sha256(v10_metadata_path),
            "feature_artifact": str(v10_feature_path),
            "feature_artifact_sha256": file_sha256(v10_feature_path),
            "result": str(v10_result_path),
            "result_sha256": file_sha256(v10_result_path),
        },
        "implementation": {path: file_sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "limits": {
            "frozen_feature_extractions_permitted": 2,
            "complete_24_fold_model_evaluations_permitted": 2,
            "alternative_layer_extractions_permitted": 0,
            "adapter_training_runs_permitted": 0,
            "final_mechanic_evaluations_permitted": 0,
        },
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
            "final_v9_mechanic_records_read": 0,
        },
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V11 lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
