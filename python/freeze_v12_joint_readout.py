#!/usr/bin/env python3
"""Freeze V12 source artifacts and implementation before fitting readouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


IMPLEMENTATION_PATHS = [
    "python/evaluate_v12_joint_readout.py",
    "python/test_v12_joint_readout.py",
    "python/v10_protocol.py",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/v12-joint-readout.json")
    plan_path = Path("docs/v12-joint-readout-plan.md")
    output_path = Path("configs/v12-joint-readout-lock.json")
    config = json.loads(config_path.read_text())
    v10_lock_path = Path(config["sourceV10FrozenLock"])
    v10_metadata_path = Path(config["sourceV10FeatureMetadata"])
    v10_feature_path = Path(config["sourceV10FeatureArtifact"])
    v10_result_path = Path(config["sourceV10Result"])
    v11_lock_path = Path(config["sourceV11Lock"])
    v11_result_path = Path(config["sourceV11Result"])
    v10_result = json.loads(v10_result_path.read_text())
    v11_result = json.loads(v11_result_path.read_text())
    if v10_result["lora_authorized"] or v11_result["lora_authorized"]:
        raise RuntimeError("V12 requires the prior no-LoRA firewall")
    if v11_result["decision"] != "frozen_scale_insufficient_test_nonlinear_token_aware_readout":
        raise RuntimeError("V11 did not authorize the V12 readout diagnostic")
    if v11_result["final_mechanic_authorized"]:
        raise RuntimeError("V11 unexpectedly authorized final-mechanic access")
    if config["modelOrder"] != ["qwen35_0_8b", "qwen35_4b", "qwen35_9b"]:
        raise RuntimeError("V12 models must run once in increasing-size order")

    models = {}
    for model_key in config["modelOrder"]:
        specification = config["models"][model_key]
        feature_path = Path(specification["featureArtifact"])
        metadata_path = Path(specification["featureMetadata"])
        metadata = json.loads(metadata_path.read_text())
        if metadata["feature_artifact_sha256"] != file_sha256(feature_path):
            raise RuntimeError(f"V12 {model_key} feature artifact differs from its metadata")
        models[model_key] = {
            "feature_artifact": str(feature_path),
            "feature_artifact_sha256": file_sha256(feature_path),
            "feature_metadata": str(metadata_path),
            "feature_metadata_sha256": file_sha256(metadata_path),
        }

    lock = {
        "schema_version": 12,
        "experiment": "v12_locked_frozen_joint_hypothesis_readout",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model_order": config["modelOrder"],
        "models": models,
        "primary_head": config["primaryHead"],
        "conditional_head": config["conditionalHead"],
        "seed": config["seed"],
        "gates": config["gates"],
        "folds": json.loads(v10_lock_path.read_text())["folds"],
        "source_v10": {
            "frozen_lock": str(v10_lock_path),
            "frozen_lock_sha256": file_sha256(v10_lock_path),
            "feature_metadata": str(v10_metadata_path),
            "feature_metadata_sha256": file_sha256(v10_metadata_path),
            "feature_artifact": str(v10_feature_path),
            "feature_artifact_sha256": file_sha256(v10_feature_path),
            "result": str(v10_result_path),
            "result_sha256": file_sha256(v10_result_path),
        },
        "source_v11": {
            "frozen_lock": str(v11_lock_path),
            "frozen_lock_sha256": file_sha256(v11_lock_path),
            "result": str(v11_result_path),
            "result_sha256": file_sha256(v11_result_path),
        },
        "implementation": {path: file_sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "limits": {
            "new_feature_extractions_permitted": 0,
            "linear_24_fold_evaluations_permitted": 3,
            "conditional_mlp_24_fold_evaluations_permitted": 3,
            "hyperparameter_searches_permitted": 0,
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
        raise RuntimeError(f"Refusing to overwrite changed V12 lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
