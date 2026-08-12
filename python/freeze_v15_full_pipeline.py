#!/usr/bin/env python3
"""Freeze V15 full-pipeline extraction and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


IMPLEMENTATION_PATHS = [
    "python/extract_v15_full_features_mlx.py",
    "python/evaluate_v15_full_pipeline.py",
    "python/test_v15_full_pipeline.py",
    "python/v14_protocol.py",
    "python/evaluate_v10_frozen.py",
    "python/v9_symbolic.py",
    "python/binary_metrics.py",
    "python/extract_v10_features_mlx.py",
    "python/extract_v11_scale_features_mlx.py",
    "python/extract_v13_token_local_mlx.py",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/v15-full-pipeline.json")
    plan_path = Path("docs/v15-full-pipeline-plan.md")
    output_path = Path("configs/v15-full-pipeline-lock.json")
    config = json.loads(config_path.read_text())
    grounding_lock_path = Path(config["sourceV14GroundingLock"])
    manifest_path = Path(config["sourceV14Manifest"])
    model_lock_path = Path(config["sourceV14ModelLock"])
    model_result_path = Path(config["sourceV14ModelResult"])
    model_features_path = Path(config["sourceV14ModelFeatures"])
    grounding_lock = json.loads(grounding_lock_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    model_lock = json.loads(model_lock_path.read_text())
    model_result = json.loads(model_result_path.read_text())
    if model_result["decision"] != "operator_supported_surface_transfer_passes_repair_temporal_then_full_pipeline":
        raise RuntimeError("V14 did not authorize V15 full-pipeline recomposition")
    if not model_result["primary_transfer_gates"]["passed"]:
        raise RuntimeError("V14 primary transfer gates did not pass")
    if model_result["lora_authorized"] or model_result["final_mechanic_authorized"]:
        raise RuntimeError("V14 firewall state is inconsistent with V15")
    expected = {
        "model_key": config["modelKey"], "model": config["model"], "revision": config["revision"],
        "config_sha256": config["configSha256"], "total_layers": config["totalLayers"],
        "hidden_size": config["hiddenSize"], "extraction_layer": config["extractionLayer"],
    }
    if model_lock["model"] != expected:
        raise RuntimeError("V15 model differs from V14")
    if manifest["grounding_lock_sha256"] != file_sha256(grounding_lock_path):
        raise RuntimeError("V15 manifest and grounding lock differ")
    lock = {
        "schema_version": 15,
        "experiment": "v15_locked_operator_supported_frozen_full_pipeline",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "config_path": str(config_path), "config_sha256": file_sha256(config_path),
        "model": expected, "c_value": config["cValue"], "seed": config["seed"],
        "max_sequence_length": config["maxSequenceLength"], "gates": config["gates"],
        "source": {
            "grounding_lock": str(grounding_lock_path), "grounding_lock_sha256": file_sha256(grounding_lock_path),
            "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            "dataset_sha256": manifest["dataset_sha256"],
            "v14_model_lock": str(model_lock_path), "v14_model_lock_sha256": file_sha256(model_lock_path),
            "v14_model_result": str(model_result_path), "v14_model_result_sha256": file_sha256(model_result_path),
            "v14_feature_artifact": str(model_features_path),
            "v14_feature_artifact_sha256": file_sha256(model_features_path),
        },
        "implementation": {path: file_sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "limits": {
            "new_model_forward_passes_permitted": 13554,
            "reused_nli_features_permitted": 1512,
            "linear_fold_evaluations_permitted": 30,
            "alternate_representations_permitted": 0,
            "adapter_training_runs_permitted": 0,
            "final_mechanic_evaluations_permitted": 0,
        },
        "data_access": grounding_lock["data_access"],
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V15 lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
