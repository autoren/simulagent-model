#!/usr/bin/env python3
"""Freeze V14's single 4B hypothesis-mean baseline before extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


IMPLEMENTATION_PATHS = [
    "python/extract_v14_4b_token_mean_mlx.py",
    "python/evaluate_v14_4b_baseline.py",
    "python/test_v14_4b_baseline.py",
    "python/v14_protocol.py",
    "python/extract_v10_features_mlx.py",
    "python/extract_v11_scale_features_mlx.py",
    "python/extract_v13_token_local_mlx.py",
    "python/evaluate_v12_joint_readout.py",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/v14-4b-baseline.json")
    plan_path = Path("docs/v14-4b-baseline-plan.md")
    output_path = Path("configs/v14-4b-baseline-lock.json")
    config = json.loads(config_path.read_text())
    source_lock_path = Path(config["sourceGroundingLock"])
    manifest_path = Path(config["sourceManifest"])
    shortcut_path = Path(config["sourceShortcutAudit"])
    overlap_path = Path(config["sourcePromptOverlapAudit"])
    source_lock = json.loads(source_lock_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    shortcut = json.loads(shortcut_path.read_text())
    overlap = json.loads(overlap_path.read_text())
    if manifest["grounding_lock_sha256"] != file_sha256(source_lock_path):
        raise RuntimeError("V14 manifest and corpus lock differ")
    if not shortcut["gates"]["passed"] or shortcut["decision"] != "authorize_separately_locked_v14_4b_token_mean_baseline":
        raise RuntimeError("V14 shortcut audit does not authorize the model baseline")
    if not overlap["passed"] or overlap["decision"] != "gate_unique_pairs_on_26_clean_transfer_folds_context_diagnostic_only":
        raise RuntimeError("V14 local prompt overlap audit does not authorize the model baseline")
    v11 = json.loads(Path("configs/v11-frozen-scale-lock.json").read_text())["models"]["qwen35_4b"]
    expected = {
        "model": config["model"], "revision": config["revision"], "configSha256": config["configSha256"],
        "totalLayers": config["totalLayers"], "hiddenSize": config["hiddenSize"],
        "extractionLayer": config["extractionLayer"],
    }
    if v11 != expected:
        raise RuntimeError("V14 model differs from the pinned V11 4B model")
    lock = {
        "schema_version": 14,
        "experiment": "v14_locked_operator_supported_4b_token_mean_baseline",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model": {
            "model_key": config["modelKey"], "model": config["model"], "revision": config["revision"],
            "config_sha256": config["configSha256"], "total_layers": config["totalLayers"],
            "hidden_size": config["hiddenSize"], "extraction_layer": config["extractionLayer"],
        },
        "representation": config["representation"],
        "head": config["head"],
        "seed": config["seed"],
        "max_sequence_length": config["maxSequenceLength"],
        "gates": config["gates"],
        "source": {
            "grounding_lock": str(source_lock_path), "grounding_lock_sha256": file_sha256(source_lock_path),
            "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            "dataset_sha256": manifest["dataset_sha256"],
            "shortcut_audit": str(shortcut_path), "shortcut_audit_sha256": file_sha256(shortcut_path),
            "prompt_overlap_audit": str(overlap_path), "prompt_overlap_audit_sha256": file_sha256(overlap_path),
        },
        "implementation": {path: file_sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "limits": {
            "frozen_4b_extractions_permitted": 1,
            "unique_nli_prompts_permitted": 1512,
            "linear_fits_permitted": 30,
            "alternate_representations_permitted": 0,
            "hyperparameter_searches_permitted": 0,
            "adapter_training_runs_permitted": 0,
            "final_mechanic_evaluations_permitted": 0,
        },
        "data_access": source_lock["data_access"],
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V14 model lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
