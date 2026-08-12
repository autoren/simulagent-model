#!/usr/bin/env python3
"""Freeze the development-only V8 structured-head protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = {
    "component_extractor": "python/extract_v8_structured_components_mlx.py",
    "structured_head": "python/train_v8_action_conditioned_head.py",
    "structured_head_tests": "python/test_v8_structured_head.py",
    "forward_core": "python/extract_v6_development_features_mlx.py",
    "binary_metrics": "python/binary_metrics.py",
    "stage3_diagnostic": "python/run_v8_lomo_diagnostics.py",
}

FORBIDDEN_ACCESS = (
    "v3_test_records_read",
    "prior_holdout_records_read",
    "v7_tone_drift_records_read",
    "v7_model_results_read",
    "untouched_v8_mechanic_records_read",
)

LEGACY_UNTOUCHED_ACCESS = (
    "untouched_v8_mechanic_records_created",
    "untouched_v8_mechanic_model_scores_read",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-lock", default="configs/v8-development-lock.json")
    parser.add_argument("--stage3-result", default="outputs/v8-frozen-diagnostics/lomo-result.json")
    parser.add_argument("--stage3-features", default="outputs/v8-frozen-diagnostics/features/metadata.json")
    parser.add_argument("--config", default="configs/v8-structured-head.json")
    parser.add_argument("--plan", default="docs/v8-structured-head-plan.md")
    parser.add_argument("--output", default="configs/v8-structured-head-lock.json")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def no_forbidden_access(value: dict[str, Any], label: str) -> None:
    access = value.get("data_access", value)
    if any(access.get(key, 0) != 0 for key in (*FORBIDDEN_ACCESS, *LEGACY_UNTOUCHED_ACCESS)):
        raise RuntimeError(f"{label} crossed the V8 data firewall")


def main() -> None:
    args = parse_args()
    stage3_lock_path = Path(args.stage3_lock)
    stage3_result_path = Path(args.stage3_result)
    stage3_metadata_path = Path(args.stage3_features)
    config_path = Path(args.config)
    plan_path = Path(args.plan)
    output_path = Path(args.output)

    stage3_lock = json.loads(stage3_lock_path.read_text())
    stage3_result = json.loads(stage3_result_path.read_text())
    stage3_metadata = json.loads(stage3_metadata_path.read_text())
    config = json.loads(config_path.read_text())
    manifest_path = Path(stage3_lock["dataset_manifest"])
    manifest = json.loads(manifest_path.read_text())

    if stage3_result["protocol_lock_sha256"] != file_sha256(stage3_lock_path):
        raise RuntimeError("V8 Stage 3 result does not share the Stage 3 lock")
    if not stage3_result["gates"]["passed"]:
        raise RuntimeError("V8 Stage 3 did not authorize the structured head")
    if stage3_result["decision"] != "advance_to_action_conditioned_head":
        raise RuntimeError("V8 Stage 3 decision is inconsistent with authorization")
    if stage3_metadata["protocol_lock_sha256"] != file_sha256(stage3_lock_path):
        raise RuntimeError("V8 Stage 3 features do not share the Stage 3 lock")
    stage3_artifact_path = Path(stage3_metadata["feature_artifact"])
    if file_sha256(stage3_artifact_path) != stage3_metadata["feature_artifact_sha256"]:
        raise RuntimeError("V8 Stage 3 feature artifact changed")
    if stage3_result["feature_artifact_sha256"] != stage3_metadata["feature_artifact_sha256"]:
        raise RuntimeError("V8 Stage 3 result and features differ")
    if file_sha256(manifest_path) != stage3_lock["dataset_manifest_sha256"]:
        raise RuntimeError("V8 manifest changed after Stage 3")
    if manifest["dataset_sha256"] != stage3_lock["dataset_sha256"]:
        raise RuntimeError("V8 dataset identity changed after Stage 3")
    for relative, expected in stage3_lock["dataset_artifact_sha256"].items():
        if file_sha256(manifest_path.parent / relative) != expected:
            raise RuntimeError(f"V8 artifact changed after Stage 3: {relative}")
    for value, label in (
        (stage3_lock, "Stage 3 lock"),
        (stage3_result, "Stage 3 result"),
        (stage3_metadata, "Stage 3 features"),
        (manifest, "V8 manifest"),
    ):
        no_forbidden_access(value, label)

    if config["model"] != stage3_lock["method"]["model"]:
        raise RuntimeError("V8 structured head must use the locked Stage 3 model")
    if config["layer"] != 6 or config["pooling"] != "mean":
        raise RuntimeError("V8 structured head must use the locked layer-6 mean representation")

    implementation = {
        name: {"path": path, "sha256": file_sha256(Path(path))}
        for name, path in IMPLEMENTATION_PATHS.items()
    }
    lock = {
        "schema_version": 8,
        "experiment": "v8_locked_structured_head_lomo",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "head_config_path": str(config_path),
        "head_config_sha256": file_sha256(config_path),
        "head_config": config,
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "dataset_sha256": stage3_lock["dataset_sha256"],
        "dataset_artifact_sha256": stage3_lock["dataset_artifact_sha256"],
        "mechanics": stage3_lock["mechanics"],
        "surfaces": stage3_lock["surfaces"],
        "stage3_lock": {
            "path": str(stage3_lock_path),
            "sha256": file_sha256(stage3_lock_path),
        },
        "stage3_result": {
            "path": str(stage3_result_path),
            "sha256": file_sha256(stage3_result_path),
            "gates_passed": True,
        },
        "stage3_features": {
            "metadata": str(stage3_metadata_path),
            "metadata_sha256": file_sha256(stage3_metadata_path),
            "artifact": str(stage3_artifact_path),
            "artifact_sha256": stage3_metadata["feature_artifact_sha256"],
        },
        "method": {
            "model": config["model"],
            "frozen_backbone": True,
            "layer": config["layer"],
            "pooling": config["pooling"],
            "max_component_tokens": config["maxComponentTokens"],
            "adapter_path": None,
            "threshold_selection": "other_mechanics_canonical_calibration_only",
        },
        "implementation": implementation,
        "limits": {
            "structured_component_extractions_permitted": 1,
            "structured_head_lomo_runs_permitted": 1,
            "untouched_v8_mechanic_evaluations_permitted": 0,
            "adapter_training_runs_permitted": 0,
        },
        "data_access": {key: 0 for key in FORBIDDEN_ACCESS},
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V8 structured-head lock: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
