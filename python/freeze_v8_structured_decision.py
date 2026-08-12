#!/usr/bin/env python3
"""Freeze the threshold-free V8 ledger-derived development diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = {
    "evaluator": "python/evaluate_v8_structured_decision_mlx.py",
    "tests": "python/test_v8_structured_decision.py",
    "structured_head": "python/train_v8_action_conditioned_head.py",
    "binary_metrics": "python/binary_metrics.py",
    "pair_metrics": "python/run_v8_lomo_diagnostics.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4-lock", default="configs/v8-structured-head-lock.json")
    parser.add_argument("--stage4-result", default="outputs/v8-structured-head/lomo/result.json")
    parser.add_argument("--components", default="outputs/v8-structured-head/components/metadata.json")
    parser.add_argument("--config", default="configs/v8-structured-decision.json")
    parser.add_argument("--plan", default="docs/v8-structured-decision-plan.md")
    parser.add_argument("--output", default="configs/v8-structured-decision-lock.json")
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


def main() -> None:
    args = parse_args()
    stage4_lock_path = Path(args.stage4_lock)
    stage4_result_path = Path(args.stage4_result)
    component_metadata_path = Path(args.components)
    config_path = Path(args.config)
    plan_path = Path(args.plan)
    output_path = Path(args.output)
    stage4_lock = json.loads(stage4_lock_path.read_text())
    stage4_result = json.loads(stage4_result_path.read_text())
    component_metadata = json.loads(component_metadata_path.read_text())
    config = json.loads(config_path.read_text())

    if stage4_result["structured_head_lock_sha256"] != file_sha256(stage4_lock_path):
        raise RuntimeError("V8 Stage 4 result does not share the Stage 4 lock")
    if stage4_result["gates"]["passed"]:
        raise RuntimeError("V8 Stage 4 already passed; no corrective diagnostic is authorized")
    failed = [check["name"] for check in stage4_result["gates"]["checks"] if not check["passed"]]
    if failed != ["minimum_fold_surface_balanced_accuracy"]:
        raise RuntimeError(f"V8 Stage 4 does not support the narrow calibration diagnosis: {failed}")
    if stage4_result["decision"] != "structured_head_not_ready":
        raise RuntimeError("V8 Stage 4 failure decision is inconsistent")
    if component_metadata["structured_head_lock_sha256"] != file_sha256(stage4_lock_path):
        raise RuntimeError("V8 components do not share the Stage 4 lock")
    component_path = Path(component_metadata["artifact"])
    if file_sha256(component_path) != component_metadata["artifact_sha256"]:
        raise RuntimeError("V8 component artifact changed after Stage 4")
    if config["threshold"] != 0.0:
        raise RuntimeError("V8 ledger-derived decision must use the fixed multiclass boundary")
    if config["gates"] != stage4_lock["head_config"]["gates"]:
        raise RuntimeError("V8 corrective diagnostic may not weaken Stage 4 gates")

    parameters = {}
    for mechanic in stage4_lock["mechanics"]:
        saved = stage4_result["folds"][mechanic]
        path = Path(saved["parameter_artifact"])
        if file_sha256(path) != saved["parameter_artifact_sha256"]:
            raise RuntimeError(f"V8 Stage 4 head changed for {mechanic}")
        parameters[mechanic] = {"path": str(path), "sha256": file_sha256(path)}

    implementation = {
        name: {"path": path, "sha256": file_sha256(Path(path))}
        for name, path in IMPLEMENTATION_PATHS.items()
    }
    lock = {
        "schema_version": 8,
        "experiment": "v8_locked_ledger_derived_decision",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "decision_config_path": str(config_path),
        "decision_config_sha256": file_sha256(config_path),
        "decision_config": config,
        "dataset_sha256": stage4_lock["dataset_sha256"],
        "mechanics": stage4_lock["mechanics"],
        "surfaces": stage4_lock["surfaces"],
        "projection_width": stage4_lock["head_config"]["projectionWidth"],
        "stage4_lock": {"path": str(stage4_lock_path), "sha256": file_sha256(stage4_lock_path)},
        "stage4_result": {"path": str(stage4_result_path), "sha256": file_sha256(stage4_result_path)},
        "stage3_features": stage4_lock["stage3_features"],
        "components": {
            "metadata": str(component_metadata_path),
            "metadata_sha256": file_sha256(component_metadata_path),
            "artifact": str(component_path),
            "artifact_sha256": component_metadata["artifact_sha256"],
        },
        "parameters": parameters,
        "implementation": implementation,
        "limits": {
            "ledger_derived_development_evaluations_permitted": 1,
            "training_runs_permitted": 0,
            "adapter_training_runs_permitted": 0,
            "untouched_v8_mechanic_evaluations_permitted": 0,
        },
        "data_access": {
            "v3_test_records_read": 0,
            "prior_holdout_records_read": 0,
            "v7_tone_drift_records_read": 0,
            "v7_model_results_read": 0,
            "untouched_v8_mechanic_records_read": 0,
        },
        "lock_payload_sha256": "",
    }
    lock["lock_payload_sha256"] = canonical_sha256({**lock, "lock_payload_sha256": ""})
    content = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if output_path.exists() and output_path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite changed V8 structured-decision lock: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
