#!/usr/bin/env python3
"""Freeze the final development-only V8 query-conditioned relational head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_PATHS = {
    "trainer": "python/train_v8_relational_head.py",
    "tests": "python/test_v8_relational_head.py",
    "structured_components": "python/extract_v8_structured_components_mlx.py",
    "shared_structured_head": "python/train_v8_action_conditioned_head.py",
    "binary_metrics": "python/binary_metrics.py",
    "pair_metrics": "python/run_v8_lomo_diagnostics.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4-lock", default="configs/v8-structured-head-lock.json")
    parser.add_argument("--stage4-result", default="outputs/v8-structured-head/lomo/result.json")
    parser.add_argument("--decision-lock", default="configs/v8-structured-decision-lock.json")
    parser.add_argument("--decision-result", default="outputs/v8-structured-decision/result.json")
    parser.add_argument("--config", default="configs/v8-relational-head.json")
    parser.add_argument("--plan", default="docs/v8-relational-head-plan.md")
    parser.add_argument("--output", default="configs/v8-relational-head-lock.json")
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


def require_only_absolute_failure(result: dict[str, Any], label: str) -> None:
    if result["gates"]["passed"]:
        raise RuntimeError(f"{label} already passed")
    failed = [check["name"] for check in result["gates"]["checks"] if not check["passed"]]
    if failed != ["minimum_fold_surface_balanced_accuracy"]:
        raise RuntimeError(f"{label} does not support the relational-head diagnosis: {failed}")


def main() -> None:
    args = parse_args()
    stage4_lock_path = Path(args.stage4_lock)
    stage4_result_path = Path(args.stage4_result)
    decision_lock_path = Path(args.decision_lock)
    decision_result_path = Path(args.decision_result)
    config_path = Path(args.config)
    plan_path = Path(args.plan)
    output_path = Path(args.output)
    stage4_lock = json.loads(stage4_lock_path.read_text())
    stage4_result = json.loads(stage4_result_path.read_text())
    decision_lock = json.loads(decision_lock_path.read_text())
    decision_result = json.loads(decision_result_path.read_text())
    config = json.loads(config_path.read_text())

    if stage4_result["structured_head_lock_sha256"] != file_sha256(stage4_lock_path):
        raise RuntimeError("V8 Stage 4 result does not share its lock")
    if decision_result["decision_lock_sha256"] != file_sha256(decision_lock_path):
        raise RuntimeError("V8 structured-decision result does not share its lock")
    require_only_absolute_failure(stage4_result, "V8 Stage 4")
    require_only_absolute_failure(decision_result, "V8 structured decision")
    if config["gates"] != stage4_lock["head_config"]["gates"]:
        raise RuntimeError("V8 relational head may not weaken the original gates")
    if config["model"] != stage4_lock["method"]["model"]:
        raise RuntimeError("V8 relational head must reuse the frozen model")
    if config["layer"] != 6 or config["pooling"] != "mean":
        raise RuntimeError("V8 relational head must reuse the locked representation")

    component_metadata_path = Path(decision_lock["components"]["metadata"])
    component_metadata = json.loads(component_metadata_path.read_text())
    component_path = Path(component_metadata["artifact"])
    if file_sha256(component_metadata_path) != decision_lock["components"]["metadata_sha256"]:
        raise RuntimeError("V8 component metadata changed")
    if file_sha256(component_path) != decision_lock["components"]["artifact_sha256"]:
        raise RuntimeError("V8 component artifact changed")
    feature_metadata_path = Path(stage4_lock["stage3_features"]["metadata"])
    feature_metadata = json.loads(feature_metadata_path.read_text())
    feature_path = Path(feature_metadata["feature_artifact"])
    if file_sha256(feature_path) != stage4_lock["stage3_features"]["artifact_sha256"]:
        raise RuntimeError("V8 Stage 3 feature artifact changed")

    implementation = {
        name: {"path": path, "sha256": file_sha256(Path(path))}
        for name, path in IMPLEMENTATION_PATHS.items()
    }
    lock = {
        "schema_version": 8,
        "experiment": "v8_locked_query_conditioned_relational_head",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "head_config_path": str(config_path),
        "head_config_sha256": file_sha256(config_path),
        "head_config": config,
        "dataset_sha256": stage4_lock["dataset_sha256"],
        "mechanics": stage4_lock["mechanics"],
        "surfaces": stage4_lock["surfaces"],
        "stage4": {
            "lock": str(stage4_lock_path),
            "lock_sha256": file_sha256(stage4_lock_path),
            "result": str(stage4_result_path),
            "result_sha256": file_sha256(stage4_result_path),
        },
        "structured_decision": {
            "lock": str(decision_lock_path),
            "lock_sha256": file_sha256(decision_lock_path),
            "result": str(decision_result_path),
            "result_sha256": file_sha256(decision_result_path),
        },
        "stage3_features": {
            "metadata": str(feature_metadata_path),
            "metadata_sha256": file_sha256(feature_metadata_path),
            "artifact": str(feature_path),
            "artifact_sha256": stage4_lock["stage3_features"]["artifact_sha256"],
        },
        "components": {
            "metadata": str(component_metadata_path),
            "metadata_sha256": file_sha256(component_metadata_path),
            "artifact": str(component_path),
            "artifact_sha256": decision_lock["components"]["artifact_sha256"],
        },
        "implementation": implementation,
        "limits": {
            "relational_head_lomo_runs_permitted": 1,
            "additional_component_extractions_permitted": 0,
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
        raise RuntimeError(f"Refusing to overwrite changed V8 relational-head lock: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
