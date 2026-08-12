#!/usr/bin/env python3
"""Freeze V17 before any final-mechanic record is constructed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


IMPLEMENTATION_PATHS = [
    "src/compile-v17.ts",
    "src/v17-final-mechanic.ts",
    "src/v14-grounding.ts",
    "src/v9-symbolic.ts",
    "src/serialization.ts",
    "src/io.ts",
    "python/seal_v17_final_dataset.py",
    "python/extract_v17_final_features_mlx.py",
    "python/evaluate_v17_final_mechanic.py",
    "python/summarize_v17.py",
    "python/test_v17_final_mechanic.py",
    "python/v17_protocol.py",
    "python/evaluate_v15_full_pipeline.py",
    "python/extract_v15_full_features_mlx.py",
    "python/evaluate_v10_frozen.py",
    "python/extract_v10_features_mlx.py",
    "python/extract_v11_scale_features_mlx.py",
    "python/extract_v13_token_local_mlx.py",
    "python/v14_protocol.py",
    "python/v9_symbolic.py",
    "python/binary_metrics.py",
]


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    config_path = Path("configs/v17-final-mechanic.json")
    plan_path = Path("docs/v17r2-visible-outcome-amendment.md")
    output_path = Path("configs/v17-final-construction-lock.json")
    config = json.loads(config_path.read_text())
    final_dir = Path(config["outputDir"])
    if final_dir.exists():
        raise RuntimeError("V17 final data already exists; the construction lock must precede it")

    v16_lock_path = Path(config["sourceV16Lock"])
    v16_result_path = Path(config["sourceV16Result"])
    v16_lock = json.loads(v16_lock_path.read_text())
    v16_result = json.loads(v16_result_path.read_text())
    if v16_result["protocol_lock_sha256"] != file_sha256(v16_lock_path):
        raise RuntimeError("V16 lock/result mismatch")
    if v16_result["decision"] != "authorize_separately_locked_final_mechanic_evaluation":
        raise RuntimeError("V16 did not authorize a final-mechanic protocol")
    if v16_result["final_mechanic_accessed"] or not v16_result["scope_correct_gates"]["passed"]:
        raise RuntimeError("V16 firewall or gates do not authorize V17")

    v15_lock_path = Path(config["sourceV15Lock"])
    v15_metadata_path = Path(config["sourceV15Features"])
    v15_lock = json.loads(v15_lock_path.read_text())
    v15_metadata = json.loads(v15_metadata_path.read_text())
    if v15_metadata["protocol_lock_sha256"] != file_sha256(v15_lock_path):
        raise RuntimeError("V15 lock/features mismatch")
    expected_model = {
        "model_key": config["modelKey"], "model": config["model"], "revision": config["revision"],
        "config_sha256": config["configSha256"], "total_layers": config["totalLayers"],
        "hidden_size": config["hiddenSize"], "extraction_layer": config["extractionLayer"],
    }
    if v15_lock["model"] != expected_model or v15_lock["c_value"] != config["cValue"]:
        raise RuntimeError("V17 changes the frozen V15 model or readout")
    if v15_lock["gates"] != config["gates"]:
        raise RuntimeError("V17 changes a V15 gate")

    v8_manifest_path = Path(config["sourceV8Manifest"])
    v8_manifest = json.loads(v8_manifest_path.read_text())
    v8_artifacts = {}
    for path_text in config["sourceV8Records"]:
        path = Path(path_text)
        relative = str(path.relative_to(v8_manifest_path.parent))
        digest = file_sha256(path)
        if v8_manifest["artifact_sha256"][relative] != digest:
            raise RuntimeError(f"V17 V8 scaffold artifact changed: {path}")
        v8_artifacts[path_text] = digest
    v14_manifest_path = Path(config["sourceV14Manifest"])
    if file_sha256(v14_manifest_path) != v15_lock["source"]["manifest_sha256"]:
        raise RuntimeError("V17 V14 source differs from V15")

    abort_lock_path = Path("configs/v17-aborted-construction-lock.json")
    abort_config_path = Path("configs/v17-aborted-final-mechanic.json")
    abort_report_path = Path("docs/v17-construction-abort.md")
    if not abort_lock_path.exists() or not abort_config_path.exists() or not abort_report_path.exists():
        raise RuntimeError("V17r2 requires the preserved pre-data V17 abort audit")

    lock = {
        "schema_version": 17,
        "experiment": "v17r2_locked_final_mechanic_construction_and_one_shot_evaluation",
        "preregistration": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "config_path": str(config_path), "config_sha256": file_sha256(config_path),
        "model": expected_model, "c_value": config["cValue"], "seed": config["seed"],
        "max_sequence_length": config["maxSequenceLength"],
        "mechanic": config["mechanic"], "operator_family": config["operatorFamily"],
        "lexical_reference_mechanic": config["lexicalReferenceMechanic"],
        "expected": config["expected"], "gates": config["gates"],
        "source": {
            "aborted_v17_lock": str(abort_lock_path), "aborted_v17_lock_sha256": file_sha256(abort_lock_path),
            "aborted_v17_config": str(abort_config_path), "aborted_v17_config_sha256": file_sha256(abort_config_path),
            "aborted_v17_report": str(abort_report_path), "aborted_v17_report_sha256": file_sha256(abort_report_path),
            "v8_manifest": str(v8_manifest_path), "v8_manifest_sha256": file_sha256(v8_manifest_path),
            "v8_dataset_sha256": v8_manifest["dataset_sha256"], "v8_artifacts": v8_artifacts,
            "v14_manifest": str(v14_manifest_path), "v14_manifest_sha256": file_sha256(v14_manifest_path),
            "v15_lock": str(v15_lock_path), "v15_lock_sha256": file_sha256(v15_lock_path),
            "v15_features": str(v15_metadata_path), "v15_features_sha256": file_sha256(v15_metadata_path),
            "v15_feature_artifact": v15_metadata["feature_artifact"],
            "v15_feature_artifact_sha256": v15_metadata["feature_artifact_sha256"],
            "v16_lock": str(v16_lock_path), "v16_lock_sha256": file_sha256(v16_lock_path),
            "v16_result": str(v16_result_path), "v16_result_sha256": file_sha256(v16_result_path),
        },
        "implementation": {path: file_sha256(Path(path)) for path in IMPLEMENTATION_PATHS},
        "limits": {
            "final_corpus_constructions_permitted": 1,
            "final_dataset_seals_permitted": 1,
            "final_feature_extractions_permitted": 1,
            "final_mechanic_evaluations_permitted": 1,
            "development_linear_fits_permitted": 3,
            "adapter_training_runs_permitted": 0,
            "alternate_models_permitted": 0,
            "alternate_layers_permitted": 0,
            "alternate_representations_permitted": 0,
            "threshold_changes_permitted": 0,
            "hyperparameter_searches_permitted": 0,
            "final_retries_permitted": 0,
        },
        "data_access_before_lock": {
            "final_v17_mechanic_records_created": 0,
            "final_v17_mechanic_records_read": 0,
            "final_v17_model_scores_read": 0,
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
        raise RuntimeError(f"Refusing to overwrite changed V17 construction lock: {output_path}")
    output_path.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
