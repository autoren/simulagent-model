"""Freeze systems and all V21 execution code before drawing the final seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_IMPLEMENTATION = (
    "python/audit_v21_final_suite.py",
    "python/seal_v21_final_dataset.py",
    "python/extract_v21_final_features_mlx.py",
    "python/evaluate_v21_final.py",
    "python/audit_v21_result.py",
    "python/freeze_v21_execution.py",
    "python/evaluate_v20_probabilistic_interface.py",
    "python/v20_probabilistic_grounding.py",
    "python/evaluate_v19_frozen_integration.py",
    "python/extract_v19_grounding_features_mlx.py",
    "python/audit_v19_compatibility.py",
    "python/evaluate_v15_full_pipeline.py",
    "python/evaluate_v10_frozen.py",
    "python/run_v18_schema_baselines.py",
    "python/v10_protocol.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-lock", default="configs/v21r2-multimechanic-design-lock.json")
    parser.add_argument("--v20-lock", default="configs/v20-probabilistic-interface-lock.json")
    parser.add_argument("--v20-result", default="outputs/v20-probabilistic-interface/evaluation/result.json")
    parser.add_argument("--output", default="configs/v21-multimechanic-execution-lock.json")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise RuntimeError(f"V21 execution lock already exists: {output}")
    if (PROJECT_ROOT / "data/v21-final").exists() or (PROJECT_ROOT / "outputs/v21-final/seed-draw.json").exists():
        raise RuntimeError("V21 final seed or records exist before execution lock")
    design_path = PROJECT_ROOT / args.design_lock
    design = json.loads(design_path.read_text())
    for path, expected in design["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V21 design implementation changed: {path}")
    v20_lock_path = PROJECT_ROOT / args.v20_lock
    v20_lock = json.loads(v20_lock_path.read_text())
    for path, expected in v20_lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V20 locked implementation changed: {path}")
    v20_result_path = PROJECT_ROOT / args.v20_result
    v20 = json.loads(v20_result_path.read_text())
    if (
        v20["evaluation_number"] != 1
        or v20["data_access"]["adapter_training_runs"] != 0
        or v20["data_access"]["new_feature_extractions"] != 0
        or v20["data_access"]["new_linear_fits"] != 0
    ):
        raise RuntimeError("V20 result is not the single frozen zero-training evaluation")
    if not v20["supported_preservation_passed"]:
        challenger_eligible = False
    else:
        challenger_eligible = True
    v19_lock_path = PROJECT_ROOT / "configs/v19-frozen-integration-lock.json"
    v19_lock = json.loads(v19_lock_path.read_text())
    for path, expected in v19_lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V19 locked implementation changed: {path}")
    head_path = PROJECT_ROOT / v19_lock["source"]["deployment_heads"]
    if file_sha256(head_path) != v19_lock["source"]["deployment_heads_sha256"]:
        raise RuntimeError("V19 deployment heads changed")
    v18_manifest_path = PROJECT_ROOT / v19_lock["source"]["v18_manifest"]
    if file_sha256(v18_manifest_path) != v19_lock["source"]["v18_manifest_sha256"]:
        raise RuntimeError("V18 manifest changed")
    v18_manifest = json.loads(v18_manifest_path.read_text())
    if v18_manifest["dataset_sha256"] != v19_lock["source"]["v18_dataset_sha256"]:
        raise RuntimeError("V18 dataset identity changed")
    for relative, expected in v18_manifest["artifact_sha256"].items():
        if file_sha256(v18_manifest_path.parent / relative) != expected:
            raise RuntimeError(f"V18 dataset artifact changed: {relative}")
    implementation = dict(design["implementation"])
    implementation.update({
        path: file_sha256(PROJECT_ROOT / path) for path in NEW_IMPLEMENTATION
    })
    config = design["config"]
    lock = {
        "schema_version": 21,
        "experiment": "v21_final_execution_lock_before_delayed_seed",
        "design_lock": args.design_lock,
        "design_lock_sha256": file_sha256(design_path),
        "base_design_lock": design["base_design_lock"],
        "base_design_lock_sha256": design["base_design_lock_sha256"],
        "config": config,
        "implementation": implementation,
        "model": v19_lock["model"],
        "probabilistic_interface": v20_lock["config"]["interface"],
        "challenger_eligible": challenger_eligible,
        "challenger_status": v20["decision"],
        "source": {
            "v18_dataset": v19_lock["source"]["v18_dataset"],
            "v18_dataset_sha256": v19_lock["source"]["v18_dataset_sha256"],
            "v19_lock": "configs/v19-frozen-integration-lock.json",
            "v19_lock_sha256": file_sha256(v19_lock_path),
            "v20_lock": args.v20_lock,
            "v20_lock_sha256": file_sha256(v20_lock_path),
            "v20_result": args.v20_result,
            "v20_result_sha256": file_sha256(v20_result_path),
            "deployment_heads": v19_lock["source"]["deployment_heads"],
            "deployment_heads_sha256": v19_lock["source"]["deployment_heads_sha256"],
        },
        "limits": config["limits"],
        "data_access_before_lock": {
            "v20_results_read": 1,
            "final_seed_draws": 0,
            "final_records_created_or_read": 0,
            "final_model_forward_passes": 0,
            "final_evaluations": 0,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
        },
    }
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": args.output,
        "lock_sha256": file_sha256(output),
        "implementation_files": len(implementation),
        "challenger_eligible": challenger_eligible,
        "challenger_status": v20["decision"],
        "final_seed_draws": 0,
        "final_records": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
