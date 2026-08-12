"""Freeze the V19 corpus, prompts, model, heads, implementation, and one-shot limits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = (
    "python/v18_schema.py",
    "python/build_v19_grounding_views.py",
    "python/audit_v19_compatibility.py",
    "python/extract_v19_grounding_features_mlx.py",
    "python/evaluate_v19_frozen_integration.py",
    "python/run_v18_schema_baselines.py",
    "python/test_v18_schema.py",
    "python/test_v18_protocol.py",
    "python/test_v19_protocol.py",
    "python/extract_v10_features_mlx.py",
    "python/extract_v11_scale_features_mlx.py",
    "python/extract_v13_token_local_mlx.py",
    "python/evaluate_v10_frozen.py",
    "python/evaluate_v15_full_pipeline.py",
    "python/v10_protocol.py",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v19-frozen-integration.json")
    parser.add_argument("--plan", default="docs/v19-frozen-integration-plan.md")
    parser.add_argument("--audit", default="outputs/v19-frozen-integration/pre-extraction-audit.json")
    parser.add_argument("--output", default="configs/v19-frozen-integration-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    feature_dir = PROJECT_ROOT / "outputs/v19-frozen-integration/features"
    evaluation_dir = PROJECT_ROOT / "outputs/v19-frozen-integration/evaluation"
    if feature_dir.exists() or evaluation_dir.exists():
        raise RuntimeError("V19 model artifacts already exist; refusing to create a post-access lock")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_single_v19_feature_extraction":
        raise RuntimeError("V19 pre-extraction audit does not authorize locking")
    if audit["prompt_inventory"]["new_model_forward_passes"] > config["gates"]["preExtraction"]["maximumNewModelForwardPasses"]:
        raise RuntimeError("V19 prompt count exceeds config before lock")
    v19_manifest_path = PROJECT_ROOT / config["outputDir"] / "manifest.json"
    v19_manifest = json.loads(v19_manifest_path.read_text())
    v18_manifest_path = PROJECT_ROOT / config["sourceV18Manifest"]
    v18_result_path = PROJECT_ROOT / config["sourceV18Result"]
    v18_result = json.loads(v18_result_path.read_text())
    if not v18_result["passed"] or v18_result["decision"] != "authorize_frozen_grounding_integration":
        raise RuntimeError("Strengthened V18 result does not authorize V19")
    if v19_manifest["source_v18_dataset_sha256"] != json.loads(v18_manifest_path.read_text())["dataset_sha256"]:
        raise RuntimeError("V19 latent source differs from V18")
    head_path = PROJECT_ROOT / config["sourceDeploymentHeads"]
    if file_sha256(head_path) != audit["head_provenance"]["head_artifact_sha256"]:
        raise RuntimeError("Deployment heads differ from the provenance audit")
    source = {
        "v18_dataset": str(v18_manifest_path.parent.relative_to(PROJECT_ROOT)),
        "v18_manifest": config["sourceV18Manifest"],
        "v18_manifest_sha256": file_sha256(v18_manifest_path),
        "v18_dataset_sha256": v19_manifest["source_v18_dataset_sha256"],
        "v18_result": config["sourceV18Result"],
        "v18_result_sha256": file_sha256(v18_result_path),
        "v19_dataset": config["outputDir"],
        "v19_manifest": str(v19_manifest_path.relative_to(PROJECT_ROOT)),
        "v19_manifest_sha256": file_sha256(v19_manifest_path),
        "v19_dataset_sha256": v19_manifest["dataset_sha256"],
        "pre_extraction_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "pre_extraction_audit_sha256": file_sha256(audit_path),
        "v15_lock": config["sourceV15Lock"],
        "v15_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV15Lock"]),
        "v15_features": config["sourceV15Features"],
        "v15_features_sha256": file_sha256(PROJECT_ROOT / config["sourceV15Features"]),
        "deployment_heads": config["sourceDeploymentHeads"],
        "deployment_heads_sha256": file_sha256(head_path),
    }
    lock = {
        "schema_version": 19,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "model": config["model"],
        "views": {key: value["role"] for key, value in config["views"].items()},
        "conditions": config["conditions"],
        "primary_split": config["primarySplit"],
        "empty_version_policy": config["emptyVersionPolicy"],
        "gates": config["gates"],
        "limits": config["limits"],
        "source": source,
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_lock": {
            "v18_records_read": 72,
            "v19_scenes_created": v19_manifest["scenes"],
            "v19_model_forward_passes": 0,
            "v19_model_predictions_read": 0,
            "v17_head_artifacts_read": 1,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "adapter_training_runs": 0,
            "new_selection_linear_fits": 0,
            "head_provenance_verification_refits": 3,
        },
    }
    lock["lock_payload_sha256"] = canonical_sha256(lock)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
