"""Freeze the audited V22r2 corpus, representation, heads, and one-shot evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
    "python/build_v22r2_relational_grounding.py",
    "python/audit_v22r2_grounding.py",
    "python/freeze_v22r2_relational_grounding.py",
    "python/extract_v22r2_relational_features_mlx.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/extract_v10_features_mlx.py",
    "python/extract_v11_scale_features_mlx.py",
    "python/run_v22_oracle_baselines.py",
    "python/test_v22r2_grounding.py",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.v22r2.json")
    parser.add_argument("--plan", default="docs/v22r2-relational-grounding-plan.md")
    parser.add_argument("--audit", default="outputs/v22r2-relational-grounding/pre-extraction-audit.json")
    parser.add_argument("--output", default="configs/v22r2-relational-grounding-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    model_root = PROJECT_ROOT / "outputs/v22r2-relational-grounding"
    forbidden = (
        model_root / "feature-extraction-attempt.json",
        model_root / "evaluation-attempt.json",
        model_root / "features",
        model_root / "evaluation",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("V22r2 model access artifacts exist; refusing a post-access lock")
    if output_path.exists():
        raise RuntimeError("V22r2 protocol lock already exists")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v22r2_protocol_lock":
        raise RuntimeError("V22r2 pre-extraction audit does not authorize locking")
    if audit["firewall"]["new_model_forward_passes"] != 0:
        raise RuntimeError("V22r2 audit reports model access before lock")
    dataset = PROJECT_ROOT / config["outputDir"]
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V22r2 corpus was generated under a different configuration")
    if manifest["scenes"] != audit["population"]["scenes"]:
        raise RuntimeError("V22r2 manifest and audit scene counts differ")
    if (
        audit["surface_and_prompts"]["new_model_forward_passes"]
        > config["gates"]["preExtraction"]["maximumNewModelForwardPasses"]
    ):
        raise RuntimeError("V22r2 prompt inventory exceeds the registered budget")
    source = {
        "dataset": config["outputDir"],
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "record_corpus_sha256": manifest["record_corpus_sha256"],
        "scene_corpus_sha256": manifest["scene_corpus_sha256"],
        "pre_extraction_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "pre_extraction_audit_sha256": file_sha256(audit_path),
        "v22_config": config["sourceV22Config"],
        "v22_config_sha256": file_sha256(PROJECT_ROOT / config["sourceV22Config"]),
        "v22_manifest": config["sourceV22Manifest"],
        "v22_manifest_sha256": file_sha256(PROJECT_ROOT / config["sourceV22Manifest"]),
        "v22_audit": config["sourceV22Audit"],
        "v22_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV22Audit"]),
        "v22_result": config["sourceV22Result"],
        "v22_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV22Result"]),
    }
    lock = {
        "schema_version": "22r2",
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "model": config["model"],
        "heads": config["heads"],
        "integration_conditions": config["integrationConditions"],
        "empty_version_space_policy": config["emptyVersionSpacePolicy"],
        "excess_unknown_policy": config["excessUnknownPolicy"],
        "gates": config["gates"],
        "limits": config["limits"],
        "source": source,
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_lock": {
            "v22_records_read": 24,
            "v22r2_scenes_created": manifest["scenes"],
            "new_model_forward_passes": 0,
            "model_predictions_read": 0,
            "linear_fits": 0,
            "adapter_training_runs": 0,
            "v21_final_records_read": 0,
            "v21_final_model_results_read": 0,
        },
    }
    lock["lock_payload_sha256"] = canonical_sha256(lock)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
