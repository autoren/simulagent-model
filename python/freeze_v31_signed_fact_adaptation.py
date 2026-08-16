#!/usr/bin/env python3
"""Hash-lock the complete V31 matched-head protocol before model access."""

from __future__ import annotations

import argparse
import hashlib
import json

from audit_v22r2_grounding import read_jsonl_directory
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v31_language.py",
    "python/generate_v31_signed_fact_adaptation.py",
    "python/audit_v31_signed_fact_adaptation.py",
    "python/v31_structured_model.py",
    "python/v31_evaluation.py",
    "python/extract_v31_fit_calibration_features_mlx.py",
    "python/train_v31_frozen_readout.py",
    "python/train_v31_lora_readout_mlx.py",
    "python/freeze_v31_signed_fact_adaptation.py",
    "python/freeze_v31_trained_systems.py",
    "python/evaluate_v31_sealed_mlx.py",
    "python/v31_integration.py",
    "python/evaluate_v31_v28_integration_mlx.py",
    "python/audit_and_summarize_v31.py",
    "python/test_v31_signed_fact_adaptation.py",
    "python/v30_language.py",
    "python/evaluate_v30_signed_fact_language_mlx.py",
    "python/v28_marginal_map.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
    "python/extract_v10_features_mlx.py",
    "python/extract_v22r2_relational_features_mlx.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v31-signed-fact-adaptation.json")
    parser.add_argument("--plan", default="docs/v31-signed-fact-adaptation-plan.md")
    parser.add_argument("--audit", default="outputs/v31-signed-fact-adaptation/pre-model-audit.json")
    parser.add_argument("--output", default="configs/v31-signed-fact-adaptation-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V31 protocol lock already exists")
    output_root = PROJECT_ROOT / "outputs/v31-signed-fact-adaptation"
    forbidden = (
        output_root / "fit-calibration-features",
        output_root / "frozen-readout", output_root / "lora-readout",
        output_root / "sealed-evaluation", output_root / "integration",
        PROJECT_ROOT / "configs/v31-trained-systems-lock.json",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("V31 model/training/evaluation artifact exists before protocol lock")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v31_protocol_lock":
        raise RuntimeError("V31 pre-model audit does not authorize lock")
    if audit["integrity"]["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V31 config changed after audit")
    corpus_root = PROJECT_ROOT / config["outputDir"]
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    corpus_files = {
        path.name: file_sha256(path) for path in sorted(corpus_root.glob("*.jsonl"))
    }
    corpus_files["manifest.json"] = file_sha256(manifest_path)
    split_counts = manifest["split_counts"]
    v22_lock_path = PROJECT_ROOT / config["sourceV22r2Lock"]
    v22_lock = json.loads(v22_lock_path.read_text())
    integration_scenes = [
        row for row in read_jsonl_directory(PROJECT_ROOT / v22_lock["source"]["dataset"] / "scenes")
        if row["split"] == "grounding_evaluation"
    ]
    integration_clauses = sum(len(row["agent_input"]["evidence"]) for row in integration_scenes)
    lock = {
        "schema_version": 31, "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path), "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "model": config["model"], "systems": config["systems"],
        "training": config["training"], "evaluation": config["evaluation"],
        "gates": config["gates"], "limits": config["limits"],
        "source": {
            "corpus": config["outputDir"],
            "corpus_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "corpus_manifest_sha256": file_sha256(manifest_path),
            "corpus_sha256": manifest["corpus_sha256"],
            "corpus_file_sha256": corpus_files,
            "pre_model_audit": str(audit_path.relative_to(PROJECT_ROOT)),
            "pre_model_audit_sha256": file_sha256(audit_path),
            "v30_lock": config["sourceV30Lock"],
            "v30_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV30Lock"]),
            "v30_result": config["sourceV30Result"],
            "v30_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV30Result"]),
            "v30_post_audit": config["sourceV30PostAudit"],
            "v30_post_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV30PostAudit"]),
            "v28_lock": config["sourceV28Lock"],
            "v28_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV28Lock"]),
            "v28_result": config["sourceV28Result"],
            "v28_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV28Result"]),
            "v22r2_lock": config["sourceV22r2Lock"],
            "v22r2_lock_sha256": file_sha256(v22_lock_path),
        },
        "planned_training": {
            "fit_records": split_counts["adaptation_fit"],
            "calibration_records": split_counts["adaptation_calibration"],
            "fit_calibration_records": split_counts["adaptation_fit"] + split_counts["adaptation_calibration"],
            "frozen_feature_forward_passes": split_counts["adaptation_fit"] + split_counts["adaptation_calibration"],
            "frozen_readout_training_runs": len(config["training"]["seeds"]),
            "lora_training_runs": len(config["training"]["seeds"]),
            "evaluation_records_read": 0,
        },
        "planned_evaluation": {
            "records": split_counts["adaptation_evaluation"],
            "surface_families": config["gates"]["preModel"]["requiredEvaluationSurfaceFamilies"],
            "zero_shot_forward_passes": split_counts["adaptation_evaluation"] * 4,
            "frozen_feature_forward_passes": split_counts["adaptation_evaluation"],
            "lora_forward_passes": split_counts["adaptation_evaluation"] * len(config["training"]["seeds"]),
        },
        "conditional_integration": {
            "run_condition": config["integration"]["runCondition"],
            "scenes": len(integration_scenes), "evidence_clauses": integration_clauses,
            "seed_combination": config["integration"]["seedCombination"],
            "maximum_model_forward_passes": integration_clauses * len(config["training"]["seeds"]),
            "v28_integration_replays": 1,
        },
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION},
        "data_access_before_lock": {
            "benchmark_constructions": 1, "model_forward_passes": 0,
            "feature_extractions": 0, "frozen_training_runs": 0,
            "lora_training_runs": 0, "evaluation_records_read": 0,
            "evaluation_features_read": 0, "evaluation_predictions": 0,
            "v28_integration_replays": 0, "hyperparameter_selections": 0,
            "checkpoint_selections": 0,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
