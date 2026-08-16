#!/usr/bin/env python3
"""Hash-lock the complete V32 protocol before any model access."""

from __future__ import annotations

import argparse
import hashlib
import json

from audit_v22r2_grounding import read_jsonl_directory
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v32_language.py", "python/generate_v32_factorized_semantics.py",
    "python/audit_v32_factorized_semantics.py", "python/v32_structured_model.py",
    "python/v32_evaluation.py", "python/extract_v32_fit_calibration_features_mlx.py",
    "python/train_v32_heads.py", "python/freeze_v32_factorized_semantics.py",
    "python/freeze_v32_trained_systems.py", "python/evaluate_v32_sealed_mlx.py",
    "python/v32_integration.py", "python/evaluate_v32_v28_integration_mlx.py",
    "python/audit_and_summarize_v32.py", "python/test_v32_factorized_semantics.py",
    "python/v28_marginal_map.py", "python/evaluate_v22r2_relational_grounding.py",
    "python/v22_relational.py", "python/v22r2_grounding.py",
    "python/extract_v10_features_mlx.py", "python/extract_v22r2_relational_features_mlx.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v32-factorized-semantics.json")
    parser.add_argument("--plan", default="docs/v32-factorized-semantics-plan.md")
    parser.add_argument("--audit", default="outputs/v32-factorized-semantics/pre-model-audit.json")
    parser.add_argument("--output", default="configs/v32-factorized-semantics-lock.json")
    args = parser.parse_args()
    config_path, plan_path, audit_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.config, args.plan, args.audit, args.output))
    if output_path.exists():
        raise RuntimeError("V32 protocol lock already exists")
    output_root = PROJECT_ROOT / "outputs/v32-factorized-semantics"
    forbidden = (output_root / "fit-calibration-features", output_root / "training", output_root / "sealed-evaluation", output_root / "integration", PROJECT_ROOT / "configs/v32-trained-systems-lock.json")
    if any(path.exists() for path in forbidden):
        raise RuntimeError("V32 model/training/evaluation artifact exists before protocol lock")
    config, audit = json.loads(config_path.read_text()), json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v32_protocol_lock" or audit["integrity"]["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V32 pre-model audit does not authorize this config")
    corpus_root, manifest_path = PROJECT_ROOT / config["outputDir"], PROJECT_ROOT / config["outputDir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    corpus_files = {path.name: file_sha256(path) for path in sorted(corpus_root.glob("*.jsonl"))}
    corpus_files["manifest.json"] = file_sha256(manifest_path)
    v22_path = PROJECT_ROOT / config["sourceV22r2Lock"]
    v22_lock = json.loads(v22_path.read_text())
    integration_scenes = [row for row in read_jsonl_directory(PROJECT_ROOT / v22_lock["source"]["dataset"] / "scenes") if row["split"] == "grounding_evaluation"]
    clauses = sum(len(row["agent_input"]["evidence"]) for row in integration_scenes)
    counts = manifest["split_counts"]
    lock = {
        "schema_version": 32, "experiment": config["experiment"], "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path), "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)), "preregistration_sha256": file_sha256(plan_path),
        "model": config["model"], "systems": config["systems"], "training": config["training"], "evaluation": config["evaluation"], "gates": config["gates"], "limits": config["limits"],
        "source": {
            "corpus": config["outputDir"], "corpus_manifest": str(manifest_path.relative_to(PROJECT_ROOT)), "corpus_manifest_sha256": file_sha256(manifest_path),
            "corpus_sha256": manifest["corpus_sha256"], "corpus_file_sha256": corpus_files,
            "pre_model_audit": str(audit_path.relative_to(PROJECT_ROOT)), "pre_model_audit_sha256": file_sha256(audit_path),
            **{key: config[key] for key in ("sourceV31ProtocolLock", "sourceV31TrainedLock", "sourceV31Result", "sourceV31PostAudit", "sourceV31ForensicAudit", "sourceV28Lock", "sourceV28Result", "sourceV22r2Lock")},
            **{f"{key}_sha256": file_sha256(PROJECT_ROOT / config[key]) for key in ("sourceV31ProtocolLock", "sourceV31TrainedLock", "sourceV31Result", "sourceV31PostAudit", "sourceV31ForensicAudit", "sourceV28Lock", "sourceV28Result", "sourceV22r2Lock")},
        },
        "planned_training": {"fit_records": counts["factor_fit"], "calibration_records": counts["factor_calibration"], "fit_calibration_records": counts["factor_fit"] + counts["factor_calibration"], "frozen_feature_forward_passes": counts["factor_fit"] + counts["factor_calibration"], "monolithic_training_runs": 3, "joint_auxiliary_training_runs": 3, "evaluation_records_read": 0},
        "planned_evaluation": {"records": counts["factor_evaluation_paraphrase"] + counts["factor_evaluation_composition"], "paraphrase_records": counts["factor_evaluation_paraphrase"], "composition_records": counts["factor_evaluation_composition"], "frozen_feature_forward_passes": counts["factor_evaluation_paraphrase"] + counts["factor_evaluation_composition"]},
        "conditional_integration": {"run_condition": config["integration"]["runCondition"], "scenes": len(integration_scenes), "evidence_clauses": clauses, "seed_combination": config["integration"]["seedCombination"], "maximum_model_forward_passes": clauses, "v28_integration_replays": 1},
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION},
        "data_access_before_lock": {"benchmark_constructions": 1, "model_forward_passes": 0, "feature_extractions": 0, "training_runs": 0, "evaluation_records_read": 0, "evaluation_features_read": 0, "evaluation_predictions": 0, "v28_integration_replays": 0, "hyperparameter_selections": 0, "checkpoint_selections": 0},
    }
    lock["lock_payload_sha256"] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__": main()
