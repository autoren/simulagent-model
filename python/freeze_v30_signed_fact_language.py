#!/usr/bin/env python3
"""Hash-lock the V30 benchmark, methods, gates, and conditional integration."""

from __future__ import annotations

import argparse
import hashlib
import json

from audit_v22r2_grounding import read_jsonl_directory
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v30_language.py",
    "python/generate_v30_signed_fact_language.py",
    "python/audit_v30_signed_fact_language.py",
    "python/v30_evaluation.py",
    "python/evaluate_v30_signed_fact_language_mlx.py",
    "python/v30_integration.py",
    "python/evaluate_v30_v28_integration_mlx.py",
    "python/audit_and_summarize_v30_language.py",
    "python/audit_and_summarize_v30_integration.py",
    "python/freeze_v30_signed_fact_language.py",
    "python/test_v30_signed_fact_language.py",
    "python/v28_marginal_map.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
    "python/extract_v10_features_mlx.py"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v30-signed-fact-language.json")
    parser.add_argument("--plan", default="docs/v30-signed-fact-language-plan.md")
    parser.add_argument("--audit", default="outputs/v30-signed-fact-language/pre-model-audit.json")
    parser.add_argument("--output", default="configs/v30-signed-fact-language-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V30 protocol lock already exists")
    output_root = PROJECT_ROOT / "outputs/v30-signed-fact-language"
    for path in (
        output_root / "evaluation", output_root / "evaluation-attempt.json",
        output_root / "integration", output_root / "integration-attempt.json",
    ):
        if path.exists():
            raise RuntimeError(f"V30 result artifact exists before lock: {path}")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v30_protocol_lock":
        raise RuntimeError("V30 pre-model audit does not authorize lock")
    if audit["integrity"]["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V30 configuration changed after audit")
    corpus_root = PROJECT_ROOT / config["outputDir"]
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    corpus_files = {
        path.name: file_sha256(path) for path in sorted(corpus_root.glob("*.jsonl"))
    }
    corpus_files["manifest.json"] = file_sha256(manifest_path)
    v28_audit = json.loads((PROJECT_ROOT / config["sourceV28PostAudit"]).read_text())
    if not v28_audit["passed"]:
        raise RuntimeError("V30 cannot lock without an accepted V28 source")
    v22r2_lock_path = PROJECT_ROOT / config["sourceV22r2Lock"]
    v22r2_lock = json.loads(v22r2_lock_path.read_text())
    evaluation_scenes = [
        row for row in read_jsonl_directory(PROJECT_ROOT / v22r2_lock["source"]["dataset"] / "scenes")
        if row["split"] == "grounding_evaluation"
    ]
    evidence_clauses = sum(len(row["agent_input"]["evidence"]) for row in evaluation_scenes)
    rows = manifest["records"]
    lock = {
        "schema_version": 30, "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path), "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "model": config["model"], "methods": config["methods"],
        "gates": config["gates"], "limits": config["limits"],
        "source": {
            "corpus": config["outputDir"],
            "corpus_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "corpus_manifest_sha256": file_sha256(manifest_path),
            "corpus_sha256": manifest["corpus_sha256"],
            "corpus_file_sha256": corpus_files,
            "pre_model_audit": str(audit_path.relative_to(PROJECT_ROOT)),
            "pre_model_audit_sha256": file_sha256(audit_path),
            "v28_lock": config["sourceV28Lock"],
            "v28_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV28Lock"]),
            "v28_result": config["sourceV28Result"],
            "v28_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV28Result"]),
            "v28_post_audit": config["sourceV28PostAudit"],
            "v28_post_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV28PostAudit"]),
            "v22r2_lock": config["sourceV22r2Lock"],
            "v22r2_lock_sha256": file_sha256(v22r2_lock_path),
        },
        "planned_language_evaluation": {
            "records": rows,
            "primary_model_forward_passes": rows * 4,
            "v26_baseline_model_forward_passes": rows,
            "conditional_nli_maximum_model_forward_passes": rows,
            "maximum_total_model_forward_passes": rows * 6,
            "head_fits": 0, "threshold_fits": 0, "hyperparameter_selections": 0,
        },
        "conditional_integration": {
            "run_condition": "all_primary_language_evaluation_gates_pass",
            "evaluation_scenes": len(evaluation_scenes),
            "planned_evidence_clauses": evidence_clauses,
            "fields_per_clause": 4,
            "planned_model_forward_passes": evidence_clauses * 4,
            "v28_integration_replays": 1,
        },
        "implementation": {path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION},
        "data_access_before_lock": {
            "benchmark_constructions": 1, "evaluation_predictions": 0,
            "model_forward_passes": 0, "primary_evaluations": 0,
            "v26_baseline_evaluations": 0, "candidate_nli_evaluations": 0,
            "v28_integration_replays": 0, "head_fits": 0, "threshold_fits": 0,
            "hyperparameter_selections": 0, "adapter_training_runs": 0,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
