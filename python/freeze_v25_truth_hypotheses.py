"""Freeze the V25 explicit truth-hypothesis protocol."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v25_truth_hypotheses.py",
    "python/build_v25_truth_hypotheses.py",
    "python/audit_v25_truth_hypotheses.py",
    "python/extract_v25_truth_features_mlx.py",
    "python/evaluate_v25_truth_hypotheses.py",
    "python/test_v25_truth_hypotheses.py",
    "python/freeze_v25_truth_hypotheses.py",
    "python/extract_v10_features_mlx.py",
    "python/extract_v11_scale_features_mlx.py",
    "python/extract_v22r2_relational_features_mlx.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v25-truth-hypotheses.json")
    parser.add_argument("--plan", default="docs/v25-truth-hypotheses-plan.md")
    parser.add_argument("--audit", default="outputs/v25-truth-hypotheses/pre-extraction-audit.json")
    parser.add_argument("--output", default="configs/v25-truth-hypotheses-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V25 protocol lock already exists")
    output_root = PROJECT_ROOT / "outputs/v25-truth-hypotheses"
    for path in (
        output_root / "features", output_root / "evaluation",
        output_root / "feature-extraction-attempt.json", output_root / "evaluation-attempt.json",
    ):
        if path.exists():
            raise RuntimeError(f"V25 model/evaluation artifact exists before lock: {path}")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v25_protocol_lock":
        raise RuntimeError("V25 audit does not authorize protocol lock")
    corpus_root = PROJECT_ROOT / config["outputDir"]
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V25 manifest and config differ")
    corpus_files = {
        path.name: file_sha256(path) for path in sorted(corpus_root.glob("*.jsonl"))
    }
    corpus_files["manifest.json"] = file_sha256(manifest_path)
    v24_audit = json.loads((PROJECT_ROOT / config["sourceV24PostAudit"]).read_text())
    v24_result = json.loads((PROJECT_ROOT / config["sourceV24Result"]).read_text())
    if (
        not v24_audit["passed"]
        or v24_audit["decision"] != "accept_v24_exposed_development_result"
        or v24_result["decision"] != "factor_truth_semantics_before_fresh_benchmark_no_lora"
    ):
        raise RuntimeError("V24 source status does not authorize V25")

    lock = {
        "schema_version": 25,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "model": config["model"],
        "head": config["head"],
        "integration_conditions": config["integrationConditions"],
        "gates": config["gates"],
        "limits": config["limits"],
        "pre_extraction_audit": audit,
        "source": {
            "corpus": config["outputDir"],
            "corpus_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "corpus_manifest_sha256": file_sha256(manifest_path),
            "corpus_sha256": manifest["corpus_sha256"],
            "corpus_file_sha256": corpus_files,
            "pre_extraction_audit": str(audit_path.relative_to(PROJECT_ROOT)),
            "pre_extraction_audit_sha256": file_sha256(audit_path),
            "v24_lock": config["sourceV24Lock"],
            "v24_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV24Lock"]),
            "v24_result": config["sourceV24Result"],
            "v24_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV24Result"]),
            "v24_post_audit": config["sourceV24PostAudit"],
            "v24_post_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV24PostAudit"]),
            "v24_diagnostic": config["sourceV24Diagnostic"],
            "v24_diagnostic_sha256": file_sha256(PROJECT_ROOT / config["sourceV24Diagnostic"]),
            "v24_feature_metadata": config["sourceV24FeatureMetadata"],
            "v24_feature_metadata_sha256": file_sha256(PROJECT_ROOT / config["sourceV24FeatureMetadata"]),
            "v24_heads": config["sourceV24Heads"],
            "v24_heads_sha256": file_sha256(PROJECT_ROOT / config["sourceV24Heads"]),
            "v24_predictions": config["sourceV24Predictions"],
            "v24_predictions_sha256": file_sha256(PROJECT_ROOT / config["sourceV24Predictions"]),
        },
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_lock": {
            "all_v22r2_splits_exposed": True,
            "v24_assignments_exposed_and_frozen": True,
            "truth_hypothesis_rows_materialized": manifest["rows"],
            "new_model_forward_passes": 0,
            "new_feature_extractions": 0,
            "new_linear_fits": 0,
            "match_head_fits": 0,
            "integration_evaluations": 0,
            "hyperparameter_selections": 0,
            "adapter_training_runs": 0,
            "fresh_benchmark_records_created": 0,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
