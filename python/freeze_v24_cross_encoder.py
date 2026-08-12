"""Freeze the exposed-data V24 candidate-conditioned cross-encoder protocol."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v24_cross_encoder.py",
    "python/build_v24_cross_encoder_proposals.py",
    "python/audit_v24_cross_encoder.py",
    "python/extract_v24_cross_features_mlx.py",
    "python/evaluate_v24_cross_encoder.py",
    "python/test_v24_cross_encoder.py",
    "python/freeze_v24_cross_encoder.py",
    "python/extract_v10_features_mlx.py",
    "python/extract_v11_scale_features_mlx.py",
    "python/extract_v22r2_relational_features_mlx.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v24-cross-encoder.json")
    parser.add_argument("--plan", default="docs/v24-cross-encoder-plan.md")
    parser.add_argument("--audit", default="outputs/v24-cross-encoder/pre-extraction-audit.json")
    parser.add_argument("--output", default="configs/v24-cross-encoder-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V24 protocol lock already exists")
    feature_root = PROJECT_ROOT / "outputs/v24-cross-encoder/features"
    evaluation_root = PROJECT_ROOT / "outputs/v24-cross-encoder/evaluation"
    for path in (
        feature_root,
        evaluation_root,
        feature_root.parent / "feature-extraction-attempt.json",
        evaluation_root.parent / "evaluation-attempt.json",
    ):
        if path.exists():
            raise RuntimeError(f"V24 model/evaluation artifact exists before lock: {path}")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v24_protocol_lock":
        raise RuntimeError("V24 pre-extraction audit does not authorize protocol lock")
    firewall = audit["firewall"]
    if firewall["new_model_forward_passes_before_lock"] or firewall["new_linear_fits_before_lock"]:
        raise RuntimeError("V24 audit reports model access before lock")
    if audit["budget"]["planned_model_forwards"] > config["gates"]["preExtraction"]["maximumNewModelForwardPasses"]:
        raise RuntimeError("V24 planned extraction exceeds the registered budget")

    proposal_root = PROJECT_ROOT / config["outputDir"]
    manifest_path = proposal_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V24 proposal manifest does not share the current config")
    proposal_files = {
        path.name: file_sha256(path) for path in sorted(proposal_root.glob("*.jsonl"))
    }
    proposal_files["manifest.json"] = file_sha256(manifest_path)

    v22r2_lock_path = PROJECT_ROOT / config["sourceV22r2Lock"]
    v22r2a_result = json.loads((PROJECT_ROOT / config["sourceV22r2aResult"]).read_text())
    v22r2a_audit = json.loads((PROJECT_ROOT / config["sourceV22r2aPostAudit"]).read_text())
    v23_result = json.loads((PROJECT_ROOT / config["sourceV23Result"]).read_text())
    v23_audit = json.loads((PROJECT_ROOT / config["sourceV23PostAudit"]).read_text())
    if (
        not v22r2a_audit["passed"]
        or v22r2a_audit["decision"] != "accept_v22r2a_negative_result"
        or v22r2a_result["decision"] != "develop_probabilistic_support_interface_no_lora"
    ):
        raise RuntimeError("V22r2a source status does not authorize V24 continuation")
    if (
        not v23_audit["passed"]
        or v23_audit["decision"] != "accept_v23_negative_result"
        or v23_result["decision"] != "probabilistic_support_insufficient_revise_language_interface"
    ):
        raise RuntimeError("V23 source status does not authorize V24 continuation")

    lock = {
        "schema_version": 24,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "model": config["model"],
        "heads": config["heads"],
        "integration_conditions": config["integrationConditions"],
        "gates": config["gates"],
        "limits": config["limits"],
        "pre_extraction_audit": audit,
        "source": {
            "proposal_corpus": config["outputDir"],
            "proposal_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "proposal_manifest_sha256": file_sha256(manifest_path),
            "proposal_corpus_sha256": manifest["corpus_sha256"],
            "proposal_file_sha256": proposal_files,
            "pre_extraction_audit": str(audit_path.relative_to(PROJECT_ROOT)),
            "pre_extraction_audit_sha256": file_sha256(audit_path),
            "v22r2_lock": config["sourceV22r2Lock"],
            "v22r2_lock_sha256": file_sha256(v22r2_lock_path),
            "v22r2a_lock": config["sourceV22r2aLock"],
            "v22r2a_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV22r2aLock"]),
            "v22r2a_result": config["sourceV22r2aResult"],
            "v22r2a_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV22r2aResult"]),
            "v22r2a_post_audit": config["sourceV22r2aPostAudit"],
            "v22r2a_post_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV22r2aPostAudit"]),
            "v23_result": config["sourceV23Result"],
            "v23_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV23Result"]),
            "v23_post_audit": config["sourceV23PostAudit"],
            "v23_post_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV23PostAudit"]),
        },
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_lock": {
            "all_v22r2_splits_exposed": True,
            "pair_proposals_materialized": manifest["pairs"],
            "new_model_forward_passes": 0,
            "new_feature_extractions": 0,
            "new_linear_fits": 0,
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
