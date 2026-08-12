"""Freeze the V27 outcome-constrained support MAP protocol."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/build_v27_support_edges.py",
    "python/audit_v27_support_map.py",
    "python/v27_support_map.py",
    "python/score_v27_support_edges_mlx.py",
    "python/evaluate_v27_support_map.py",
    "python/test_v27_support_map.py",
    "python/freeze_v27_support_map.py",
    "python/evaluate_v26_native_decoder_mlx.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/v23_probabilistic_relational.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v27-support-map.json")
    parser.add_argument("--plan", default="docs/v27-support-map-plan.md")
    parser.add_argument("--audit", default="outputs/v27-support-map/pre-decoder-audit.json")
    parser.add_argument("--output", default="configs/v27-support-map-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V27 protocol lock already exists")
    output_root = PROJECT_ROOT / "outputs/v27-support-map"
    for path in (
        output_root / "edge-scores", output_root / "evaluation",
        output_root / "edge-decoder-attempt.json", output_root / "map-evaluation-attempt.json",
    ):
        if path.exists():
            raise RuntimeError(f"V27 model/MAP artifact exists before lock: {path}")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v27_protocol_lock":
        raise RuntimeError("V27 audit does not authorize protocol lock")
    corpus_root = PROJECT_ROOT / config["outputDir"]
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V27 manifest and config differ")
    corpus_files = {
        path.name: file_sha256(path) for path in sorted(corpus_root.glob("*.jsonl"))
    }
    corpus_files["manifest.json"] = file_sha256(manifest_path)
    v26_audit = json.loads((PROJECT_ROOT / config["sourceV26PostAudit"]).read_text())
    v26_result = json.loads((PROJECT_ROOT / config["sourceV26Result"]).read_text())
    if (
        not v26_audit["passed"] or v26_audit["decision"] != "accept_v26_exposed_development_result"
        or v26_result["decision"] != "repair_exact_graph_or_symbolic_composition_no_lora"
    ):
        raise RuntimeError("V26 source status does not authorize V27")
    feature_metadata = json.loads((PROJECT_ROOT / config["sourceV24Features"]).read_text())
    lock = {
        "schema_version": 27,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "model": config["model"],
        "labels": ["A", "B", "C"],
        "joint_map": config["jointMap"],
        "integration_conditions": config["integrationConditions"],
        "gates": config["gates"],
        "limits": config["limits"],
        "pre_decoder_audit": audit,
        "source_v26_reference": {
            "frozen_support_oracle_query_exact": v26_result["integration"]["frozen_support_oracle_query"]["transition_set_exact_match"],
            "frozen_frozen_exact": v26_result["integration"]["frozen_support_frozen_query"]["transition_set_exact_match"],
            "target_retention": v26_result["integration"]["frozen_support_oracle_query"]["target_retention_rate"],
            "empty_version_space": v26_result["integration"]["frozen_support_oracle_query"]["empty_version_space_rate"],
        },
        "source": {
            "corpus": config["outputDir"],
            "corpus_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "corpus_manifest_sha256": file_sha256(manifest_path),
            "corpus_sha256": manifest["corpus_sha256"],
            "corpus_file_sha256": corpus_files,
            "pre_decoder_audit": str(audit_path.relative_to(PROJECT_ROOT)),
            "pre_decoder_audit_sha256": file_sha256(audit_path),
            "v26_lock": config["sourceV26Lock"],
            "v26_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV26Lock"]),
            "v26_result": config["sourceV26Result"],
            "v26_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV26Result"]),
            "v26_post_audit": config["sourceV26PostAudit"],
            "v26_post_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV26PostAudit"]),
            "v26_residual": config["sourceV26Residual"],
            "v26_residual_sha256": file_sha256(PROJECT_ROOT / config["sourceV26Residual"]),
            "v26_scores": config["sourceV26Scores"],
            "v26_scores_sha256": file_sha256(PROJECT_ROOT / config["sourceV26Scores"]),
            "v26_predictions": config["sourceV26Predictions"],
            "v26_predictions_sha256": file_sha256(PROJECT_ROOT / config["sourceV26Predictions"]),
            "v24_proposal_corpus": config["sourceV24ProposalCorpus"],
            "v24_proposal_manifest_sha256": file_sha256(
                PROJECT_ROOT / config["sourceV24ProposalCorpus"] / "manifest.json"
            ),
            "v24_feature_metadata": config["sourceV24Features"],
            "v24_feature_metadata_sha256": file_sha256(PROJECT_ROOT / config["sourceV24Features"]),
            "v24_feature_artifact": feature_metadata["feature_artifact"],
            "v24_feature_artifact_sha256": feature_metadata["feature_artifact_sha256"],
            "v24_heads": config["sourceV24Heads"],
            "v24_heads_sha256": file_sha256(PROJECT_ROOT / config["sourceV24Heads"]),
        },
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_lock": {
            "all_v22r2_splits_exposed": True,
            "new_model_forward_passes": 0, "head_fits": 0, "threshold_fits": 0,
            "joint_map_evaluations": 0, "hyperparameter_selections": 0,
            "adapter_training_runs": 0, "fresh_benchmark_records_created": 0,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
