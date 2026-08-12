"""Freeze the V26 full-depth native truth-decoder protocol."""

from __future__ import annotations

import argparse
import hashlib
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/v26_native_decoder.py",
    "python/build_v26_native_decoder.py",
    "python/audit_v26_native_decoder.py",
    "python/evaluate_v26_native_decoder_mlx.py",
    "python/test_v26_native_decoder.py",
    "python/freeze_v26_native_decoder.py",
    "python/extract_v10_features_mlx.py",
    "python/evaluate_v25_truth_hypotheses.py",
    "python/evaluate_v22r2_relational_grounding.py",
    "python/v22_relational.py",
    "python/v22r2_grounding.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v26-native-truth-decoder.json")
    parser.add_argument("--plan", default="docs/v26-native-truth-decoder-plan.md")
    parser.add_argument("--audit", default="outputs/v26-native-truth-decoder/pre-evaluation-audit.json")
    parser.add_argument("--output", default="configs/v26-native-truth-decoder-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    audit_path = (PROJECT_ROOT / args.audit).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V26 protocol lock already exists")
    output_root = PROJECT_ROOT / "outputs/v26-native-truth-decoder"
    for path in (output_root / "evaluation", output_root / "evaluation-attempt.json"):
        if path.exists():
            raise RuntimeError(f"V26 evaluation artifact exists before lock: {path}")
    config = json.loads(config_path.read_text())
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["decision"] != "authorize_v26_protocol_lock":
        raise RuntimeError("V26 pre-evaluation audit does not authorize lock")
    corpus_root = PROJECT_ROOT / config["outputDir"]
    manifest_path = corpus_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("V26 manifest and config differ")
    corpus_files = {
        path.name: file_sha256(path) for path in sorted(corpus_root.glob("*.jsonl"))
    }
    corpus_files["manifest.json"] = file_sha256(manifest_path)
    v25_audit = json.loads((PROJECT_ROOT / config["sourceV25PostAudit"]).read_text())
    v25_result = json.loads((PROJECT_ROOT / config["sourceV25Result"]).read_text())
    if (
        not v25_audit["passed"] or v25_audit["decision"] != "accept_v25_exposed_development_result"
        or v25_result["decision"] != "explicit_truth_hypotheses_insufficient_no_lora"
    ):
        raise RuntimeError("V25 source status does not authorize V26")
    lock = {
        "schema_version": 26,
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "model": config["model"],
        "inference": config["inference"],
        "integration_conditions": config["integrationConditions"],
        "gates": config["gates"],
        "limits": config["limits"],
        "pre_evaluation_audit": audit,
        "source": {
            "corpus": config["outputDir"],
            "corpus_manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
            "corpus_manifest_sha256": file_sha256(manifest_path),
            "corpus_sha256": manifest["corpus_sha256"],
            "corpus_file_sha256": corpus_files,
            "pre_evaluation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
            "pre_evaluation_audit_sha256": file_sha256(audit_path),
            "v25_lock": config["sourceV25Lock"],
            "v25_lock_sha256": file_sha256(PROJECT_ROOT / config["sourceV25Lock"]),
            "v25_result": config["sourceV25Result"],
            "v25_result_sha256": file_sha256(PROJECT_ROOT / config["sourceV25Result"]),
            "v25_post_audit": config["sourceV25PostAudit"],
            "v25_post_audit_sha256": file_sha256(PROJECT_ROOT / config["sourceV25PostAudit"]),
            "v25_diagnostic": config["sourceV25Diagnostic"],
            "v25_diagnostic_sha256": file_sha256(PROJECT_ROOT / config["sourceV25Diagnostic"]),
            "v24_predictions": config["sourceV24Predictions"],
            "v24_predictions_sha256": file_sha256(PROJECT_ROOT / config["sourceV24Predictions"]),
        },
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_lock": {
            "all_v22r2_splits_exposed": True,
            "v24_assignments_exposed_and_frozen": True,
            "new_model_forward_passes": 0,
            "head_fits": 0,
            "threshold_fits": 0,
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
