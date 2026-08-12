"""Freeze the nondiscretionary V22r2a scikit-learn execution amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


IMPLEMENTATION = (
    "python/evaluate_v22r2_relational_grounding.py",
    "python/evaluate_v22r2a_relational_grounding.py",
    "python/v22r2_grounding.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v22r2a-evaluation-amendment.json")
    parser.add_argument("--plan", default="docs/v22r2a-evaluation-amendment.md")
    parser.add_argument("--output", default="configs/v22r2a-evaluation-amendment-lock.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    plan_path = (PROJECT_ROOT / args.plan).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    if output_path.exists():
        raise RuntimeError("V22r2a amendment lock already exists")
    config = json.loads(config_path.read_text())
    original_lock_path = PROJECT_ROOT / config["sourceLock"]
    feature_metadata_path = PROJECT_ROOT / config["sourceFeatures"]
    attempt_path = PROJECT_ROOT / config["failedAttempt"]
    failure_path = PROJECT_ROOT / config["failureRecord"]
    replacement_output = PROJECT_ROOT / "outputs/v22r2-relational-grounding/evaluation-v22r2a"
    replacement_attempt = PROJECT_ROOT / "outputs/v22r2-relational-grounding/evaluation-v22r2a-attempt.json"
    if replacement_output.exists() or replacement_attempt.exists():
        raise RuntimeError("V22r2a replacement artifacts exist before the amendment lock")
    original_lock = json.loads(original_lock_path.read_text())
    metadata = json.loads(feature_metadata_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    failure = json.loads(failure_path.read_text())
    if attempt["status"] != "started_before_head_fitting":
        raise RuntimeError("Original V22r2 attempt ledger has unexpected status")
    if failure["status"] != "aborted_before_predictions":
        raise RuntimeError("V22r2 failure record does not establish a pre-prediction abort")
    if failure["scientific_information_revealed"]["evaluation_metrics"] != 0:
        raise RuntimeError("V22r2 failure record reports evaluation information")
    if metadata["protocol_lock_sha256"] != file_sha256(original_lock_path):
        raise RuntimeError("Frozen features do not share the original V22r2 lock")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V22r2 frozen feature artifact changed")
    lock = {
        "schema_version": "22r2a",
        "experiment": config["experiment"],
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "preregistration": str(plan_path.relative_to(PROJECT_ROOT)),
        "preregistration_sha256": file_sha256(plan_path),
        "amendment": config["amendment"],
        "replacement_limits": config["replacementLimits"],
        "source": {
            "original_lock": config["sourceLock"],
            "original_lock_sha256": file_sha256(original_lock_path),
            "feature_metadata": config["sourceFeatures"],
            "feature_metadata_sha256": file_sha256(feature_metadata_path),
            "feature_artifact": metadata["feature_artifact"],
            "feature_artifact_sha256": metadata["feature_artifact_sha256"],
            "failed_attempt": config["failedAttempt"],
            "failed_attempt_sha256": file_sha256(attempt_path),
            "failure_record": config["failureRecord"],
            "failure_record_sha256": file_sha256(failure_path),
        },
        "gates": original_lock["gates"],
        "integration_conditions": original_lock["integration_conditions"],
        "implementation": {
            path: file_sha256(PROJECT_ROOT / path) for path in IMPLEMENTATION
        },
        "data_access_before_amendment_lock": {
            "completed_model_forward_passes": metadata["new_model_forward_passes"],
            "aborted_atom_head_fits": 1,
            "truth_head_fits": 0,
            "held_out_predictions_read": 0,
            "evaluation_metrics_read": 0,
            "integration_evaluations": 0,
            "adapter_training_runs": 0,
        },
    }
    lock["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
