#!/usr/bin/env python3
"""Pre-run source, budget, and firewall audit for V33."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_v32_factorized_semantics import read_rows
from train_v32_heads import load_arrays
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v33-development-adequacy.json")
    parser.add_argument("--output", default="outputs/v33-development-adequacy/pre-run-audit.json")
    args = parser.parse_args()
    config_path, output_path = (PROJECT_ROOT / args.config).resolve(), (PROJECT_ROOT / args.output).resolve()
    config, errors = json.loads(config_path.read_text()), []
    v32_protocol_path = PROJECT_ROOT / config["sourceV32ProtocolLock"]
    v32_trained_path = PROJECT_ROOT / config["sourceV32TrainedLock"]
    feature_metadata_path = PROJECT_ROOT / config["sourceV32FeatureMetadata"]
    post_audit_path = PROJECT_ROOT / config["sourceV32PostAudit"]
    v32_protocol, v32_trained, metadata, post_audit = (
        json.loads(path.read_text()) for path in (
            v32_protocol_path, v32_trained_path, feature_metadata_path, post_audit_path
        )
    )
    if not post_audit["passed"] or post_audit["decision"] != "accept_v32_result":
        errors.append("Accepted V32 result is unavailable")
    if v32_trained["protocol_lock_sha256"] != file_sha256(v32_protocol_path):
        errors.append("V32 trained lock does not bind the V32 protocol")
    if metadata["protocol_lock_sha256"] != file_sha256(v32_protocol_path):
        errors.append("V32 feature metadata does not bind the V32 protocol")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        errors.append("V32 fit/calibration feature artifact changed")
    arrays = load_arrays(feature_path)
    allowed_rows = sorted(read_rows(
        PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])
    ), key=lambda row: row["id"])
    if arrays["record_ids"].tolist() != [row["id"] for row in allowed_rows]:
        errors.append("Feature artifact contains a population other than registered fit/calibration")
    if set(arrays["splits"].tolist()) != set(config["allowedSplits"]):
        errors.append("Feature artifact split inventory differs from V33 allowed splits")
    if any(row["split"] in config["forbiddenSplits"] for row in allowed_rows):
        errors.append("Forbidden V32 evaluation record entered V33")
    objectives = config["search"]["objectives"]
    expected_paths = len(objectives) * len(config["search"]["learningRates"])
    expected_checkpoints = expected_paths * len(config["search"]["checkpointEpochs"])
    expected_confirmation = len(objectives) * len(config["confirmation"]["seeds"])
    if expected_paths != config["limits"]["searchTrainingPaths"]:
        errors.append("V33 search path budget mismatch")
    if expected_checkpoints != config["limits"]["searchCheckpointEvaluations"]:
        errors.append("V33 search checkpoint budget mismatch")
    if expected_confirmation != config["limits"]["confirmationTrainingRuns"]:
        errors.append("V33 confirmation budget mismatch")
    expected_objectives = {"atom", "truth", "lexicalSign", "outerOperation", "jointAuxiliary"}
    if set(objectives) != expected_objectives:
        errors.append("V33 objective inventory is incomplete")
    forbidden = (
        PROJECT_ROOT / "outputs/v33-development-adequacy/run-attempt.json",
        PROJECT_ROOT / "outputs/v33-development-adequacy/result.json",
        PROJECT_ROOT / "configs/v33-development-outcome-lock.json",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V33 training/result artifact exists before the pre-run lock")
    corpus_hashes = {
        f"{split}.jsonl": file_sha256(PROJECT_ROOT / config["sourceCorpus"] / f"{split}.jsonl")
        for split in config["allowedSplits"]
    }
    result = {
        "schema_version": 33, "experiment": "v33_pre_run_audit",
        "passed": not errors, "decision": "authorize_v33_development_lock" if not errors else "repair_before_v33",
        "errors": errors,
        "population": {
            "records": len(allowed_rows),
            "fit_records": int(np.sum(arrays["splits"] == "factor_fit")),
            "calibration_records": int(np.sum(arrays["splits"] == "factor_calibration")),
            "allowed_splits": sorted(set(arrays["splits"].tolist())),
            "forbidden_evaluation_records_read": 0,
        },
        "budget": {
            "search_training_paths": expected_paths,
            "search_checkpoint_evaluations": expected_checkpoints,
            "confirmation_training_runs": expected_confirmation,
        },
        "source": {
            "config_sha256": file_sha256(config_path),
            "v32_protocol_lock_sha256": file_sha256(v32_protocol_path),
            "v32_trained_lock_sha256": file_sha256(v32_trained_path),
            "feature_metadata_sha256": file_sha256(feature_metadata_path),
            "feature_artifact_sha256": file_sha256(feature_path),
            "v32_post_audit_sha256": file_sha256(post_audit_path),
            "allowed_corpus_file_sha256": corpus_hashes,
        },
        "firewall": {
            "v32_evaluation_records_read": 0, "v32_evaluation_features_read": 0,
            "v32_evaluation_predictions_read": 0, "backbone_forward_passes": 0,
            "v28_integration_replays": 0, "fresh_suite_constructions": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
