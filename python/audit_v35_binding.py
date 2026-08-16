#!/usr/bin/env python3
"""Pre-run source, population, prompt, and budget audit for V35."""

from __future__ import annotations

import argparse
import copy
import json

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v35_binding import atom_prompt_layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v35-binding-assembly.json")
    parser.add_argument("--output", default="outputs/v35-binding-assembly/pre-run-audit.json")
    args = parser.parse_args()
    config_path, output_path = (PROJECT_ROOT / args.config).resolve(), (PROJECT_ROOT / args.output).resolve()
    config, errors = json.loads(config_path.read_text()), []
    v32_lock_path = PROJECT_ROOT / config["sourceV32ProtocolLock"]
    v32_metadata_path = PROJECT_ROOT / config["sourceV32FeatureMetadata"]
    v32_audit_path = PROJECT_ROOT / config["sourceV32PostAudit"]
    v34_lock_path = PROJECT_ROOT / config["sourceV34ProtocolLock"]
    v34_outcome_path = PROJECT_ROOT / config["sourceV34OutcomeLock"]
    v34_result_path = PROJECT_ROOT / config["sourceV34Result"]
    v34_predictions_path = PROJECT_ROOT / config["sourceV34Predictions"]
    v32_lock, v32_metadata, v32_audit, v34_outcome, v34_result = (
        json.loads(path.read_text()) for path in (
            v32_lock_path, v32_metadata_path, v32_audit_path, v34_outcome_path, v34_result_path,
        )
    )
    config_for_prompt = {**config, "v32_config": v32_lock["config_payload"]}
    if not v32_audit["passed"]:
        errors.append("Accepted V32 result is unavailable")
    if not v34_outcome["qualification"]["passed"] or not v34_outcome["authorization"]["continue_binding_and_assembly_development"]:
        errors.append("V34 outcome does not authorize V35 development")
    if v34_outcome["result_sha256"] != file_sha256(v34_result_path):
        errors.append("V34 outcome does not bind the registered result")
    if v34_result["predictions_sha256"] != file_sha256(v34_predictions_path):
        errors.append("V34 registered predictions changed")
    rows = sorted(read_rows(PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])), key=lambda row: row["id"])
    fit = [row for row in rows if row["split"] == "factor_fit"]
    calibration = [row for row in rows if row["split"] == "factor_calibration"]
    if len(rows) != config["limits"]["backboneForwardPasses"]:
        errors.append("V35 forward-pass budget differs from allowed population")
    expected_fits = (3 + 2 + 1) * (len(config["readouts"]["alphas"]) * 4 + 1)
    if expected_fits != config["limits"]["ridgeTrainingFits"]:
        errors.append("V35 ridge-fit budget mismatch")
    target_dependencies = []
    for row in rows:
        content, spans = atom_prompt_layout(row, config_for_prompt)
        mutated = copy.deepcopy(row); mutated["target"] = {"sentinel": "must not affect prompt"}
        if atom_prompt_layout(mutated, config_for_prompt) != (content, spans):
            target_dependencies.append(row["id"])
    if target_dependencies:
        errors.append("V35 prompt or entity spans depend on target fields")
    forbidden = (
        PROJECT_ROOT / "configs/v35-binding-assembly-lock.json",
        PROJECT_ROOT / "configs/v35-binding-outcome-lock.json",
        PROJECT_ROOT / "outputs/v35-binding-assembly/feature-attempt.json",
        PROJECT_ROOT / "outputs/v35-binding-assembly/result.json",
    )
    if any(path.exists() for path in forbidden):
        errors.append("V35 model/result artifact exists before protocol lock")
    feature_path = PROJECT_ROOT / v32_metadata["feature_artifact"]
    corpus_hashes = {
        f"{split}.jsonl": file_sha256(PROJECT_ROOT / config["sourceCorpus"] / f"{split}.jsonl")
        for split in config["allowedSplits"]
    }
    result = {
        "schema_version": 35, "experiment": "v35_pre_run_audit", "passed": not errors,
        "decision": "authorize_v35_protocol_lock" if not errors else "repair_before_v35", "errors": errors,
        "population": {"records": len(rows), "fit_records": len(fit), "calibration_records": len(calibration), "forbidden_evaluation_records_read": 0},
        "budget": {"backbone_forward_passes": len(rows), "ridge_training_fits": expected_fits},
        "source": {
            "config_sha256": file_sha256(config_path), "v32_protocol_lock_sha256": file_sha256(v32_lock_path),
            "v32_feature_metadata_sha256": file_sha256(v32_metadata_path), "v32_feature_artifact_sha256": file_sha256(feature_path),
            "v32_post_audit_sha256": file_sha256(v32_audit_path), "v34_protocol_lock_sha256": file_sha256(v34_lock_path),
            "v34_outcome_lock_sha256": file_sha256(v34_outcome_path), "v34_result_sha256": file_sha256(v34_result_path),
            "v34_predictions_sha256": file_sha256(v34_predictions_path), "allowed_corpus_file_sha256": corpus_hashes,
        },
        "firewall": {"prompt_target_fields": 0, "v32_evaluation_records_read": 0, "v32_evaluation_features_read": 0, "v32_evaluation_predictions_read": 0, "v28_integration_replays": 0, "fresh_suite_constructions": 0},
        "v32_config_payload": v32_lock["config_payload"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
