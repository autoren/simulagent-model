#!/usr/bin/env python3
"""Pre-run source, population, and firewall audit for V34."""

from __future__ import annotations

import argparse
import copy
import json

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v34_operation import operation_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v34-operation-interface.json")
    parser.add_argument("--output", default="outputs/v34-operation-interface/pre-run-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    config, errors = json.loads(config_path.read_text()), []
    v32_lock_path = PROJECT_ROOT / config["sourceV32ProtocolLock"]
    v32_metadata_path = PROJECT_ROOT / config["sourceV32FeatureMetadata"]
    v32_audit_path = PROJECT_ROOT / config["sourceV32PostAudit"]
    v33_outcome_path = PROJECT_ROOT / config["sourceV33OutcomeLock"]
    v32_lock = json.loads(v32_lock_path.read_text())
    v32_metadata = json.loads(v32_metadata_path.read_text())
    v32_audit = json.loads(v32_audit_path.read_text())
    v33_outcome = json.loads(v33_outcome_path.read_text())
    if not v32_audit["passed"]:
        errors.append("Accepted V32 result is unavailable")
    if v32_metadata["protocol_lock_sha256"] != file_sha256(v32_lock_path):
        errors.append("V32 feature metadata does not bind the V32 protocol")
    if v33_outcome["decision"] != "stop_before_fresh_suite_and_pivot_as_diagnosed":
        errors.append("V33 outcome does not authorize a development pivot")
    rows = sorted(read_rows(
        PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])
    ), key=lambda row: row["id"])
    fit = [row for row in rows if row["split"] == "factor_fit"]
    calibration = [row for row in rows if row["split"] == "factor_calibration"]
    if len(rows) != config["limits"]["backboneForwardPasses"]:
        errors.append("V34 forward-pass budget differs from allowed population")
    if sorted({row["oracle_metadata"]["surface_name"] for row in fit}) != ["fit_a", "fit_b", "fit_c", "fit_d"]:
        errors.append("V34 fit surface groups differ from the registered four-fold design")
    expected_fits = 3 * (len(config["readouts"]["alphas"]) * 4 + 1)
    if expected_fits != config["limits"]["ridgeTrainingFits"]:
        errors.append("V34 ridge-fit budget mismatch")
    target_dependency = []
    for row in rows:
        prompt = operation_prompt(row, config)
        mutated = copy.deepcopy(row)
        mutated["target"] = {"sentinel": "target fields must not affect this prompt"}
        if operation_prompt(mutated, config) != prompt:
            target_dependency.append(row["id"])
    if target_dependency:
        errors.append("V34 prompt depends on target fields")
    forbidden_artifacts = (
        PROJECT_ROOT / "configs/v34-operation-interface-lock.json",
        PROJECT_ROOT / "configs/v34-operation-outcome-lock.json",
        PROJECT_ROOT / "outputs/v34-operation-interface/feature-attempt.json",
        PROJECT_ROOT / "outputs/v34-operation-interface/result.json",
    )
    if any(path.exists() for path in forbidden_artifacts):
        errors.append("V34 model/result artifact exists before the protocol lock")
    corpus_hashes = {
        f"{split}.jsonl": file_sha256(PROJECT_ROOT / config["sourceCorpus"] / f"{split}.jsonl")
        for split in config["allowedSplits"]
    }
    feature_path = PROJECT_ROOT / v32_metadata["feature_artifact"]
    result = {
        "schema_version": 34, "experiment": "v34_pre_run_audit",
        "passed": not errors,
        "decision": "authorize_v34_protocol_lock" if not errors else "repair_before_v34",
        "errors": errors,
        "population": {"records": len(rows), "fit_records": len(fit), "calibration_records": len(calibration), "forbidden_evaluation_records_read": 0},
        "budget": {"backbone_forward_passes": len(rows), "ridge_training_fits": expected_fits},
        "source": {
            "config_sha256": file_sha256(config_path),
            "v32_protocol_lock_sha256": file_sha256(v32_lock_path),
            "v32_feature_metadata_sha256": file_sha256(v32_metadata_path),
            "v32_feature_artifact_sha256": file_sha256(feature_path),
            "v32_post_audit_sha256": file_sha256(v32_audit_path),
            "v33_outcome_lock_sha256": file_sha256(v33_outcome_path),
            "allowed_corpus_file_sha256": corpus_hashes,
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
