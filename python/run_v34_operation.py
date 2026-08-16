#!/usr/bin/env python3
"""Select V34 ridge readouts on fit groups, then score calibration once."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v34_operation import (
    cross_validate_ridge, make_ridge, qualification, score_operations, select_alpha,
    select_prompt_method, surface_groups, target_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v34-operation-interface-lock.json")
    parser.add_argument("--feature-metadata", default="outputs/v34-operation-interface/features/metadata.json")
    parser.add_argument("--output", default="outputs/v34-operation-interface/result.json")
    args = parser.parse_args()
    lock_path, metadata_path, output_path = map(
        lambda value: (PROJECT_ROOT / value).resolve(),
        (args.lock, args.feature_metadata, args.output),
    )
    attempt_path = output_path.parent / "run-attempt.json"
    if output_path.exists() or attempt_path.exists():
        raise RuntimeError("V34 readout run was already attempted")
    lock, metadata = json.loads(lock_path.read_text()), json.loads(metadata_path.read_text())
    config = lock["config_payload"]
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V34 locked implementation changed: {path}")
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V34 feature metadata does not bind this lock")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V34 semantic feature artifact changed")
    legacy_metadata_path = PROJECT_ROOT / config["sourceV32FeatureMetadata"]
    legacy_metadata = json.loads(legacy_metadata_path.read_text())
    legacy_path = PROJECT_ROOT / legacy_metadata["feature_artifact"]
    if file_sha256(legacy_path) != lock["source"]["v32_feature_artifact_sha256"]:
        raise RuntimeError("V34 legacy V32 feature artifact changed")
    rows = sorted(read_rows(PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])), key=lambda row: row["id"])
    semantic, legacy = np.load(feature_path), np.load(legacy_path)
    ids = [row["id"] for row in rows]
    if semantic["record_ids"].tolist() != ids or legacy["record_ids"].tolist() != ids:
        raise RuntimeError("V34 feature populations do not match allowed records")
    fit_indices = np.flatnonzero(semantic["splits"] == "factor_fit")
    calibration_indices = np.flatnonzero(semantic["splits"] == "factor_calibration")
    fit_rows = [rows[index] for index in fit_indices]
    calibration_rows = [rows[index] for index in calibration_indices]
    targets = target_indices(rows, config)
    groups = surface_groups(fit_rows)
    method_features = {
        "legacyHiddenRidge": legacy["clause_features"],
        "semanticHiddenRidge": semantic["semantic_hidden"],
        "nativeLogitRidge": semantic["native_label_logits"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 34, "attempt_number": 1, "status": "started",
        "protocol_lock_sha256": file_sha256(lock_path),
        "calibration_opened": False, "v32_evaluation_records_read": 0,
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    methods, predictions = {}, {}
    ridge_fit_count = 0
    for name, features in method_features.items():
        reports = cross_validate_ridge(
            features[fit_indices], targets[fit_indices], groups,
            config["readouts"]["alphas"],
        )
        ridge_fit_count += len(reports) * len(set(groups.tolist()))
        selected_cv = select_alpha(reports)
        model = make_ridge(selected_cv["alpha"])
        model.fit(features[fit_indices], targets[fit_indices])
        ridge_fit_count += 1
        predictions[name] = {
            "fit": model.predict(features[fit_indices]),
            "calibration": model.predict(features[calibration_indices]),
        }
        methods[name] = {"cv_reports": reports, "selected_cv": selected_cv}
    selected_method = select_prompt_method(methods)
    # Calibration is opened only after all method/alpha choices above are fixed from fit CV.
    attempt = json.loads(attempt_path.read_text())
    attempt["calibration_opened"] = True
    attempt["fit_selected_prompt_method"] = selected_method
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    predictions["nativeArgmax"] = {
        "fit": np.argmax(semantic["native_label_logits"][fit_indices], axis=1),
        "calibration": np.argmax(semantic["native_label_logits"][calibration_indices], axis=1),
    }
    for name in config["readouts"]["methods"]:
        methods.setdefault(name, {})
        methods[name]["fit"] = score_operations(fit_rows, predictions[name]["fit"], {**config, "v32_config": lock["v32_config_payload"]})
        methods[name]["calibration"] = score_operations(calibration_rows, predictions[name]["calibration"], {**config, "v32_config": lock["v32_config_payload"]})
    selected_qualification = qualification(
        methods[selected_method]["calibration"], methods["legacyHiddenRidge"]["calibration"], config
    )
    gain = selected_qualification["calibration_operation_gain_over_legacy"]
    if selected_qualification["passed"]:
        decision = "operation_interface_qualified_continue_binding_and_assembly"
    elif gain >= 0.10:
        decision = "operation_interface_improves_continue_repair_without_new_suite"
    else:
        decision = "operation_interface_insufficient_pivot_parser_or_grounder"
    prediction_path = output_path.parent / "predictions.npz"
    np.savez_compressed(
        prediction_path, record_ids=np.asarray(ids),
        **{f"{name}_{split}": np.asarray(values[split], dtype=np.int64)
           for name, values in predictions.items() for split in ("fit", "calibration")},
    )
    result = {
        "schema_version": 34, "experiment": config["experiment"],
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_metadata": str(metadata_path.relative_to(PROJECT_ROOT)), "feature_metadata_sha256": file_sha256(metadata_path),
        "methods": methods, "fit_selected_prompt_method": selected_method,
        "qualification": selected_qualification, "decision": decision,
        "predictions": str(prediction_path.relative_to(PROJECT_ROOT)), "predictions_sha256": file_sha256(prediction_path),
        "runtime_seconds": time.perf_counter() - started,
        "authorization": {"continue_binding_and_assembly_development": selected_qualification["passed"], "fresh_suite": False, "v32_evaluation_reuse": False, "v28_replay": False},
        "data_access": {
            "fit_records_read": len(fit_rows), "calibration_records_read": len(calibration_rows),
            "ridge_training_fits": ridge_fit_count,
            "backbone_forward_passes": metadata["data_access"]["backbone_forward_passes"],
            "calibration_selections": 0, "v32_evaluation_records_read": 0,
            "v32_evaluation_features_read": 0, "v32_evaluation_predictions_read": 0,
            "adapter_training_runs": 0, "v28_integration_replays": 0, "fresh_suite_constructions": 0,
        },
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt.update({"status": "completed", "result_sha256": file_sha256(output_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "fit_selected_prompt_method": selected_method,
        "calibration": {name: values["calibration"] for name, values in methods.items()},
        "qualification": selected_qualification, "decision": decision,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
