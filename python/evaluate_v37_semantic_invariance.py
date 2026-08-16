#!/usr/bin/env python3
"""Run the one locked V37 fit-only selection and development validation."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v36_interface import predict_component, unpack_component
from v37_semantic import (
    ALL_METHODS,
    cross_validate_component,
    fit_predict_method,
    qualification,
    score_semantics,
    select_method,
    semantic_predictions,
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def target_indices(rows, component, classes):
    key = "lexical_sign" if component == "lexical_sign" else "outer_operation"
    return np.asarray([
        classes.index(row["target"]["factorization"][key]) for row in rows
    ], dtype=np.int64)


def best_for_method(reports, method, method_order):
    return select_method([row for row in reports if row["method"] == method], method_order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-lock", default="configs/v37-features-lock.json")
    parser.add_argument("--output-dir", default="outputs/v37-semantic-invariance/evaluation")
    args = parser.parse_args()
    feature_lock_path = (PROJECT_ROOT / args.features_lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V37 validation evaluation was already attempted")
    feature_lock = json.loads(feature_lock_path.read_text())
    if feature_lock["authorization"]["validation_evaluations"] != 1:
        raise RuntimeError("V37 features lock does not authorize one evaluation")
    seal_path = PROJECT_ROOT / feature_lock["corpus_seal"]
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V37 locked implementation changed: {path}")
    metadata_path = PROJECT_ROOT / feature_lock["feature_metadata"]
    if file_sha256(metadata_path) != feature_lock["feature_metadata_sha256"]:
        raise RuntimeError("V37 feature metadata changed")
    metadata = json.loads(metadata_path.read_text())
    artifact_path = PROJECT_ROOT / feature_lock["feature_artifact"]
    if file_sha256(artifact_path) != feature_lock["feature_artifact_sha256"]:
        raise RuntimeError("V37 feature artifact changed")
    npz = np.load(artifact_path)
    ids = npz["record_ids"].tolist()
    config, v32_config = implementation["config_payload"], implementation["v32_config_payload"]
    fit_rows = sorted(read_jsonl(PROJECT_ROOT / seal["corpora"]["fit"]["path"]), key=lambda row: row["id"])
    validation_rows = sorted(read_jsonl(PROJECT_ROOT / seal["corpora"]["validation"]["path"]), key=lambda row: row["id"])
    fit_indices = np.asarray([ids.index(row["id"]) for row in fit_rows], dtype=np.int64)
    validation_indices = np.asarray([ids.index(row["id"]) for row in validation_rows], dtype=np.int64)
    if len(set(fit_indices.tolist() + validation_indices.tolist())) != len(ids):
        raise RuntimeError("V37 feature population does not exactly match sealed corpora")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 37,
        "attempt_number": 1,
        "status": "started",
        "features_lock_sha256": file_sha256(feature_lock_path),
        "fit_records": len(fit_rows),
        "validation_records": len(validation_rows),
        "validation_evaluations": 1,
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    component_specs = {
        "lexical_sign": {
            "classes": config["interfaces"]["lexicalSignClasses"],
            "candidate_hidden": npz["sign_candidate_hidden"],
            "candidate_margin": npz["sign_candidate_margin"],
            "direct_hidden": npz["direct_sign_hidden"],
        },
        "outer_operation": {
            "classes": config["interfaces"]["outerOperationClasses"],
            "candidate_hidden": npz["operation_candidate_hidden"],
            "candidate_margin": npz["operation_candidate_margin"],
            "direct_hidden": npz["direct_operation_hidden"],
        },
    }
    method_order = config["interfaces"]["methods"]
    alphas = config["developmentFit"]["alphas"]
    groups = [row["oracle_metadata"]["v37_selection_group"] for row in fit_rows]
    cv_reports, selected, validation_indices_by_component_method = {}, {}, {}
    for component, spec in component_specs.items():
        fit_bundle = {
            key: np.asarray(value[fit_indices], dtype=np.float32)
            for key, value in spec.items() if key != "classes"
        }
        validation_bundle = {
            key: np.asarray(value[validation_indices], dtype=np.float32)
            for key, value in spec.items() if key != "classes"
        }
        targets = target_indices(fit_rows, component, spec["classes"])
        reports = cross_validate_component(
            fit_bundle, targets, groups, alphas,
            config["developmentFit"]["selectionFolds"],
            config["developmentFit"]["samplingSeed"],
        )
        cv_reports[component] = reports
        selected[component] = select_method(reports, method_order)
        validation_indices_by_component_method[component] = {}
        for method in ALL_METHODS:
            choice = best_for_method(reports, method, method_order)
            predicted, _ = fit_predict_method(
                method, choice["alpha"], fit_bundle, targets, validation_bundle
            )
            validation_indices_by_component_method[component][method] = predicted

    validation_by_method = {}
    for method in ALL_METHODS:
        predictions = semantic_predictions(
            validation_indices_by_component_method["lexical_sign"][method],
            validation_indices_by_component_method["outer_operation"][method],
            config,
        )
        validation_by_method[method] = score_semantics(validation_rows, predictions, v32_config)
    selected_predictions = semantic_predictions(
        validation_indices_by_component_method["lexical_sign"][selected["lexical_sign"]["method"]],
        validation_indices_by_component_method["outer_operation"][selected["outer_operation"]["method"]],
        config,
    )
    selected_metrics = score_semantics(validation_rows, selected_predictions, v32_config)

    frozen_v36 = implementation["frozen_v36_interface"]
    v36_lock_path = PROJECT_ROOT / frozen_v36["path"]
    if file_sha256(v36_lock_path) != frozen_v36["sha256"]:
        raise RuntimeError("Frozen V36 interface lock changed")
    v36_lock = json.loads(v36_lock_path.read_text())
    v36_parameters_path = PROJECT_ROOT / v36_lock["parameter_artifact"]
    if (
        v36_lock["parameter_artifact"] != frozen_v36["parameter_artifact"]
        or v36_lock["parameter_artifact_sha256"] != frozen_v36["parameter_artifact_sha256"]
        or file_sha256(v36_parameters_path) != frozen_v36["parameter_artifact_sha256"]
    ):
        raise RuntimeError("Frozen V36 parameters changed")
    v36_npz = np.load(v36_parameters_path)
    baseline_sign = predict_component(
        npz["direct_sign_hidden"][validation_indices], unpack_component(v36_npz, "lexical_sign")
    )
    baseline_operation = predict_component(
        npz["direct_operation_hidden"][validation_indices], unpack_component(v36_npz, "outer_operation")
    )
    baseline_predictions = semantic_predictions(baseline_sign, baseline_operation, config)
    baseline_metrics = score_semantics(validation_rows, baseline_predictions, v32_config)
    qualified = qualification(selected_metrics, baseline_metrics, config)

    output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = output_dir / "selected-predictions.jsonl"
    baseline_path = output_dir / "frozen-v36-predictions.jsonl"
    selected_path.write_text("".join(json.dumps({
        "id": row["id"], "prediction": prediction
    }, sort_keys=True, separators=(",", ":")) + "\n" for row, prediction in zip(validation_rows, selected_predictions, strict=True)))
    baseline_path.write_text("".join(json.dumps({
        "id": row["id"], "prediction": prediction
    }, sort_keys=True, separators=(",", ":")) + "\n" for row, prediction in zip(validation_rows, baseline_predictions, strict=True)))
    result = {
        "schema_version": 37,
        "experiment": config["experiment"],
        "features_lock": str(feature_lock_path.relative_to(PROJECT_ROOT)),
        "features_lock_sha256": file_sha256(feature_lock_path),
        "evaluation_number": 1,
        "component_selection": selected,
        "cv_reports": cv_reports,
        "selected_validation": selected_metrics,
        "validation_by_method": validation_by_method,
        "frozen_v36_baseline": baseline_metrics,
        "qualification": qualified,
        "decision": qualified["decision"],
        "selected_predictions": str(selected_path.relative_to(PROJECT_ROOT)),
        "selected_predictions_sha256": file_sha256(selected_path),
        "baseline_predictions": str(baseline_path.relative_to(PROJECT_ROOT)),
        "baseline_predictions_sha256": file_sha256(baseline_path),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "fit_records_used": len(fit_rows),
            "validation_records_scored": len(validation_rows),
            "validation_evaluations": 1,
            "cv_ridge_fits": 150,
            "final_ridge_fits": 6,
            "selection_on_validation": 0,
            "threshold_changes": 0,
            "model_forward_passes": 0,
            "v32_calibration_records_read": 0,
            "v32_evaluation_records_read": 0,
            "v28_runs": 0,
            "adapter_training_runs": 0,
        },
        "authorization": {
            "fresh_semantic_confirmation_preregistration": qualified["passed"],
            "fresh_semantic_confirmation_construction": False,
            "end_to_end_relational_suite": False,
            "v32_evaluation": False,
            "v28": False,
            "adapter_training": False,
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "result_sha256": file_sha256(result_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
