#!/usr/bin/env python3
"""Run fit-only V35 readout selection and the modular calibration assembly."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v35_binding import (
    build_entity_examples, decode_bindings, fixed_gaussian_projection, make_ridge,
    qualification, score_assembly, select_binding_method, select_new_predicate_method,
    select_report,
)


def classification_cv(
    features: np.ndarray, targets: np.ndarray, fit_indices: np.ndarray,
    groups: np.ndarray, alphas: list[float],
) -> tuple[list[dict[str, Any]], Any, np.ndarray]:
    reports = []
    for alpha in alphas:
        folds = []
        for group in sorted(set(groups.tolist())):
            held = groups == group
            model = make_ridge(alpha); model.fit(features[fit_indices][~held], targets[fit_indices][~held])
            accuracy = float(np.mean(model.predict(features[fit_indices][held]) == targets[fit_indices][held]))
            folds.append({"held_out_surface_name": group, "records": int(np.sum(held)), "primary_accuracy": accuracy})
        reports.append({
            "alpha": float(alpha), "folds": folds,
            "mean_group_cv_primary_accuracy": float(np.mean([fold["primary_accuracy"] for fold in folds])),
            "minimum_group_cv_primary_accuracy": float(min(fold["primary_accuracy"] for fold in folds)),
        })
    selected = select_report(reports)
    model = make_ridge(selected["alpha"]); model.fit(features[fit_indices], targets[fit_indices])
    return reports, model, model.predict(features)


def binding_primary(
    rows: list[dict[str, Any]], indices: np.ndarray,
    decoded: list[tuple[int, int | None]],
) -> float:
    values = []
    for row_index, (argument1, argument2) in zip(indices, decoded, strict=True):
        row = rows[int(row_index)]
        if row["target"]["predicate_kind"] != "relation":
            continue
        entities = [entity["id"] for entity in row["agent_input"]["entities"]]
        values.append(
            entities[argument1] == row["target"]["arguments"][0]
            and argument2 is not None
            and entities[argument2] == row["target"]["arguments"][1]
        )
    return float(np.mean(values))


def binding_cv(
    name: str, rows: list[dict[str, Any]], entity_features: np.ndarray,
    predicate_targets: np.ndarray, fit_indices: np.ndarray, groups: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    raw, targets, example_rows, _ = build_entity_examples(rows, entity_features)
    projection = config["readouts"]["bindingProjection"]
    features = fixed_gaussian_projection(raw, projection["outputDimensions"], projection["seed"])
    reports = []
    for alpha in config["readouts"]["alphas"]:
        folds = []
        for group in sorted(set(groups.tolist())):
            train_rows = fit_indices[groups != group]; held_rows = fit_indices[groups == group]
            train_mask = np.isin(example_rows, train_rows)
            model = make_ridge(alpha); model.fit(features[train_mask], targets[train_mask])
            scores = model.decision_function(features)
            decoded = decode_bindings(rows, held_rows, predicate_targets[held_rows], scores, example_rows, config)
            accuracy = binding_primary(rows, held_rows, decoded)
            folds.append({"held_out_surface_name": group, "relation_records": int(sum(rows[int(index)]["target"]["predicate_kind"] == "relation" for index in held_rows)), "primary_accuracy": accuracy})
        reports.append({
            "alpha": float(alpha), "folds": folds,
            "mean_group_cv_primary_accuracy": float(np.mean([fold["primary_accuracy"] for fold in folds])),
            "minimum_group_cv_primary_accuracy": float(min(fold["primary_accuracy"] for fold in folds)),
        })
    selected = select_report(reports)
    train_mask = np.isin(example_rows, fit_indices)
    model = make_ridge(selected["alpha"]); model.fit(features[train_mask], targets[train_mask])
    return {"name": name, "cv_reports": reports, "selected_cv": selected}, model.decision_function(features), example_rows


def full_operation_predictions(
    total: int, fit_indices: np.ndarray, calibration_indices: np.ndarray,
    path, expected_ids: list[str],
) -> np.ndarray:
    saved = np.load(path)
    if saved["record_ids"].tolist() != expected_ids:
        raise RuntimeError("V35 V34 prediction population differs")
    values = np.empty(total, dtype=np.int64)
    values[fit_indices] = saved["semanticHiddenRidge_fit"]
    values[calibration_indices] = saved["semanticHiddenRidge_calibration"]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v35-binding-assembly-lock.json")
    parser.add_argument("--feature-metadata", default="outputs/v35-binding-assembly/features/metadata.json")
    parser.add_argument("--output", default="outputs/v35-binding-assembly/result.json")
    args = parser.parse_args()
    lock_path, metadata_path, output_path = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.lock, args.feature_metadata, args.output))
    attempt_path = output_path.parent / "run-attempt.json"
    if output_path.exists() or attempt_path.exists():
        raise RuntimeError("V35 readout run was already attempted")
    lock, metadata = json.loads(lock_path.read_text()), json.loads(metadata_path.read_text())
    config = {**lock["config_payload"], "v32_config": lock["v32_config_payload"]}
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V35 locked implementation changed: {path}")
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V35 feature metadata does not bind this lock")
    focused_path = PROJECT_ROOT / metadata["feature_artifact"]
    legacy_metadata = json.loads((PROJECT_ROOT / config["sourceV32FeatureMetadata"]).read_text())
    legacy_path = PROJECT_ROOT / legacy_metadata["feature_artifact"]
    if file_sha256(focused_path) != metadata["feature_artifact_sha256"] or file_sha256(legacy_path) != lock["source"]["v32_feature_artifact_sha256"]:
        raise RuntimeError("V35 feature artifact changed")
    rows = sorted(read_rows(PROJECT_ROOT / config["sourceCorpus"], tuple(config["allowedSplits"])), key=lambda row: row["id"])
    focused_npz, legacy_npz = np.load(focused_path), np.load(legacy_path)
    focused = {key: focused_npz[key] for key in focused_npz.files}; legacy = {key: legacy_npz[key] for key in legacy_npz.files}
    ids = [row["id"] for row in rows]
    if focused["record_ids"].tolist() != ids or legacy["record_ids"].tolist() != ids:
        raise RuntimeError("V35 feature populations differ from allowed records")
    fit_indices = np.flatnonzero(focused["splits"] == "factor_fit")
    calibration_indices = np.flatnonzero(focused["splits"] == "factor_calibration")
    groups = np.asarray([rows[int(index)]["oracle_metadata"]["surface_name"] for index in fit_indices])
    predicates = config["atomInterface"]["predicateClasses"]
    signs = config["v32_config"]["sharedHead"]["lexicalSignClasses"]
    predicate_targets = np.asarray([predicates.index(row["target"]["predicate"]) for row in rows], dtype=np.int64)
    sign_targets = np.asarray([signs.index(row["target"]["factorization"]["lexical_sign"]) for row in rows], dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({"schema_version": 35, "attempt_number": 1, "status": "started", "protocol_lock_sha256": file_sha256(lock_path), "calibration_selection_calls": 0, "v32_evaluation_records_read": 0}, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter(); alphas = config["readouts"]["alphas"]
    predicate_features = {
        "legacyHiddenRidge": legacy["clause_features"],
        "atomHiddenRidge": focused["clause_features"],
        "nativePredicateLogitRidge": focused["native_predicate_logits"],
    }
    predicate_methods, predicate_predictions = {}, {}
    for name, features in predicate_features.items():
        reports, _, predicted = classification_cv(features, predicate_targets, fit_indices, groups, alphas)
        predicate_methods[name] = {"cv_reports": reports, "selected_cv": select_report(reports)}
        predicate_predictions[name] = predicted
    selected_predicate = select_new_predicate_method(predicate_methods)
    sign_reports, _, sign_predictions = classification_cv(legacy["clause_features"], sign_targets, fit_indices, groups, alphas)
    sign_method = {"cv_reports": sign_reports, "selected_cv": select_report(sign_reports)}
    binding_methods, binding_scores, binding_example_rows = {}, {}, None
    for name, features in {
        "legacyEntityRoleRidge": legacy["entity_features"],
        "atomEvidenceEntityRoleRidge": focused["evidence_entity_features"],
    }.items():
        report, scores, example_rows = binding_cv(name, rows, features, predicate_targets, fit_indices, groups, config)
        binding_methods[name] = report; binding_scores[name] = scores
        if binding_example_rows is None:
            binding_example_rows = example_rows
        elif not np.array_equal(binding_example_rows, example_rows):
            raise RuntimeError("V35 binding example populations differ")
    selected_binding = select_binding_method(binding_methods)
    operation_predictions = full_operation_predictions(
        len(rows), fit_indices, calibration_indices, PROJECT_ROOT / config["sourceV34Predictions"], ids
    )
    system_specs = {
        "legacyAssembly": ("legacyHiddenRidge", "legacyEntityRoleRidge"),
        "modularAssembly": (selected_predicate, selected_binding),
    }
    systems, saved_predictions = {}, {}
    for system, (predicate_method, binding_method) in system_specs.items():
        predicted_predicates = predicate_predictions[predicate_method]
        bindings = decode_bindings(rows, np.arange(len(rows)), predicted_predicates, binding_scores[binding_method], binding_example_rows, config)
        argument1 = np.asarray([value[0] for value in bindings], dtype=np.int64)
        argument2 = np.asarray([-1 if value[1] is None else value[1] for value in bindings], dtype=np.int64)
        saved_predictions[system] = {"predicate": predicted_predicates, "argument1": argument1, "argument2": argument2, "lexical_sign": sign_predictions, "outer_operation": operation_predictions}
        systems[system] = {
            "components": {"predicate": predicate_method, "binding": binding_method, "lexical_sign": "legacyHiddenRidge", "outer_operation": "fixedV34SemanticHiddenRidge"},
            "fit": score_assembly(rows, fit_indices, predicted_predicates[fit_indices], [bindings[int(index)] for index in fit_indices], sign_predictions[fit_indices], operation_predictions[fit_indices], config),
            "calibration": score_assembly(rows, calibration_indices, predicted_predicates[calibration_indices], [bindings[int(index)] for index in calibration_indices], sign_predictions[calibration_indices], operation_predictions[calibration_indices], config),
        }
    selected_qualification = qualification(systems["modularAssembly"]["calibration"], systems["legacyAssembly"]["calibration"], config)
    if selected_qualification["passed"]:
        decision = "development_interface_qualified_preregister_independent_confirmation"
    elif not selected_qualification["checks"]["predicate"] or not selected_qualification["checks"]["atom"] or not selected_qualification["checks"]["relation_order"]:
        decision = "atom_interface_incomplete_continue_local_repair"
    else:
        decision = "modular_assembly_insufficient_revisit_interface"
    predictions_path = output_path.parent / "predictions.npz"
    np.savez_compressed(predictions_path, record_ids=np.asarray(ids), **{f"{system}_{field}": values for system, fields in saved_predictions.items() for field, values in fields.items()})
    result = {
        "schema_version": 35, "experiment": config["experiment"], "protocol_lock_sha256": file_sha256(lock_path),
        "feature_metadata": str(metadata_path.relative_to(PROJECT_ROOT)), "feature_metadata_sha256": file_sha256(metadata_path),
        "predicate_methods": predicate_methods, "binding_methods": binding_methods, "lexical_sign_method": sign_method,
        "fit_selected_predicate_method": selected_predicate, "fit_selected_binding_method": selected_binding,
        "systems": systems, "qualification": selected_qualification, "decision": decision,
        "predictions": str(predictions_path.relative_to(PROJECT_ROOT)), "predictions_sha256": file_sha256(predictions_path),
        "runtime_seconds": time.perf_counter() - started,
        "authorization": {"independent_confirmation_preregistration": selected_qualification["passed"], "independent_confirmation_construction": False, "fresh_final_suite": False, "v32_evaluation_reuse": False, "v28_replay": False},
        "data_access": {"fit_records_read": len(fit_indices), "calibration_records_read": len(calibration_indices), "ridge_training_fits": 150, "backbone_forward_passes": metadata["data_access"]["backbone_forward_passes"], "calibration_selections": 0, "v32_evaluation_records_read": 0, "v32_evaluation_features_read": 0, "v32_evaluation_predictions_read": 0, "adapter_training_runs": 0, "v28_integration_replays": 0, "fresh_suite_constructions": 0},
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text()); attempt.update({"status": "completed", "result_sha256": file_sha256(output_path)}); attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fit_selected_predicate_method": selected_predicate, "fit_selected_binding_method": selected_binding, "systems": systems, "qualification": selected_qualification, "decision": decision}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
