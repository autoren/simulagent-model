#!/usr/bin/env python3
"""Perform the one authorized V36 confirmation evaluation."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v35_binding import build_entity_examples, decode_bindings
from v36_evaluation import decision_from_checks, gate_checks, score_confirmation
from v36_interface import decision_function, predict_component, unpack_component


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-lock", default="configs/v36-features-lock.json")
    parser.add_argument("--output-dir", default="outputs/v36-independent-confirmation/evaluation")
    args = parser.parse_args()
    feature_lock_path, output_dir = (PROJECT_ROOT / args.features_lock).resolve(), (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V36 confirmation evaluation was already attempted")
    feature_lock = json.loads(feature_lock_path.read_text())
    if feature_lock["authorization"]["confirmation_evaluations"] != 1 or feature_lock["authorization"]["selection_runs"] != 0 or feature_lock["authorization"]["threshold_changes"] != 0:
        raise RuntimeError("V36 feature lock does not authorize the fixed evaluation")
    seal_path = PROJECT_ROOT / feature_lock["confirmation_seal"]
    seal = json.loads(seal_path.read_text())
    interface_path = PROJECT_ROOT / seal["interface_lock"]
    interface = json.loads(interface_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    config = implementation["config_payload"]; v32_config = implementation["v32_config_payload"]
    runtime_config = {**config, "v32_config": v32_config, "atomInterface": {"predicateClasses": v32_config["sharedHead"]["predicateClasses"]}}
    corpus_path = PROJECT_ROOT / seal["corpus_artifact"]
    feature_path = PROJECT_ROOT / feature_lock["feature_artifact"]
    parameter_path = PROJECT_ROOT / interface["parameter_artifact"]
    for path, expected in ((corpus_path, seal["corpus_artifact_sha256"]), (feature_path, feature_lock["feature_artifact_sha256"]), (parameter_path, interface["parameter_artifact_sha256"])):
        if file_sha256(path) != expected:
            raise RuntimeError(f"V36 frozen artifact changed: {path}")
    rows = sorted(jsonl(corpus_path), key=lambda row: row["id"])
    feature_npz, parameter_npz = np.load(feature_path), np.load(parameter_path)
    features = {key: feature_npz[key] for key in feature_npz.files}
    if features["record_ids"].tolist() != [row["id"] for row in rows]:
        raise RuntimeError("V36 feature/corpus population mismatch")
    parameters = {name: unpack_component(parameter_npz, name) for name in ("predicate", "binding", "lexical_sign", "outer_operation")}
    started = time.perf_counter()
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({"schema_version": 36, "attempt_number": 1, "status": "started", "features_lock_sha256": file_sha256(feature_lock_path), "selection_runs": 0, "threshold_changes": 0}, indent=2, sort_keys=True) + "\n")
    predicate_indices = predict_component(features["atom_hidden"], parameters["predicate"])
    sign_indices = predict_component(features["generic_hidden"], parameters["lexical_sign"])
    operation_indices = predict_component(features["operation_hidden"], parameters["outer_operation"])
    entity_raw, _, example_rows, _ = build_entity_examples(rows, features["atom_entity_features"])
    projected = entity_raw @ np.asarray(parameter_npz["binding__projection"], dtype=np.float32)
    role_scores = decision_function(projected, parameters["binding"])
    bindings = decode_bindings(rows, np.arange(len(rows)), predicate_indices, role_scores, example_rows, runtime_config)
    predicates = v32_config["sharedHead"]["predicateClasses"]
    signs = v32_config["sharedHead"]["lexicalSignClasses"]
    operations = v32_config["sharedHead"]["outerOperationClasses"]
    predictions = []
    for row, predicate_index, binding, sign_index, operation_index in zip(rows, predicate_indices, bindings, sign_indices, operation_indices, strict=True):
        entities = [entity["id"] for entity in row["agent_input"]["entities"]]
        first, second = binding
        predictions.append({
            "id": row["id"], "scene_id": row["scene_id"], "split": row["split"],
            "selected_fields": {"predicate": predicates[int(predicate_index)], "argument_1": entities[first], "argument_2": "N/A" if second is None else entities[second]},
            "selected_intermediates": {"lexical_sign": signs[int(sign_index)], "outer_operation": operations[int(operation_index)]},
        })
    metrics = score_confirmation(rows, predictions, v32_config, config["execution"]["bootstrapSeed"], config["execution"]["bootstrapReplicates"])
    checks = gate_checks(metrics, config); decision, magnitude = decision_from_checks(metrics, checks)
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions))
    v35_result = json.loads((PROJECT_ROOT / config["sourceV35Result"]).read_text())
    v35_exact = v35_result["systems"]["modularAssembly"]["calibration"]["compiled_exact_fact_accuracy"]
    result = {
        "schema_version": 36, "experiment": config["experiment"], "evaluation_number": 1,
        "features_lock": str(feature_lock_path.relative_to(PROJECT_ROOT)), "features_lock_sha256": file_sha256(feature_lock_path),
        "metrics": metrics, "gate_checks": checks, "passed": all(checks.values()), "decision": decision,
        "confirmation_magnitude": magnitude, "exact_fact_change_from_v35_development": metrics["compiled_exact_fact_accuracy"] - v35_exact,
        "predictions": str(predictions_path.relative_to(PROJECT_ROOT)), "predictions_sha256": file_sha256(predictions_path),
        "runtime_seconds": time.perf_counter() - started,
        "authorization": {"preregister_end_to_end_relational_suite": all(checks.values()), "construct_end_to_end_relational_suite": False, "reuse_v32_evaluation": False, "run_v28": False, "adapter_training": False, "change_backbone": False},
        "data_access": {"confirmation_evaluations": 1, "confirmation_records_scored": len(rows), "selection_runs": 0, "threshold_changes": 0, "model_forward_passes": 0, "interface_fit_runs": 0, "v32_evaluation_records_read": 0, "v28_integration_replays": 0, "adapter_training_runs": 0},
    }
    result_path = output_dir / "result.json"; result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text()); attempt.update({"status": "completed", "result_sha256": file_sha256(result_path)}); attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "decision": decision, "confirmation_magnitude": magnitude, "metrics": metrics, "gate_checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
