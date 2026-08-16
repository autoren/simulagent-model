#!/usr/bin/env python3
"""Fit exactly four locked V36 readouts on V32 factor_fit only."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v35_binding import build_entity_examples
from v36_interface import fit_component, pack_component, parameter_shapes, projection_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-lock", default="configs/v36-implementation-lock.json")
    parser.add_argument("--output-dir", default="outputs/v36-independent-confirmation/interface")
    args = parser.parse_args()
    lock_path, output_dir = (PROJECT_ROOT / args.implementation_lock).resolve(), (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "interface-fit-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V36 interface fitting was already attempted")
    lock = json.loads(lock_path.read_text()); config = lock["config_payload"]
    if not lock["authorization"]["fit_interface"] or lock["authorization"]["construct_confirmation"]:
        raise RuntimeError("V36 implementation lock has invalid interface-fit authorization")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V36 locked implementation changed: {path}")
    source = lock["training_sources"]
    for name, metadata in source.items():
        if file_sha256(PROJECT_ROOT / metadata["path"]) != metadata["sha256"]:
            raise RuntimeError(f"V36 training source changed: {name}")
    rows = sorted(read_rows(PROJECT_ROOT / config["trainingCorpus"], (config["trainingSplit"],)), key=lambda row: row["id"])
    expected_ids = [row["id"] for row in rows]
    loaded = {}
    for name in ("v32_features", "v34_features", "v35_features"):
        metadata = json.loads((PROJECT_ROOT / source[name]["path"]).read_text())
        artifact = PROJECT_ROOT / metadata["feature_artifact"]
        if file_sha256(artifact) != source[name]["artifact_sha256"]:
            raise RuntimeError(f"V36 feature artifact changed: {name}")
        npz = np.load(artifact)
        ids = npz["record_ids"].tolist(); indices = np.asarray([ids.index(identifier) for identifier in expected_ids], dtype=np.int64)
        if len(set(indices.tolist())) != len(rows):
            raise RuntimeError(f"V36 fit population mismatch: {name}")
        loaded[name] = ({key: npz[key] for key in npz.files}, indices)
    output_dir.mkdir(parents=True, exist_ok=False)
    attempt_path.write_text(json.dumps({
        "schema_version": 36, "attempt_number": 1, "status": "started",
        "implementation_lock_sha256": file_sha256(lock_path), "fit_records": len(rows),
        "legacy_calibration_targets_read": 0, "legacy_evaluation_records_read": 0,
        "confirmation_records_read": 0, "selection_runs": 0,
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    predicates = lock["v32_config_payload"]["sharedHead"]["predicateClasses"]
    signs = lock["v32_config_payload"]["sharedHead"]["lexicalSignClasses"]
    operations = lock["v32_config_payload"]["sharedHead"]["outerOperationClasses"]
    targets = {
        "predicate": np.asarray([predicates.index(row["target"]["predicate"]) for row in rows], dtype=np.int64),
        "lexical_sign": np.asarray([signs.index(row["target"]["factorization"]["lexical_sign"]) for row in rows], dtype=np.int64),
        "outer_operation": np.asarray([operations.index(row["target"]["factorization"]["outer_operation"]) for row in rows], dtype=np.int64),
    }
    v32, i32 = loaded["v32_features"]; v34, i34 = loaded["v34_features"]; v35, i35 = loaded["v35_features"]
    features = {
        "predicate": v35["clause_features"][i35],
        "lexical_sign": v32["clause_features"][i32],
        "outer_operation": v34["semantic_hidden"][i34],
    }
    alphas = {
        "predicate": config["frozenInterface"]["predicate"]["alpha"],
        "binding": config["frozenInterface"]["binding"]["alpha"],
        "lexical_sign": config["frozenInterface"]["lexicalSign"]["alpha"],
        "outer_operation": config["frozenInterface"]["outerOperation"]["alpha"],
    }
    artifact_arrays: dict[str, np.ndarray] = {}
    ledgers = {}
    for component in ("predicate", "lexical_sign", "outer_operation"):
        model, parameters = fit_component(features[component], targets[component], alphas[component])
        predictions = model.predict(features[component])
        pack_component(artifact_arrays, component, parameters)
        ledgers[component] = {
            "alpha": float(alphas[component]), "records": len(rows),
            "training_accuracy": float(np.mean(predictions == targets[component])),
            "parameter_shapes": parameter_shapes(parameters),
        }
    entity_raw, entity_targets, entity_rows, entity_indices = build_entity_examples(rows, v35["evidence_entity_features"][i35])
    projection = config["frozenInterface"]["binding"]["projection"]
    matrix = projection_matrix(entity_raw.shape[1], projection["dimensions"], projection["seed"])
    projected = entity_raw @ matrix
    binding_model, binding_parameters = fit_component(projected, entity_targets, alphas["binding"])
    binding_predictions = binding_model.predict(projected)
    pack_component(artifact_arrays, "binding", binding_parameters)
    artifact_arrays["binding__projection"] = matrix
    ledgers["binding"] = {
        "alpha": float(alphas["binding"]), "entity_examples": len(entity_targets),
        "source_records": len(rows), "training_role_accuracy": float(np.mean(binding_predictions == entity_targets)),
        "projection_shape": list(matrix.shape), "projection_seed": projection["seed"],
        "parameter_shapes": parameter_shapes(binding_parameters),
    }
    artifact_path = output_dir / "parameters.npz"
    np.savez_compressed(artifact_path, **artifact_arrays)
    ledger = {
        "schema_version": 36, "experiment": "v36_fixed_interface_fit",
        "implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)), "implementation_lock_sha256": file_sha256(lock_path),
        "training_split": config["trainingSplit"], "fit_records": len(rows), "fit_record_ids_sha256": file_sha256(PROJECT_ROOT / config["trainingCorpus"] / f"{config['trainingSplit']}.jsonl"),
        "components": ledgers, "parameter_artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
        "parameter_artifact_sha256": file_sha256(artifact_path), "runtime_seconds": time.perf_counter() - started,
        "data_access": {"interface_fit_runs": 4, "selection_runs": 0, "fit_records_used": len(rows), "legacy_calibration_targets_read": 0, "legacy_evaluation_records_read": 0, "confirmation_records_read": 0, "model_forward_passes": 0},
    }
    ledger_path = output_dir / "fit-ledger.json"; ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text()); attempt.update({"status": "completed", "ledger_sha256": file_sha256(ledger_path)}); attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
