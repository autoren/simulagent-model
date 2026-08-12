#!/usr/bin/env python3
"""Evaluate the locked V14 operator-supported 4B hypothesis-mean baseline."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from evaluate_v12_joint_readout import metrics, primary_probe, save_pipeline
from extract_v14_4b_token_mean_mlx import build_unique_pairs
from v10_protocol import file_sha256
from v14_protocol import load_records_from_manifest, primary_folds, zero_shot_operator_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v14-4b-baseline-lock.json")
    parser.add_argument("--features", default="outputs/v14-4b-baseline/features")
    parser.add_argument("--output-dir", default="outputs/v14-4b-baseline/evaluation")
    return parser.parse_args()


def pair_memberships(records: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], list[list[int]]]:
    from extract_v10_features_mlx import nli_text

    pairs: dict[tuple[str, str], int] = {}
    memberships: list[list[int]] = []
    for record_index, record in enumerate(records):
        hypotheses = {value["determinant_id"]: value["statements"] for value in record["agent_input"]["state_hypotheses"]}
        for determinant_index, target in enumerate(record["target"]["determinant_grounding"]):
            if target["temporal_status"] != "CURRENT":
                continue
            evidence_index = next(index for index, unit in enumerate(record["evidence_units"]) if (
                unit["start"] == target["evidence_span"]["start"]
                and unit["end"] == target["evidence_span"]["end"]
            ))
            pair = tuple(
                nli_text(record, determinant_index, evidence_index, hypothesis)
                for hypothesis in hypotheses[target["determinant_id"]]
            )
            if pair not in pairs:
                pairs[pair] = len(pairs)
                memberships.append([])
            memberships[pairs[pair]].append(record_index)
    return list(pairs), memberships


def membership_mask(fold_mask: np.ndarray, memberships: list[list[int]]) -> np.ndarray:
    return np.asarray([any(fold_mask[index] for index in values) for values in memberships])


def evaluate_fold(
    fold: dict[str, Any], fold_index: int, records: list[dict[str, Any]], memberships: list[list[int]],
    targets: np.ndarray, features: np.ndarray, swapped: np.ndarray, head: dict[str, Any], seed: int,
    output_dir: Path, prefix: str,
) -> dict[str, Any]:
    training = membership_mask(fold["train"], memberships)
    evaluation = membership_mask(fold["evaluation"], memberships)
    overlap = int(np.sum(training & evaluation))
    if fold["name"] != "context" and overlap:
        raise RuntimeError(f"V14 transfer fold has exact local-pair overlap: {fold['name']}")
    model = primary_probe(head, seed + fold_index)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features[training], targets[training])
    scores = model.predict_proba(features)[:, 1].astype(np.float32)
    swapped_scores = model.predict_proba(swapped)[:, 1].astype(np.float32)
    artifact = output_dir / f"{prefix}-{fold['name'].replace(':', '-')}-head.npz"
    save_pipeline(artifact, "signed_difference_linear", model)
    by_surface = {}
    for surface in sorted({record["state_lexicon_family"] for record in records}):
        mask = np.asarray([
            any(fold["evaluation"][index] and records[index]["state_lexicon_family"] == surface for index in values)
            for values in memberships
        ])
        if mask.any():
            by_surface[surface] = metrics(targets[mask], scores[mask], swapped_scores[mask])
    return {
        "kind": fold["kind"],
        "training_unique_pairs": int(training.sum()),
        "evaluation_unique_pairs": int(evaluation.sum()),
        "exact_pair_overlap": overlap,
        "convergence_warnings": sum(issubclass(item.category, ConvergenceWarning) for item in caught),
        "head_artifact": str(artifact),
        "head_artifact_sha256": file_sha256(artifact),
        "overall": metrics(targets[evaluation], scores[evaluation], swapped_scores[evaluation]),
        "by_surface": by_surface,
    }


def transfer_gate_report(folds: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    transfer = [value for name, value in folds.items() if name != "context"]
    surfaces = [cell for value in transfer for cell in value["by_surface"].values()]
    checks = [
        {
            "name": "minimum_transfer_fold_accuracy",
            "value": float(min(value["overall"]["accuracy"] for value in transfer)),
            "minimum": gates["minimumEveryTransferFoldAccuracy"],
        },
        {
            "name": "minimum_transfer_surface_accuracy",
            "value": float(min(value["accuracy"] for value in surfaces)),
            "minimum": gates["minimumEveryTransferSurfaceAccuracy"],
        },
    ]
    for check in checks:
        check["passed"] = check["value"] >= check["minimum"]
    return {"passed": all(value["passed"] for value in checks), "checks": checks}


def decision(report: dict[str, Any]) -> str:
    return (
        "operator_supported_surface_transfer_passes_repair_temporal_then_full_pipeline"
        if report["passed"]
        else "operator_supported_surface_transfer_fails_audit_before_adaptation"
    )


def main() -> None:
    args = parse_args()
    lock_path = Path(args.lock)
    feature_root = Path(args.features)
    output_dir = Path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError(f"V14 baseline result already exists: {result_path}")
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(Path(path)) != expected:
            raise RuntimeError(f"V14 locked model implementation changed: {path}")
    records = load_records_from_manifest(Path(lock["source"]["manifest"]))
    pairs, memberships = pair_memberships(records)
    metadata = json.loads((feature_root / "metadata.json").read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V14 features do not share the model lock")
    feature_path = Path(metadata["feature_artifact"])
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V14 feature artifact changed")
    with np.load(feature_path, allow_pickle=False) as values:
        saved_pairs = [tuple(row) for row in values["pair_prompts"].tolist()]
        targets = values["pair_targets"].astype(np.int8)
        hypothesis = values["hypothesis_mean_features"].astype(np.float32)
    if saved_pairs != pairs or len(memberships) != 756:
        raise RuntimeError("V14 evaluation pair order differs from extraction")
    if hypothesis.shape != (1512, lock["model"]["hidden_size"]):
        raise RuntimeError("V14 hypothesis feature shape differs from lock")
    active = hypothesis[0::2]
    inactive = hypothesis[1::2]
    features = active - inactive
    swapped = inactive - active
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_results = {}
    all_primary = primary_folds(records)
    for fold_index, fold in enumerate(all_primary):
        primary_results[fold["name"]] = evaluate_fold(
            fold, fold_index, records, memberships, targets, features, swapped,
            lock["head"], lock["seed"], output_dir, "primary",
        )
    diagnostic_results = {}
    for diagnostic_index, fold in enumerate(zero_shot_operator_folds(records), start=len(all_primary)):
        diagnostic_results[fold["name"]] = evaluate_fold(
            fold, diagnostic_index, records, memberships, targets, features, swapped,
            lock["head"], lock["seed"], output_dir, "diagnostic",
        )
    gates = transfer_gate_report(primary_results, lock["gates"])
    result = {
        "schema_version": 14,
        "experiment": "v14_operator_supported_4b_token_mean_baseline",
        "protocol_lock": str(lock_path),
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "record_weighted_examples": 11070,
        "unique_local_pairs": 756,
        "primary_folds": primary_results,
        "zero_shot_operator_diagnostics": diagnostic_results,
        "primary_transfer_gates": gates,
        "context_gating": False,
        "decision": decision(gates),
        "lora_authorized": False,
        "final_mechanic_authorized": False,
        "data_access": lock["data_access"],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "primary_transfer_gates": gates,
        "context": primary_results["context"]["overall"],
        "zero_shot_operator_diagnostics": {
            name: value["overall"] for name, value in diagnostic_results.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
