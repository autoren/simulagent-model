#!/usr/bin/env python3
"""Fit the locked V24 cross heads and run the four-way V22 integration once."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from audit_v22r2_grounding import read_jsonl_directory
from audit_v24_cross_encoder import read_pairs
from evaluate_v22r2_relational_grounding import (
    condition_modes,
    grounding_summary,
    integration_condition,
    load_npz,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT, predicted_epistemic_rows, validate_scene_prediction


def feature_lookup(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        str(identifier): arrays["pair_features"][index]
        for index, identifier in enumerate(arrays["pair_ids"].tolist())
    }


def fit_heads(
    pairs: Sequence[dict[str, Any]], arrays: dict[str, np.ndarray], config: dict[str, Any],
) -> tuple[LogisticRegression, OneVsRestClassifier, dict[str, Any]]:
    lookup = feature_lookup(arrays)
    fit = [row for row in pairs if row["split"] == config["heads"]["fitSplit"]]
    match_x = np.stack([lookup[row["id"]] for row in fit]).astype(np.float32)
    match_y = np.asarray([row["target"]["same_atom"] for row in fit], dtype=np.uint8)
    positive = [row for row in fit if row["target"]["same_atom"]]
    truth_x = np.stack([lookup[row["id"]] for row in positive]).astype(np.float32)
    truth_y = np.asarray([row["target"]["truth_label"] for row in positive])
    match_spec = config["heads"]["match"]
    truth_spec = config["heads"]["truth"]
    match_head = LogisticRegression(
        C=match_spec["C"], class_weight=match_spec["classWeight"],
        solver=match_spec["solver"], max_iter=match_spec["maximumIterations"],
        random_state=config["seed"],
    ).fit(match_x, match_y)
    truth_head = OneVsRestClassifier(LogisticRegression(
        C=truth_spec["C"], class_weight=truth_spec["classWeight"],
        solver=truth_spec["solver"], max_iter=truth_spec["maximumIterations"],
        random_state=config["seed"],
    )).fit(truth_x, truth_y)
    diagnostics = {
        "match_rows": len(match_y),
        "match_positive_rate": float(np.mean(match_y)),
        "match_class_counts": {
            str(key): value for key, value in sorted(Counter(match_y.tolist()).items())
        },
        "truth_rows": len(truth_y),
        "truth_class_counts": dict(sorted(Counter(truth_y.tolist()).items())),
        "match_iterations": match_head.n_iter_.tolist(),
        "truth_iterations_by_one_vs_rest_estimator": [
            estimator.n_iter_.tolist() for estimator in truth_head.estimators_
        ],
        "truth_multiclass_strategy": "explicit_one_vs_rest",
    }
    return match_head, truth_head, diagnostics


def predict_scenes(
    scenes: Sequence[dict[str, Any]], pairs: Sequence[dict[str, Any]],
    arrays: dict[str, np.ndarray], match_head: LogisticRegression,
    truth_head: OneVsRestClassifier,
) -> list[dict[str, Any]]:
    lookup = feature_lookup(arrays)
    proposals: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        proposals[(row["scene_id"], row["evidence_id"])].append(row)
    positive_column = int(np.flatnonzero(match_head.classes_ == 1)[0])
    predictions = []
    for scene in scenes:
        candidate_ids = [row["id"] for row in scene["agent_input"]["atom_candidates"]]
        evidence_ids = [row["id"] for row in scene["agent_input"]["evidence"]]
        candidate_index = {value: index for index, value in enumerate(candidate_ids)}
        scores = np.full((len(evidence_ids), len(candidate_ids)), -1e6, dtype=np.float32)
        pair_by_edge = {}
        for evidence_index, evidence_id in enumerate(evidence_ids):
            edges = proposals[(scene["id"], evidence_id)]
            x = np.stack([lookup[row["id"]] for row in edges]).astype(np.float32)
            probabilities = match_head.predict_proba(x)[:, positive_column]
            for row, probability in zip(edges, probabilities, strict=True):
                index = candidate_index[row["candidate_id"]]
                scores[evidence_index, index] = probability
                pair_by_edge[(evidence_id, row["candidate_id"])] = row
        evidence_indices, candidate_indices = linear_sum_assignment(-scores)
        assignment = dict(zip(evidence_indices.tolist(), candidate_indices.tolist(), strict=True))
        rows = []
        for evidence_index, evidence_id in enumerate(evidence_ids):
            candidate_id = candidate_ids[assignment[evidence_index]]
            if scores[evidence_index, assignment[evidence_index]] < 0:
                raise RuntimeError(f"Sparse V24 assignment used an unproposed edge in {scene['id']}")
            pair = pair_by_edge[(evidence_id, candidate_id)]
            truth = str(truth_head.predict(lookup[pair["id"]][None, :])[0])
            rows.append({
                "evidence_id": evidence_id,
                "candidate_id": candidate_id,
                "truth_label": truth,
                "assignment_score": float(scores[evidence_index, assignment[evidence_index]]),
                "pair_id": pair["id"],
            })
        validate_scene_prediction(scene, rows)
        predictions.append({
            "scene_id": scene["id"],
            "episode_id": scene["episode_id"],
            "split": scene["split"],
            "role": scene["role"],
            "rows": rows,
            "epistemic_state": predicted_epistemic_rows(scene, rows),
        })
    return predictions


def gate_checks(
    grounding: dict[str, Any], integration: dict[str, Any], gates: dict[str, float],
) -> dict[str, bool]:
    fit = grounding["by_split"]["grounding_fit"]
    evaluation = grounding["by_split"]["grounding_evaluation"]
    oracle = integration["oracle_support_oracle_query"]
    frozen_support = integration["frozen_support_oracle_query"]
    frozen_query = integration["oracle_support_frozen_query"]
    frozen_frozen = integration["frozen_support_frozen_query"]
    return {
        "oracle_oracle_exact": oracle["transition_set_exact_match"] >= gates["minimumOracleOracleExact"],
        "fit_atom_assignment": fit["atom_assignment_accuracy"] >= gates["minimumFitAtomAssignmentAccuracy"],
        "evaluation_atom_assignment": evaluation["atom_assignment_accuracy"] >= gates["minimumEvaluationAtomAssignmentAccuracy"],
        "evaluation_relation_order": evaluation["relation_argument_order_accuracy"] >= gates["minimumEvaluationRelationOrderAccuracy"],
        "evaluation_truth": evaluation["truth_status_accuracy"] >= gates["minimumEvaluationTruthAccuracy"],
        "evaluation_exact_scene": evaluation["exact_scene_graph"] >= gates["minimumEvaluationExactSceneGraph"],
        "frozen_support_oracle_query_exact": frozen_support["transition_set_exact_match"] >= gates["minimumFrozenSupportOracleQueryExact"],
        "oracle_support_frozen_query_exact": frozen_query["transition_set_exact_match"] >= gates["minimumOracleSupportFrozenQueryExact"],
        "frozen_frozen_exact": frozen_frozen["transition_set_exact_match"] >= gates["minimumFrozenFrozenExact"],
        "frozen_support_target_retention": frozen_support["target_retention_rate"] >= gates["minimumFrozenSupportTargetRetention"],
        "frozen_support_empty_version_space": frozen_support["empty_version_space_rate"] <= gates["maximumFrozenSupportEmptyVersionSpace"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v24-cross-encoder-lock.json")
    parser.add_argument("--features", default="outputs/v24-cross-encoder/features")
    parser.add_argument("--output-dir", default="outputs/v24-cross-encoder/evaluation")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    feature_root = (PROJECT_ROOT / args.features).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V24 evaluation was already attempted")
    lock = json.loads(lock_path.read_text())
    limits = lock["limits"]
    if not (
        limits["matchHeadFits"] == 1 and limits["truthHeadFits"] == 1
        and limits["integrationEvaluations"] == 1 and limits["hyperparameterSelections"] == 0
    ):
        raise RuntimeError("V24 lock does not authorize the registered one-shot evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V24 locked implementation changed: {path}")
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V24 features do not share the evaluation lock")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V24 feature artifact changed")
    arrays = load_npz(feature_path)
    proposal_root = PROJECT_ROOT / lock["source"]["proposal_corpus"]
    pairs = sorted(read_pairs(proposal_root), key=lambda row: row["id"])
    if arrays["pair_ids"].tolist() != [row["id"] for row in pairs]:
        raise RuntimeError("V24 feature and pair ordering differ")
    original_lock_path = PROJECT_ROOT / lock["source"]["v22r2_lock"]
    if file_sha256(original_lock_path) != lock["source"]["v22r2_lock_sha256"]:
        raise RuntimeError("V22r2 source lock changed after V24 lock")
    original_lock = json.loads(original_lock_path.read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    records = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "records")
    evaluation_records = [row for row in records if row["split"] == "grounding_evaluation"]
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())

    attempt_path.write_text(json.dumps({
        "schema_version": 24,
        "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_head_fitting",
    }, indent=2, sort_keys=True) + "\n")
    match_head, truth_head, fit_diagnostics = fit_heads(
        pairs, arrays, lock["config_payload"]
    )
    predictions = predict_scenes(scenes, pairs, arrays, match_head, truth_head)
    grounding = grounding_summary(scenes, predictions)
    prediction_lookup = {row["scene_id"]: row for row in predictions}
    integration = {}
    for condition in lock["integration_conditions"]:
        support_mode, query_mode = condition_modes(condition)
        integration[condition] = integration_condition(
            evaluation_records, support_mode, query_mode, prediction_lookup,
            v22_config, original_lock["config_payload"],
        )
    checks = gate_checks(grounding, integration, lock["gates"]["development"])
    passed = all(checks.values())
    evaluation = grounding["by_split"]["grounding_evaluation"]
    if passed:
        decision = "authorize_fresh_relational_surface_benchmark_design"
        interpretation = (
            "The candidate-conditioned frozen interface clears every exposed-data development gate. "
            "Freeze it before constructing a genuinely fresh relational surface benchmark."
        )
    elif not checks["evaluation_atom_assignment"] or not checks["evaluation_relation_order"]:
        decision = "candidate_conditioned_comparison_insufficient_no_lora"
        interpretation = (
            "Direct candidate conditioning does not sufficiently repair entity/relation matching. "
            "Do not construct a fresh benchmark or train weights from this result."
        )
    elif not checks["evaluation_truth"]:
        decision = "factor_truth_semantics_before_fresh_benchmark_no_lora"
        interpretation = (
            "Candidate identity transfers, but held-out truth semantics remain below the registered gate."
        )
    else:
        decision = "repair_symbolic_composition_before_fresh_benchmark_no_lora"
        interpretation = (
            "Grounding is adequate but exact scene assembly or downstream program induction remains limiting."
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    heads_path = output_dir / "heads.npz"
    np.savez_compressed(
        heads_path,
        match_classes=match_head.classes_,
        match_coef=match_head.coef_.astype(np.float32),
        match_intercept=match_head.intercept_.astype(np.float32),
        truth_classes=truth_head.classes_,
        truth_coef=np.stack([
            estimator.coef_[0] for estimator in truth_head.estimators_
        ]).astype(np.float32),
        truth_intercept=np.asarray([
            estimator.intercept_[0] for estimator in truth_head.estimators_
        ], dtype=np.float32),
    )
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in predictions
    ))
    result = {
        "schema_version": 24,
        "experiment": lock["experiment"],
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "evaluation_number": 1,
        "fit_diagnostics": fit_diagnostics,
        "grounding": grounding,
        "integration": integration,
        "checks": checks,
        "passed": passed,
        "decision": decision,
        "interpretation": interpretation,
        "evaluation_proposal_coverage": lock["pre_extraction_audit"]["proposal"]["gold_coverage_by_split_and_role"]["grounding_evaluation"],
        "heads_artifact": str(heads_path.relative_to(PROJECT_ROOT)),
        "heads_artifact_sha256": file_sha256(heads_path),
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "lora_authorized": False,
        "final_suite_constructed": False,
        "data_access": {
            "model_forward_passes": metadata["new_model_forward_passes"],
            "match_head_fits": 1,
            "truth_head_fits": 1,
            "integration_evaluations": 1,
            "hyperparameter_selections": 0,
            "adapter_training_runs": 0,
            "fresh_benchmark_records_read": 0,
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({
        "status": "completed",
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
    })
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
