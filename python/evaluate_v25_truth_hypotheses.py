#!/usr/bin/env python3
"""Fit V25 truth compatibility once and compose it with fixed V24 assignments."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from audit_v22r2_grounding import read_jsonl_directory
from audit_v25_truth_hypotheses import read_rows
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
        str(identifier): arrays["truth_features"][index]
        for index, identifier in enumerate(arrays["row_ids"].tolist())
    }


def fit_head(
    rows: Sequence[dict[str, Any]], arrays: dict[str, np.ndarray], config: dict[str, Any],
) -> tuple[LogisticRegression, dict[str, Any]]:
    lookup = feature_lookup(arrays)
    fit = [row for row in rows if row["target"]["use_for_fit"]]
    x = np.stack([lookup[row["id"]] for row in fit]).astype(np.float32)
    y = np.asarray([row["target"]["compatible"] for row in fit], dtype=np.uint8)
    spec = config["head"]
    head = LogisticRegression(
        C=spec["C"], class_weight=spec["classWeight"], solver=spec["solver"],
        max_iter=spec["maximumIterations"], random_state=config["seed"],
    ).fit(x, y)
    return head, {
        "rows": len(y),
        "positive_rate": float(np.mean(y)),
        "class_counts": {str(key): value for key, value in sorted(Counter(y.tolist()).items())},
        "iterations": head.n_iter_.tolist(),
        "fit_split": config["head"]["fitSplit"],
        "calibration_use": config["head"]["calibrationUse"],
    }


def predict_scenes(
    scenes: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]],
    arrays: dict[str, np.ndarray], head: LogisticRegression,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    lookup = feature_lookup(arrays)
    hypotheses = {row["id"]: row for row in config["assessmentHypotheses"]}
    fixed: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "v24_fixed_assignment" in row["selection_sources"]:
            fixed[(row["scene_id"], row["evidence_id"])].append(row)
    positive_column = int(np.flatnonzero(head.classes_ == 1)[0])
    predictions = []
    for scene in scenes:
        prediction_rows = []
        for evidence in scene["agent_input"]["evidence"]:
            candidates = fixed[(scene["id"], evidence["id"])]
            if len(candidates) != len(hypotheses):
                raise RuntimeError(f"V25 fixed truth triple differs in {scene['id']}/{evidence['id']}")
            x = np.stack([lookup[row["id"]] for row in candidates]).astype(np.float32)
            probabilities = head.predict_proba(x)[:, positive_column]
            selected_index = int(np.argmax(probabilities))
            selected = candidates[selected_index]
            prediction_rows.append({
                "evidence_id": evidence["id"],
                "candidate_id": selected["candidate_id"],
                "truth_label": hypotheses[selected["assessment_id"]]["truthLabel"],
                "truth_compatibility_score": float(probabilities[selected_index]),
                "assessment_id": selected["assessment_id"],
                "truth_hypothesis_row_id": selected["id"],
            })
        validate_scene_prediction(scene, prediction_rows)
        predictions.append({
            "scene_id": scene["id"],
            "episode_id": scene["episode_id"],
            "split": scene["split"],
            "role": scene["role"],
            "rows": prediction_rows,
            "epistemic_state": predicted_epistemic_rows(scene, prediction_rows),
        })
    return predictions


def gate_checks(
    grounding: dict[str, Any], integration: dict[str, Any], gates: dict[str, float],
) -> dict[str, bool]:
    evaluation = grounding["by_split"]["grounding_evaluation"]
    oracle = integration["oracle_support_oracle_query"]
    frozen_support = integration["frozen_support_oracle_query"]
    frozen_query = integration["oracle_support_frozen_query"]
    frozen_frozen = integration["frozen_support_frozen_query"]
    return {
        "oracle_oracle_exact": oracle["transition_set_exact_match"] >= gates["minimumOracleOracleExact"],
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
    parser.add_argument("--lock", default="configs/v25-truth-hypotheses-lock.json")
    parser.add_argument("--features", default="outputs/v25-truth-hypotheses/features")
    parser.add_argument("--output-dir", default="outputs/v25-truth-hypotheses/evaluation")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    feature_root = (PROJECT_ROOT / args.features).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V25 evaluation was already attempted")
    lock = json.loads(lock_path.read_text())
    limits = lock["limits"]
    if not (
        limits["truthCompatibilityHeadFits"] == 1 and limits["matchHeadFits"] == 0
        and limits["integrationEvaluations"] == 1 and limits["hyperparameterSelections"] == 0
    ):
        raise RuntimeError("V25 lock does not authorize the registered one-shot evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V25 locked implementation changed: {path}")
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V25 features do not share the evaluation lock")
    artifact_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(artifact_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V25 feature artifact changed")
    arrays = load_npz(artifact_path)
    corpus_root = PROJECT_ROOT / lock["source"]["corpus"]
    rows = sorted(read_rows(corpus_root), key=lambda row: row["id"])
    if arrays["row_ids"].tolist() != [row["id"] for row in rows]:
        raise RuntimeError("V25 feature and row ordering differ")
    v24_lock_path = PROJECT_ROOT / lock["source"]["v24_lock"]
    v24_lock = json.loads(v24_lock_path.read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    records = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "records")
    evaluation_records = [row for row in records if row["split"] == "grounding_evaluation"]
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())

    attempt_path.write_text(json.dumps({
        "schema_version": 25,
        "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_truth_compatibility_head_fit",
    }, indent=2, sort_keys=True) + "\n")
    head, fit_diagnostics = fit_head(rows, arrays, lock["config_payload"])
    predictions = predict_scenes(scenes, rows, arrays, head, lock["config_payload"])
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
    if passed:
        decision = "authorize_fresh_relational_surface_benchmark_design"
        interpretation = (
            "The fixed V24 matcher plus explicit V25 truth hypotheses clears every exposed-data gate. "
            "Freeze the combined interface before constructing a fresh benchmark."
        )
    elif not checks["evaluation_truth"]:
        decision = "explicit_truth_hypotheses_insufficient_no_lora"
        interpretation = (
            "Explicit entailment, contradiction, and unresolved hypotheses do not repair held-out truth semantics."
        )
    else:
        decision = "repair_exact_graph_or_symbolic_composition_no_lora"
        interpretation = (
            "Truth semantics pass, but exact graph assembly or downstream schema induction remains below gate."
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    head_path = output_dir / "truth-compatibility-head.npz"
    np.savez_compressed(
        head_path,
        classes=head.classes_,
        coef=head.coef_.astype(np.float32),
        intercept=head.intercept_.astype(np.float32),
    )
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions
    ))
    source_result = json.loads((PROJECT_ROOT / lock["source"]["v24_result"]).read_text())
    result = {
        "schema_version": 25,
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
        "v24_fixed_assignment_sha256": source_result["grounding_predictions_sha256"],
        "truth_compatibility_head": str(head_path.relative_to(PROJECT_ROOT)),
        "truth_compatibility_head_sha256": file_sha256(head_path),
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "lora_authorized": False,
        "final_suite_constructed": False,
        "data_access": {
            "model_forward_passes": metadata["new_model_forward_passes"],
            "truth_compatibility_head_fits": 1,
            "match_head_fits": 0,
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
