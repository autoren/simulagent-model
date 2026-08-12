#!/usr/bin/env python3
"""Execute the locked V22r2a one-vs-rest compatibility amendment once."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from audit_v22r2_grounding import read_jsonl_directory
from evaluate_v22r2_relational_grounding import (
    build_training_arrays,
    condition_modes,
    feature_lookup,
    grounding_summary,
    integration_condition,
    load_npz,
    predict_scenes,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def fit_amended_heads(scenes, arrays, config):
    candidates, evidence = feature_lookup(arrays)
    pair_x, pair_y, truth_x, truth_y = build_training_arrays(
        scenes, candidates, evidence, config
    )
    atom_spec = config["heads"]["atomMatching"]
    truth_spec = config["heads"]["truthStatus"]
    atom_head = LogisticRegression(
        C=atom_spec["C"], class_weight=atom_spec["classWeight"],
        solver=atom_spec["solver"], max_iter=atom_spec["maximumIterations"],
        random_state=config["seed"],
    ).fit(pair_x, pair_y)
    truth_head = OneVsRestClassifier(LogisticRegression(
        C=truth_spec["C"], class_weight=truth_spec["classWeight"],
        solver=truth_spec["solver"], max_iter=truth_spec["maximumIterations"],
        random_state=config["seed"],
    )).fit(truth_x, truth_y)
    diagnostics = {
        "atom_matching_rows": len(pair_y),
        "atom_matching_positive_rate": float(np.mean(pair_y)),
        "truth_rows": len(truth_y),
        "truth_class_counts": dict(sorted(Counter(truth_y.tolist()).items())),
        "atom_matching_iterations": atom_head.n_iter_.tolist(),
        "truth_iterations_by_one_vs_rest_estimator": [
            estimator.n_iter_.tolist() for estimator in truth_head.estimators_
        ],
        "truth_multiclass_strategy": "explicit_one_vs_rest",
    }
    return atom_head, truth_head, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v22r2a-evaluation-amendment-lock.json")
    parser.add_argument("--features", default="outputs/v22r2-relational-grounding/features")
    parser.add_argument("--output-dir", default="outputs/v22r2-relational-grounding/evaluation-v22r2a")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    feature_root = (PROJECT_ROOT / args.features).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-v22r2a-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V22r2a replacement evaluation was already attempted")
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V22r2a locked implementation changed: {path}")
    limits = lock["replacement_limits"]
    if limits["evaluationAttempts"] != 1 or limits["newModelForwardPasses"] != 0:
        raise RuntimeError("V22r2a replacement limits do not authorize execution")
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if file_sha256(metadata_path) != lock["source"]["feature_metadata_sha256"]:
        raise RuntimeError("V22r2a feature metadata changed")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != lock["source"]["feature_artifact_sha256"]:
        raise RuntimeError("V22r2a feature artifact changed")
    original_lock = json.loads((PROJECT_ROOT / lock["source"]["original_lock"]).read_text())
    config = original_lock["config_payload"]
    arrays = load_npz(feature_path)
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    records = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "records")
    evaluation_records = [row for row in records if row["split"] == "grounding_evaluation"]
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())
    if arrays["scene_ids"].tolist() != [row["id"] for row in scenes]:
        raise RuntimeError("V22r2a feature and scene ordering differ")

    attempt_path.write_text(json.dumps({
        "schema_version": "22r2a", "attempt_number": 1,
        "amendment_lock_sha256": file_sha256(lock_path),
        "status": "started_before_replacement_head_fitting",
    }, indent=2, sort_keys=True) + "\n")
    atom_head, truth_head, fit_diagnostics = fit_amended_heads(scenes, arrays, config)
    predictions = predict_scenes(scenes, arrays, atom_head, truth_head)
    grounding = grounding_summary(scenes, predictions)
    prediction_lookup = {row["scene_id"]: row for row in predictions}
    integration = {}
    for condition in original_lock["integration_conditions"]:
        support_mode, query_mode = condition_modes(condition)
        integration[condition] = integration_condition(
            evaluation_records, support_mode, query_mode, prediction_lookup,
            v22_config, config,
        )

    gates = original_lock["gates"]["development"]
    fit = grounding["by_split"]["grounding_fit"]
    evaluation = grounding["by_split"]["grounding_evaluation"]
    oracle = integration["oracle_support_oracle_query"]
    primary = integration["frozen_support_frozen_query"]
    frozen_support = integration["frozen_support_oracle_query"]
    checks = {
        "oracle_oracle_transition_set_exact": oracle["transition_set_exact_match"] >= gates["minimumOracleOracleTransitionSetExact"],
        "fit_atom_assignment": fit["atom_assignment_accuracy"] >= gates["minimumFitAtomAssignmentAccuracy"],
        "evaluation_atom_assignment": evaluation["atom_assignment_accuracy"] >= gates["minimumEvaluationAtomAssignmentAccuracy"],
        "evaluation_truth_status": evaluation["truth_status_accuracy"] >= gates["minimumEvaluationTruthStatusAccuracy"],
        "evaluation_relation_orientation": evaluation["relation_argument_order_accuracy"] >= gates["minimumEvaluationRelationOrientationAccuracy"],
        "evaluation_exact_scene_graph": evaluation["exact_scene_graph"] >= gates["minimumEvaluationExactSceneGraph"],
        "frozen_frozen_transition_set_exact": primary["transition_set_exact_match"] >= gates["minimumFrozenFrozenTransitionSetExact"],
        "frozen_support_target_retention": frozen_support["target_retention_rate"] >= gates["minimumFrozenSupportTargetRetention"],
        "frozen_support_empty_version_space": frozen_support["empty_version_space_rate"] <= gates["maximumFrozenSupportEmptyVersionSpaceRate"],
    }
    passed = all(checks.values())
    query_only = integration["oracle_support_frozen_query"]
    if passed:
        decision = "authorize_separate_relational_final_design"
        interpretation = "The fixed hard grounder passes held-out surfaces and four-way relational integration within the declared ontology."
    elif frozen_support["transition_set_exact_match"] < query_only["transition_set_exact_match"]:
        decision = "develop_probabilistic_support_interface_no_lora"
        interpretation = "Support grounding is the larger downstream bottleneck; preserve multiple support groundings in a separately registered development experiment."
    else:
        decision = "repair_relational_language_grounding_no_lora"
        interpretation = "Held-out language grounding or query graph assembly is the larger bottleneck; repair the interface before final evaluation or weight adaptation."

    output_dir.mkdir(parents=True, exist_ok=False)
    heads_path = output_dir / "heads.npz"
    np.savez_compressed(
        heads_path,
        atom_classes=atom_head.classes_, atom_coef=atom_head.coef_.astype(np.float32),
        atom_intercept=atom_head.intercept_.astype(np.float32),
        truth_classes=truth_head.classes_,
        truth_coef=np.stack([estimator.coef_[0] for estimator in truth_head.estimators_]).astype(np.float32),
        truth_intercept=np.asarray([estimator.intercept_[0] for estimator in truth_head.estimators_], dtype=np.float32),
    )
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions
    ))
    result = {
        "schema_version": "22r2a", "experiment": lock["experiment"],
        "amendment_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "amendment_lock_sha256": file_sha256(lock_path),
        "original_protocol_lock_sha256": lock["source"]["original_lock_sha256"],
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "replacement_evaluation_number": 1,
        "aborted_attempts_before_predictions": 1,
        "fit_diagnostics": fit_diagnostics, "grounding": grounding,
        "integration": integration, "checks": checks, "passed": passed,
        "decision": decision, "interpretation": interpretation,
        "heads_artifact": str(heads_path.relative_to(PROJECT_ROOT)),
        "heads_artifact_sha256": file_sha256(heads_path),
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "lora_authorized": False,
        "final_suite_constructed": False,
        "data_access": {
            "replacement_atom_matching_head_fits": 1,
            "truth_status_head_fits": 1, "new_model_forward_passes": 0,
            "hyperparameter_selections": 0, "adapter_training_runs": 0,
            "evaluation_metrics_seen_before_amendment": 0,
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({
        "status": "completed", "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
    })
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
