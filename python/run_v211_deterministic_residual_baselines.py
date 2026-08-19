#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v211_deterministic_residual_baselines import audit_scores, score_predictions
from v22r2_grounding import PROJECT_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v211-deterministic-residual-baselines-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock):
        raise RuntimeError("invalid V211 design lock")
    dependency_keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V211 locked dependency changed: {key}")
    config = lock["config_payload"]
    artifacts = {key: PROJECT_ROOT / value for key, value in config["artifacts"].items()}
    if any(path.exists() for path in artifacts.values()):
        raise RuntimeError("V211 output artifact already exists")
    config_path = PROJECT_ROOT / lock["config"]
    firewall = PROJECT_ROOT / lock["firewall_worker"]
    predictor = PROJECT_ROOT / lock["prediction_worker"]

    subprocess.run(
        [
            sys.executable, str(firewall), "--config", str(config_path),
            "--surface", str(PROJECT_ROOT / config["population"]["inputDevelopmentSurface"]),
            "--truth", str(PROJECT_ROOT / config["population"]["inputDevelopmentTruth"]),
            "--projection", str(PROJECT_ROOT / config["population"]["inputDevelopmentProjection"]),
            "--split", str(artifacts["split"]),
            "--calibration-surface", str(artifacts["calibrationSurface"]),
            "--calibration-truth", str(artifacts["calibrationTruth"]),
            "--evaluation-surface", str(artifacts["evaluationSurface"]),
            "--evaluation-truth", str(artifacts["evaluationTruthSealed"]),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    subprocess.run(
        [
            sys.executable, str(predictor), "--config", str(config_path),
            "--calibration-surface", str(artifacts["calibrationSurface"]),
            "--calibration-truth", str(artifacts["calibrationTruth"]),
            "--evaluation-surface", str(artifacts["evaluationSurface"]),
            "--learned", str(artifacts["learnedLexicon"]),
            "--predictions", str(artifacts["predictions"]),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    prediction_freeze = {
        "schema_version": "211-prediction-freeze",
        "predictions": str(artifacts["predictions"].relative_to(PROJECT_ROOT)),
        "predictions_sha256": file_sha256(artifacts["predictions"]),
        "evaluation_truth_opened_before_freeze": False,
        "prediction_worker_evaluation_truth_path_count": 0,
        "prediction_worker_group_id_read_count": 0,
    }
    write_json(artifacts["predictionFreeze"], prediction_freeze)

    split = json.loads(artifacts["split"].read_text())
    learned = json.loads(artifacts["learnedLexicon"].read_text())
    predictions = read_jsonl(artifacts["predictions"])
    evaluation_truth = read_jsonl(artifacts["evaluationTruthSealed"])
    parent_outcome = json.loads((PROJECT_ROOT / lock["reference_V209r1_outcome"]).read_text())
    repair_lock = json.loads((PROJECT_ROOT / parent_outcome["repair_lock"]).read_text())
    v209_lock = json.loads((PROJECT_ROOT / repair_lock["parent_V209_design_lock"]).read_text())
    parent_config = v209_lock["config_payload"]
    scores = score_predictions(predictions, evaluation_truth, parent_config)
    audit = audit_scores(split, learned, predictions, scores, config)
    access = {
        "deterministic_evaluation_count": 1,
        "evaluation_truth_read_during_fit_or_prediction_count": 0,
        "prediction_worker_evaluation_truth_path_count": 0,
        "prediction_worker_group_id_read_count": 0,
        "protected_surface_or_truth_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "trusted_state_mutation_count": 0,
        "service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
        "fallback_count": 0,
    }
    summary = {
        "split": split,
        "learned_lexicon_sizes": {name: len(value["token_to_label"]) for name, value in learned.items()},
        "prediction_freeze": prediction_freeze,
        "scores": scores,
        "access": access,
        "model_eligible": audit["model_eligible"],
        "branch": audit["branch"],
    }
    result = {
        "schema_version": "211-deterministic-residual-baselines-result",
        "experiment": config["experiment"],
        "passed": audit["passed"],
        "branch": audit["branch"],
        "model_eligible": audit["model_eligible"],
        "decision": audit["decision"],
        "checks": audit["checks"],
        "access_checks": audit["access_checks"],
        "summary": summary,
        "authorization": {
            "separate_local_model_design_only": audit["branch"] == "NONTRIVIAL_MODEL_ELIGIBLE_RESIDUAL",
            "design_new_identifiable_open_class_population": audit["branch"] == "ZERO_MODEL_ELIGIBILITY",
            "open_protected_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    write_json(artifacts["summary"], summary)
    write_json(artifacts["result"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
