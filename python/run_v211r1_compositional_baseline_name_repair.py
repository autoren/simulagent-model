#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from v10_protocol import file_sha256
from v211r1_compositional_baseline_name_repair import audit_scores, repair_diagnostics, score_predictions
from v22r2_grounding import PROJECT_ROOT


OUTPUT = PROJECT_ROOT / "outputs/v211r1-compositional-baseline-name-repair/evaluation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v211r1-compositional-baseline-name-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    if not valid_lock(lock): raise RuntimeError("invalid V211r1 repair lock")
    for key in [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V211r1 dependency changed: {key}")
    if OUTPUT.exists(): raise RuntimeError("V211r1 output exists")
    repair_config = lock["repair_config_payload"]
    learned_path = PROJECT_ROOT / repair_config["parentLearnedLexicon"]
    surface_path = PROJECT_ROOT / repair_config["parentEvaluationSurface"]
    truth_path = PROJECT_ROOT / repair_config["parentEvaluationTruthSealed"]
    predictions_path = OUTPUT / "predictions.jsonl"
    subprocess.run([sys.executable, str(PROJECT_ROOT / lock["prediction_worker"]), "--learned", str(learned_path), "--evaluation-surface", str(surface_path), "--predictions", str(predictions_path)], check=True, cwd=PROJECT_ROOT)
    freeze = {"schema_version": "211r1-prediction-freeze", "predictions_sha256": file_sha256(predictions_path), "evaluation_truth_opened_before_freeze": False, "prediction_worker_evaluation_truth_path_count": 0, "prediction_worker_group_id_read_count": 0}
    write_json(OUTPUT / "prediction-freeze.json", freeze)
    repaired = read_jsonl(predictions_path)
    parent_lock = json.loads((PROJECT_ROOT / lock["parent_V211_design_lock"]).read_text())
    parent_config = parent_lock["config_payload"]
    parent_predictions = read_jsonl(PROJECT_ROOT / parent_config["artifacts"]["predictions"])
    diagnostics = repair_diagnostics(parent_predictions, repaired)
    split = json.loads((PROJECT_ROOT / repair_config["parentSplit"]).read_text())
    learned = json.loads(learned_path.read_text())
    truth = read_jsonl(truth_path)
    reference = json.loads((PROJECT_ROOT / parent_lock["reference_V209r1_outcome"]).read_text())
    reference_repair = json.loads((PROJECT_ROOT / reference["repair_lock"]).read_text())
    v209_lock = json.loads((PROJECT_ROOT / reference_repair["parent_V209_design_lock"]).read_text())
    scores = score_predictions(repaired, truth, v209_lock["config_payload"])
    audit = audit_scores(split, learned, repaired, scores, parent_config)
    repair_valid = bool(diagnostics["normalized_parent_matches_repaired_exactly"] and diagnostics["prediction_values_match_as_multiset"] and diagnostics["changed_prediction_value_count"] == 0 and diagnostics["old_key_count"] == diagnostics["new_key_count"] == 90)
    passed = audit["passed"] and repair_valid
    summary = {"repair_diagnostics": diagnostics, "prediction_freeze": freeze, "scores": scores, "branch": audit["branch"], "model_eligible": audit["model_eligible"], "access": {"protected_read_count": 0, "model_load_or_generation_count": 0, "API_call_count": 0, "training_run_count": 0, "actual_execution_count": 0}}
    result = {"schema_version": "211r1-compositional-baseline-name-repair-result", "experiment": lock["experiment"], "passed": passed, "branch": audit["branch"], "model_eligible": audit["model_eligible"], "decision": audit["decision"], "repair_valid": repair_valid, "checks": audit["checks"], "access_checks": audit["access_checks"], "summary": summary, "authorization": {"design_new_identifiable_open_class_population": passed and audit["branch"] == "ZERO_MODEL_ELIGIBILITY", "separate_local_model_design_only": passed and audit["branch"] == "NONTRIVIAL_MODEL_ELIGIBLE_RESIDUAL", "open_protected_or_run_model": False}}
    write_json(OUTPUT / "repair-diagnostics.json", diagnostics); write_json(OUTPUT / "summary.json", summary); write_json(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
