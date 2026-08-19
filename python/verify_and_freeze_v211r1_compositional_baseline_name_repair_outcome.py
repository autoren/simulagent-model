#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v211r1_compositional_baseline_name_repair import audit_scores, predict_evaluation, repair_diagnostics, score_predictions
from v22r2_grounding import PROJECT_ROOT


OUT = PROJECT_ROOT / "outputs/v211r1-compositional-baseline-name-repair/evaluation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v211r1-compositional-baseline-name-repair-lock.json"
    lock = json.loads(lock_path.read_text())
    audit_path = PROJECT_ROOT / "outputs/v211r1-compositional-baseline-name-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v211r1-compositional-baseline-name-repair-outcome-lock.json"
    results_path = PROJECT_ROOT / "docs/v211r1-compositional-baseline-name-repair-results.md"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V211r1 frozen")
    keys = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    dependencies_exact = valid_lock(lock) and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in keys)
    repair_config = lock["repair_config_payload"]
    learned = json.loads((PROJECT_ROOT / repair_config["parentLearnedLexicon"]).read_text())
    surfaces = read_jsonl(PROJECT_ROOT / repair_config["parentEvaluationSurface"])
    truth = read_jsonl(PROJECT_ROOT / repair_config["parentEvaluationTruthSealed"])
    parent_lock = json.loads((PROJECT_ROOT / lock["parent_V211_design_lock"]).read_text())
    parent_config = parent_lock["config_payload"]
    repaired = predict_evaluation(surfaces, learned)
    parent_predictions = read_jsonl(PROJECT_ROOT / parent_config["artifacts"]["predictions"])
    diagnostics = repair_diagnostics(parent_predictions, repaired)
    stored_predictions = read_jsonl(OUT / "predictions.jsonl")
    predictions_exact = stored_predictions == repaired
    reference = json.loads((PROJECT_ROOT / parent_lock["reference_V209r1_outcome"]).read_text())
    ref_repair = json.loads((PROJECT_ROOT / reference["repair_lock"]).read_text())
    v209_lock = json.loads((PROJECT_ROOT / ref_repair["parent_V209_design_lock"]).read_text())
    scores = score_predictions(repaired, truth, v209_lock["config_payload"])
    split = json.loads((PROJECT_ROOT / repair_config["parentSplit"]).read_text())
    rebuilt = audit_scores(split, learned, repaired, scores, parent_config)
    summary = json.loads((OUT / "summary.json").read_text()); result = json.loads((OUT / "result.json").read_text())
    freeze = json.loads((OUT / "prediction-freeze.json").read_text())
    checks = {
        "repair_lock_and_dependencies_exact": dependencies_exact,
        "prediction_values_unchanged_and_only_key_repaired": bool(diagnostics["normalized_parent_matches_repaired_exactly"] and diagnostics["changed_prediction_value_count"] == 0),
        "predictions_reconstruct_exactly": predictions_exact,
        "prediction_freeze_exact_and_before_truth": bool(freeze["predictions_sha256"] == file_sha256(OUT / "predictions.jsonl") and not freeze["evaluation_truth_opened_before_freeze"]),
        "summary_and_result_reconstruct": bool(summary["repair_diagnostics"] == diagnostics and summary["scores"] == scores and result["passed"] == rebuilt["passed"] and result["branch"] == rebuilt["branch"] and result["decision"] == rebuilt["decision"]),
        "scientific_and_access_audit_pass": rebuilt["passed"],
        "results_document_exists": results_path.is_file(),
    }
    passed = all(checks.values())
    audit = {"schema_version": "211r1-compositional-baseline-name-repair-outcome-audit", "experiment": lock["experiment"], "passed": passed, "branch": rebuilt["branch"], "model_eligible": rebuilt["model_eligible"], "decision": "freeze_verified_V211r1" if passed else "freeze_failed_V211r1_verification", "checks": checks, "scores": scores}
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2)); raise SystemExit(1)
    deps = {"repair_lock": lock_path, "audit": audit_path, "predictions": OUT / "predictions.jsonl", "prediction_freeze": OUT / "prediction-freeze.json", "summary": OUT / "summary.json", "result": OUT / "result.json", "results_document": results_path, "verifier": PROJECT_ROOT / lock["verifier"]}
    outcome: dict[str, Any] = {"schema_version": "211r1-compositional-baseline-name-repair-outcome-lock", "experiment": lock["experiment"], "outcome": {"passed": True, "branch": rebuilt["branch"], "model_eligible": rebuilt["model_eligible"], "decision": rebuilt["decision"], "scores": scores}, "authorization": {"design_new_identifiable_open_class_population": rebuilt["branch"] == "ZERO_MODEL_ELIGIBILITY", "preregister_separate_local_model_design_only": rebuilt["branch"] == "NONTRIVIAL_MODEL_ELIGIBLE_RESIDUAL", "open_protected_or_run_model": False}}
    for key, path in deps.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
