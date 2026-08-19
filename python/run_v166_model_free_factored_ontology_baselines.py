#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v166_model_free_factored_ontology_baselines import payload_hash
from v166_model_free_factored_ontology_baselines import (
    build_predictions,
    evaluate_gates,
    evaluate_predictions,
)


DEPENDENCY_KEYS = (
    "config", "parent_V165r1_outcome", "plan", "protocol", "tests", "runner",
    "verifier", "auditor", "design_audit", "frozen_ontology", "public_records",
    "hidden_records", "population_summary",
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def reconstruct(lock: dict[str, Any]) -> dict[str, Any]:
    public = json.loads((PROJECT_ROOT / lock["public_records"]).read_text())
    hidden = json.loads((PROJECT_ROOT / lock["hidden_records"]).read_text())
    ontology = json.loads((PROJECT_ROOT / lock["frozen_ontology"]).read_text())
    summary = json.loads((PROJECT_ROOT / lock["population_summary"]).read_text())
    if len(public) != summary["record_count"] or len(hidden) != summary["record_count"]:
        raise RuntimeError("V166 input population count mismatch")
    predictions = build_predictions(public, hidden, ontology)
    evaluation = evaluate_predictions(predictions, hidden)
    return {
        "predictions": predictions,
        "evaluation": evaluation,
        "population_summary": summary,
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v166-model-free-factored-ontology-baselines-lock.json"
    output_root = PROJECT_ROOT / "outputs/v166-model-free-factored-ontology/baselines"
    if output_root.exists():
        raise RuntimeError("V166 model-free baselines may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V166 lock mismatch")
    for key in DEPENDENCY_KEYS:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V166 dependency drifted: {key}")

    artifacts = reconstruct(lock)
    config = lock["config_payload"]
    access = {
        "public_record_read_count": 1,
        "hidden_truth_read_count": 1,
        "frozen_ontology_read_count": 1,
        "population_summary_read_count": 1,
        "evaluation_record_count": 0,
        "manual_judgment_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "training_run_count": 0,
        "ontology_registration_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    gates = evaluate_gates(artifacts["evaluation"], access, config)
    passed = all(gates.values())
    residual_count = artifacts["evaluation"]["model_eligible_residual_count"]
    if passed and residual_count == 0:
        decision = config["decisionRule"]["ifEveryBaselineGatePassesAndResidualIsZero"]
    elif passed:
        decision = config["decisionRule"]["ifPipelinePassesButResidualIsNonzero"]
    else:
        decision = config["decisionRule"]["otherwise"]

    predictions_path = output_root / "baseline-predictions.json"
    evaluation_path = output_root / "baseline-evaluation.json"
    residual_path = output_root / "model-eligible-residual.json"
    write_json(predictions_path, {"predictions": artifacts["predictions"], "contains_language": False})
    write_json(evaluation_path, artifacts["evaluation"])
    write_json(residual_path, {
        "record_ids": artifacts["evaluation"]["model_eligible_residual_record_ids"],
        "count": residual_count,
        "intentionally_ambiguous_records_are_residuals": False,
        "contains_language": False,
    })
    output_integrity = {
        key: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)}
        for key, path in {
            "baseline_predictions": predictions_path,
            "baseline_evaluation": evaluation_path,
            "model_eligible_residual": residual_path,
        }.items()
    }
    result = {
        "schema_version": "166-model-free-factored-ontology-baselines-result",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": decision,
        "baseline_metrics": artifacts["evaluation"]["baseline_metrics"],
        "model_eligible_residual_count": residual_count,
        "intentionally_ambiguous_record_count": artifacts["evaluation"]["intentionally_ambiguous_record_count"],
        "intentionally_ambiguous_candidate_counts": artifacts["evaluation"]["intentionally_ambiguous_candidate_counts"],
        "gates": gates,
        "access": access,
        "output_integrity": output_integrity,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_root / "result.json", result)
    print(json.dumps({
        "passed": passed,
        "decision": decision,
        "baseline_metrics": artifacts["evaluation"]["baseline_metrics"],
        "model_eligible_residual_count": residual_count,
        "intentionally_ambiguous_record_count": result["intentionally_ambiguous_record_count"],
        "ambiguous_candidate_count_values": sorted(set(result["intentionally_ambiguous_candidate_counts"])),
        "gates": gates,
        "access": access,
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
