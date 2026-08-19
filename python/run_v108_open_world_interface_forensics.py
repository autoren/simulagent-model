#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v108_open_world_interface_forensics import (
    aggregate_only_analysis, analyze_existing_outputs, evaluate_forensics_gates,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reconstruct(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads((PROJECT_ROOT / lock["model_result"]).read_text())
    population = json.loads((PROJECT_ROOT / lock["selected_population"]).read_text())
    membership = json.loads((PROJECT_ROOT / lock["development_membership"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["visible_catalog"]).read_text())
    evaluation_ids = {
        row["record_id"] for row in membership["membership"] if row["subset"] == "evaluation"
    }
    structural = {row["population_id"]: row for row in population["selected_population"]}
    evaluation_records = [{
        "record_id": identifier,
        "class_label": structural[identifier]["class_label"],
        "scenario": structural[identifier]["scenario"],
        "intent": structural[identifier]["intent"],
    } for identifier in sorted(evaluation_ids)]
    controlled_ids = {
        identifier for identifier, row in result["fixtures"].items()
        if row["kind"] == "controlled_missing_observation"
    }
    analysis = analyze_existing_outputs(
        result["fixtures"], evaluation_records, controlled_ids, catalog,
        lock["interface_config_payload"], lock["baseline_config_payload"],
    )
    return analysis, result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v108-open-world-interface-forensics-lock.json"
    output_root = PROJECT_ROOT / "outputs/v108-open-world-interface-forensics/forensics"
    if output_root.exists():
        raise RuntimeError("V108 diagnostic may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V108 diagnostic lock mismatch")
    dependency_keys = (
        "config", "parent_model_outcome", "implementation_lock", "baseline_outcome",
        "model_result", "selected_population", "development_membership", "visible_catalog",
        "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V108 dependency drifted: {key}")
    analysis, model_result = reconstruct(lock)
    access = {
        "existing_raw_response_automatic_parse_count": len(model_result["fixtures"]),
        "development_language_read_count": 0, "protected_test_language_read_count": 0,
        "manual_raw_response_inspection_count": 0, "model_load_count": 0,
        "model_generation_count": 0, "LLM_API_call_count": 0,
        "adapter_training_run_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    checks = evaluate_forensics_gates(
        analysis, model_result["metrics"], access, lock["config_payload"],
    )
    checks["aggregate_output_contains_no_raw_response_or_individual_identifier"] = (
        analysis["raw_response_or_identifier_emission_count"] == 0
    )
    passed = all(checks.values())
    aggregate = aggregate_only_analysis(analysis)
    aggregate_path = output_root / "aggregate-diagnostic.json"
    write_json(aggregate_path, aggregate)
    result = {
        "schema_version": "108-open-world-interface-forensics-result",
        "experiment": "v108_existing_output_typed_identifier_mismatch_diagnostic",
        "passed": passed,
        "decision": (
            "format_mismatch_is_dominant_preregister_fresh_constrained_typed_interface"
            if passed else "format_mismatch_is_not_sufficient_prioritize_sequential_clarification"
        ),
        "analysis": aggregate,
        "gates": checks, "access": access,
        "output_integrity": {
            "aggregate_diagnostic": {
                "path": str(aggregate_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(aggregate_path),
            }
        },
        "claim_boundary": "aggregate diagnostic counterfactual only; frozen V107 scores unchanged; no language, generation, model, protected test, retry, action, or execution",
    }
    result_path = output_root / "result.json"
    write_json(result_path, result)
    print(json.dumps({
        "passed": passed, "decision": result["decision"],
        "fixture_counts": aggregate["fixture_counts"],
        "invalid_observed_canonicalizable_fraction": aggregate["invalid_observed_canonicalizable_fraction"],
        "observed_category_counts": aggregate["observed_category_counts"],
        "transform_counts": aggregate["transform_counts"],
        "original": {
            "exact": aggregate["original_metrics"]["observed_exact_decision_accuracy"],
            "known_exact": aggregate["original_metrics"]["known_exact_intent_accuracy"],
        },
        "counterfactual": {
            "exact": aggregate["counterfactual_metrics"]["observed_exact_decision_accuracy"],
            "known_exact": aggregate["counterfactual_metrics"]["known_exact_intent_accuracy"],
        },
        "gates": checks, "access": access,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
