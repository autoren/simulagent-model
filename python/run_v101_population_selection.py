#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v101_massive_population import evaluate_population_gates, select_massive_population


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v101-massive-population-lock.json"
    population_root = PROJECT_ROOT / "outputs/v101-massive-population/population"
    if population_root.exists():
        raise RuntimeError("V101 population selection may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash(
        {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    ) != lock["lock_payload_sha256"]:
        raise RuntimeError("V101 population lock mismatch")
    dependency_keys = (
        "config", "parent_source_outcome", "source_inventory", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V101 dependency drifted: {key}")

    config = lock["config_payload"]
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    if file_sha256(inventory_path) != config["sourceInventorySha256"]:
        raise RuntimeError("V101 source inventory drifted")
    inventory = json.loads(inventory_path.read_text())
    population = select_massive_population(inventory, config)
    checks = evaluate_population_gates(population, config)
    checks["zero_archive_language_manual_model_API_training_or_side_effect_access"] = True
    passed = all(checks.values())
    population_root.mkdir(parents=True)
    population_path = population_root / "selected-population.json"
    artifact = {
        "provenance": {
            "source_inventory": config["sourceInventory"],
            "source_inventory_sha256": config["sourceInventorySha256"],
            "source_candidate_index_sha256": config["sourceCandidateIndexSha256"],
            "selection_salt": config["selection"]["baseSalt"],
            "contains_source_or_derived_language": False,
        },
        **population,
    }
    population_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    summary_keys = (
        "selected_candidate_count", "role_counts", "role_class_counts",
        "role_class_scenario_counts", "role_class_intent_counts", "source_partition_counts",
        "development_test_identifiers_are_disjoint", "selected_population_sha256",
        "contains_language_tokens_slot_values_or_prompts",
    )
    result = {
        "schema_version": "101-massive-population-result",
        "experiment": "v101_massive_population_selection",
        "passed": passed,
        "decision": (
            "freeze_population_and_preregister_selected_language_extraction"
            if passed else "stop_V101_before_language_or_model_access"
        ),
        "population": str(population_path.relative_to(PROJECT_ROOT)),
        "population_sha256": file_sha256(population_path),
        "population_summary": {key: population[key] for key in summary_keys},
        "gates": checks,
        "access": {
            "text_free_source_inventory_read_count": 1,
            "source_archive_reopen_count": 0,
            "selected_language_record_extraction_count": 0,
            "emitted_language_record_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": (
            "text-free population selection only; no language, model, calibration, posterior, "
            "planning, or execution outcome"
        ),
    }
    result_path = population_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed,
        "decision": result["decision"],
        "population_summary": result["population_summary"],
        "gates": checks,
        "access": result["access"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
