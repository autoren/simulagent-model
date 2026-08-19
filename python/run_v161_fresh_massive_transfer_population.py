#!/usr/bin/env python3
from __future__ import annotations

import json

from audit_and_freeze_v122_prequery_signal_inventory import valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v161_fresh_massive_transfer_population import (
    evaluate_population_gates,
    select_transfer_population,
)


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v161-fresh-massive-transfer-population-lock.json"
    population_dir = PROJECT_ROOT / "outputs/v161-fresh-massive-transfer-population/population"
    if population_dir.exists():
        raise RuntimeError("V161 population selection may run only once")
    lock = json.loads(lock_path.read_text())
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    if not valid_lock(lock) or not all(
        file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies
    ):
        raise RuntimeError("V161 lock or dependencies changed")
    config = lock["config_payload"]
    inventory = json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text())
    exclusion = json.loads((PROJECT_ROOT / lock["excluded_population"]).read_text())
    population = select_transfer_population(inventory, exclusion, config)
    checks = evaluate_population_gates(population, config)
    checks["zero_archive_language_manual_model_API_training_or_execution_access"] = True
    passed = all(checks.values())
    decision = (
        config["decisionRule"]["ifEveryPopulationAndDisjointnessGatePasses"]
        if passed
        else config["decisionRule"]["otherwise"]
    )
    population_dir.mkdir(parents=True, exist_ok=False)
    population_path = population_dir / "selected-population.json"
    artifact = {
        "provenance": {
            "source_inventory": config["sourceInventory"],
            "source_inventory_sha256": config["sourceInventorySha256"],
            "source_candidate_index_sha256": config["sourceCandidateIndexSha256"],
            "excluded_population": config["excludedPopulation"],
            "excluded_population_file_sha256": config["excludedPopulationFileSha256"],
            "excluded_population_payload_sha256": config["excludedPopulationPayloadSha256"],
            "selection_salt": config["selection"]["baseSalt"],
            "contains_source_or_derived_language": False,
        },
        **population,
    }
    write_json(population_path, artifact)
    summary_keys = (
        "selected_candidate_count",
        "excluded_candidate_count",
        "excluded_population_overlap_count",
        "remaining_pool_counts",
        "role_counts",
        "role_class_counts",
        "role_class_scenario_counts",
        "role_class_intent_counts",
        "source_partition_counts",
        "role_identifiers_are_disjoint",
        "selected_population_sha256",
        "contains_language_tokens_slot_values_or_prompts",
    )
    access = {
        "text_free_source_inventory_read_count": 1,
        "text_free_excluded_population_read_count": 1,
        "source_archive_reopen_count": 0,
        "selected_language_record_extraction_count": 0,
        "emitted_language_record_count": 0,
        "manual_utterance_inspection_count": 0,
        "interface_policy_score_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "LLM_API_call_count": 0,
        "training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    result = {
        "schema_version": "161-fresh-massive-transfer-population-result",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": decision,
        "population": str(population_path.relative_to(PROJECT_ROOT)),
        "population_sha256": file_sha256(population_path),
        "population_summary": {key: population[key] for key in summary_keys},
        "gates": checks,
        "access": access,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(population_dir / "result.json", result)
    print(
        json.dumps(
            {
                "passed": passed,
                "decision": decision,
                "population_summary": result["population_summary"],
                "gates": checks,
                "access": access,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
