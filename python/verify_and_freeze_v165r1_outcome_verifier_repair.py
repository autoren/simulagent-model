#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v165_factored_ontology_identifiability_population import (
    audit_population,
    build_population,
)
from audit_and_freeze_v165_factored_ontology_identifiability_population import (
    payload_hash,
    valid_lock,
)
from v165r1_outcome_verifier_repair import (
    sole_population_build_count_alias_mismatch,
    without_population_build_count,
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    repair_lock_path = PROJECT_ROOT / "configs/v165r1-outcome-verifier-repair-lock.json"
    audit_path = (
        PROJECT_ROOT / "outputs/v165r1-outcome-verifier-repair/outcome-audit.json"
    )
    outcome_path = (
        PROJECT_ROOT / "configs/v165r1-outcome-verifier-repair-outcome-lock.json"
    )
    nominal_outcome = (
        PROJECT_ROOT
        / "configs/v165-factored-ontology-identifiability-population-outcome-lock.json"
    )
    verifier_path = (
        PROJECT_ROOT / "python/verify_and_freeze_v165r1_outcome_verifier_repair.py"
    )
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V165r1 outcome is already frozen")
    if nominal_outcome.exists():
        raise RuntimeError("nominal V165 outcome unexpectedly exists")
    repair = json.loads(repair_lock_path.read_text())
    config = repair["config_payload"]
    parent = json.loads((PROJECT_ROOT / repair["parent_population_lock"]).read_text())
    failed = json.loads((PROJECT_ROOT / repair["failed_V165_outcome_audit"]).read_text())
    result = json.loads((PROJECT_ROOT / repair["V165_result"]).read_text())
    population = build_population(parent["config_payload"])
    reconstructed = audit_population(population, parent["config_payload"])
    repair_dependencies = [
        key
        for key in repair
        if not key.endswith("_sha256") and f"{key}_sha256" in repair
    ]
    parent_dependencies = [
        key
        for key in parent
        if not key.endswith("_sha256") and f"{key}_sha256" in parent
    ]
    false_checks = sorted(key for key, value in failed["checks"].items() if not value)
    asset_keys = (
        "frozen_ontology",
        "public_records",
        "hidden_records",
        "population_summary",
    )
    assets_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text())
        == population[key]
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key in asset_keys
    )
    expected_decision = config["frozenScientificOutcome"]["decision"]
    checks = {
        "repair_lock_and_dependencies_are_exact": bool(
            valid_lock(repair)
            and all(
                file_sha256(PROJECT_ROOT / repair[key]) == repair[f"{key}_sha256"]
                for key in repair_dependencies
            )
        ),
        "parent_lock_and_dependencies_are_exact": bool(
            valid_lock(parent)
            and all(
                file_sha256(PROJECT_ROOT / parent[key]) == parent[f"{key}_sha256"]
                for key in parent_dependencies
            )
        ),
        "failed_original_audit_is_preserved_with_exact_one_false_check": bool(
            not failed["passed"]
            and false_checks == config["diagnosis"]["expectedFailedChecks"]
        ),
        "sole_alias_mutation_mismatch_is_reproduced": bool(
            sole_population_build_count_alias_mismatch(
                reconstructed, result["population_audit"]
            )
            and without_population_build_count(result["population_audit"])
            == reconstructed
        ),
        "all_scientific_assets_metrics_gates_and_decision_are_exact": bool(
            assets_exact
            and reconstructed["passed"]
            and result["passed"]
            and result["decision"] == expected_decision
            and all(reconstructed["checks"].values())
        ),
        "original_build_and_zero_external_access_are_exact": bool(
            result["access"]["population_build_count"] == 1
            and all(
                result["access"][key] == 0
                for key in (
                    "evaluation_record_count",
                    "manual_judgment_count",
                    "model_load_count",
                    "model_generation_count",
                    "API_call_count",
                    "training_run_count",
                    "ontology_registration_count",
                    "real_service_call_count",
                    "external_side_effect_count",
                    "actual_execution_count",
                )
            )
            and all(value == 0 for value in config["accessGates"].values())
        ),
        "nominal_V165_outcome_remains_absent": not nominal_outcome.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "165r1-outcome-verifier-repair-outcome-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "technical_repair_only": True,
        "repair": "remove_runner_added_population_build_count_from_embedded_audit_for_exact_comparison",
        "scientific_population_passed": reconstructed["passed"],
        "decision": expected_decision,
        "population_audit": reconstructed,
        "repair_access": {key: 0 for key in config["accessGates"]},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    paths = {
        "repair_lock": repair_lock_path,
        "parent_population_lock": PROJECT_ROOT / repair["parent_population_lock"],
        "failed_V165_outcome_audit": PROJECT_ROOT / repair["failed_V165_outcome_audit"],
        "V165_result": PROJECT_ROOT / repair["V165_result"],
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": PROJECT_ROOT / repair["results_document"],
    }
    for key in asset_keys:
        paths[key] = PROJECT_ROOT / result["output_integrity"][key]["path"]
    outcome: dict[str, Any] = {
        "schema_version": "165r1-outcome-verifier-repair-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "technical_repair_only": True,
            "scientific_population_passed": True,
            "decision": expected_decision,
            "population_audit": reconstructed,
        },
        "authorization": {
            "retain_V165_as_project_authored_model_free_positive_development_evidence": True,
            "treat_repair_as_new_scientific_evidence": False,
            "create_nominal_V165_outcome_lock": False,
            "preregister_model_free_V166_deterministic_baselines": True,
            "score_baselines_without_separate_lock": False,
            "create_or_open_evaluation_population": False,
            "load_or_run_local_or_API_model": False,
            "train_or_fit_learned_component": False,
            "register_provisional_primitive": False,
            "grant_candidate_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
        },
    }
    for key, path in paths.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
