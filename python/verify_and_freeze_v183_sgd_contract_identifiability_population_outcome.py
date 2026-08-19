#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from run_v183_sgd_contract_identifiability_population import DEPENDENCY_KEYS, reconstruct


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v183-sgd-contract-identifiability-population-lock.json"
    output_root = PROJECT_ROOT / "outputs/v183-sgd-contract-identifiability-population/population"
    result_path = output_root / "result.json"
    doc_path = PROJECT_ROOT / "docs/v183-sgd-contract-identifiability-population-results.md"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v183_sgd_contract_identifiability_population_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v183-sgd-contract-identifiability-population/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v183-sgd-contract-identifiability-population-outcome-lock.json"
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V183 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write V183 results before freezing")

    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    population, population_audit = reconstruct(lock)
    expected_payloads = {
        "contract_catalog": population["contract_catalog"],
        "hidden_identifiability": {"records": population["hidden_records"], "contains_language_or_values": False},
        "development_identities": {"records": population["public_development"], "contains_language_or_hidden_labels": False},
        "protected_identities": {"records": population["public_protected"], "contains_language_or_hidden_labels": False},
        "population_summary": population["summary"],
    }
    outputs_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text()) == payload
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key, payload in expected_payloads.items()
    )
    config = lock["config_payload"]
    expected_decision = (
        config["decisionRule"]["ifEveryPopulationAndIdentifiabilityGatePasses"]
        if population_audit["passed"]
        else config["decisionRule"]["otherwise"]
    )
    zero_keys = (
        "utterance_or_dialogue_text_emission_count", "slot_value_or_span_emission_count",
        "manual_language_inspection_count", "deterministic_policy_score_count", "model_load_count",
        "model_generation_count", "API_call_count", "training_run_count",
        "ontology_registration_count", "trusted_state_mutation_count", "protected_language_read_count",
        "service_or_sensor_call_count", "external_side_effect_count", "actual_execution_count",
    )
    checks = {
        "lock_and_dependencies_are_exact": bool(
            payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"})
            == lock["lock_payload_sha256"]
            and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in DEPENDENCY_KEYS)
        ),
        "all_population_outputs_reconstruct_exactly": outputs_exact,
        "summary_gates_and_decision_reconstruct_exactly": bool(
            result["summary"] == population["summary"]
            and result["population_gates"] == population_audit["checks"]
            and result["passed"] == population_audit["passed"]
            and result["decision"] == expected_decision
        ),
        "role_isolation_retention_and_contract_identity_hold": bool(
            population["summary"]["role_identifier_overlap"] == 0
            and population["summary"]["role_source_identifier_overlap"] == 0
            and population["summary"]["target_contract_retention_rate"] == 1.0
            and population["summary"]["missing_control_insufficient_rate"] == 1.0
            and population["summary"]["cross_truth_kind_contract_collision_count"] == 0
        ),
        "no_language_model_authority_or_effect_boundary_holds": all(
            result["access"][key] == 0 for key in zero_keys
        ),
    }
    verified = all(checks.values())
    outcome_audit = {
        "schema_version": "183-sgd-contract-identifiability-population-outcome-audit",
        "experiment": config["experiment"],
        "passed": verified,
        "scientific_population_gates_passed": population_audit["passed"],
        "decision": "freeze_verified_V183_population" if verified else "reject_V183_outcome",
        "checks": checks,
        "independent_summary": population["summary"],
        "additional_access": {
            "source_archive_structured_parse_count": 1,
            "utterance_or_dialogue_text_emission_count": 0,
            "manual_language_inspection_count": 0,
            "model_load_count": 0,
            "actual_execution_count": 0,
        },
    }
    write_json(audit_path, outcome_audit)
    if not verified:
        print(json.dumps(outcome_audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "population_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
        "parent_V182_outcome": PROJECT_ROOT / lock["parent_V182_outcome"],
        "source_V134_outcome": PROJECT_ROOT / lock["source_V134_outcome"],
    }
    for key, item in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / item["path"]
    outcome: dict[str, Any] = {
        "schema_version": "183-sgd-contract-identifiability-population-outcome-lock",
        "experiment": config["experiment"],
        "outcome": {
            "passed": True,
            "scientific_population_gates_passed": population_audit["passed"],
            "decision": result["decision"],
            "summary": result["summary"],
        },
        "authorization": {
            "modify_rebuild_reselect_or_reclassify_V183": False,
            "preregister_role_isolated_language_extraction": bool(population_audit["passed"]),
            "extract_language_without_separate_lock": False,
            "open_protected_language_during_development": False,
            "score_deterministic_policy_or_run_model_API_training_without_later_authorization": False,
            "register_capability_mutate_trusted_state_call_service_act_or_execute": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome)
    write_json(outcome_path, outcome)
    print(json.dumps(outcome_audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__":
    main()
