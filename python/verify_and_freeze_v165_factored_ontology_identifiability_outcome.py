#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v165_factored_ontology_identifiability_population import (
    audit_population,
    build_population,
)
from run_v165_factored_ontology_identifiability_population import payload_hash


def main() -> None:
    lock_path = (
        PROJECT_ROOT
        / "configs/v165-factored-ontology-identifiability-population-lock.json"
    )
    result_path = (
        PROJECT_ROOT
        / "outputs/v165-factored-ontology-identifiability/population/result.json"
    )
    doc_path = (
        PROJECT_ROOT
        / "docs/v165-factored-ontology-identifiability-population-results.md"
    )
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v165_factored_ontology_identifiability_outcome.py"
    )
    audit_path = (
        PROJECT_ROOT
        / "outputs/v165-factored-ontology-identifiability/outcome-audit.json"
    )
    outcome_path = (
        PROJECT_ROOT
        / "configs/v165-factored-ontology-identifiability-population-outcome-lock.json"
    )
    if audit_path.exists() or outcome_path.exists():
        raise RuntimeError("V165 outcome is already frozen")
    if not doc_path.is_file():
        raise RuntimeError("write the V165 result before freezing")
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    population = build_population(lock["config_payload"])
    reconstructed = audit_population(population, lock["config_payload"])
    dependency_keys = (
        "config",
        "parent_track_A_outcome",
        "roadmap",
        "plan",
        "protocol",
        "tests",
        "runner",
        "verifier",
        "auditor",
        "design_audit",
    )
    assets_exact = all(
        json.loads((PROJECT_ROOT / result["output_integrity"][key]["path"]).read_text())
        == population[key]
        and file_sha256(PROJECT_ROOT / result["output_integrity"][key]["path"])
        == result["output_integrity"][key]["sha256"]
        for key in (
            "frozen_ontology",
            "public_records",
            "hidden_records",
            "population_summary",
        )
    )
    expected_decision = (
        lock["config_payload"]["decisionRule"][
            "ifEveryPopulationAndIdentifiabilityGatePasses"
        ]
        if reconstructed["passed"]
        else lock["config_payload"]["decisionRule"]["otherwise"]
    )
    zero_access_keys = (
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
    checks = {
        "design_lock_and_dependencies_are_exact": bool(
            payload_hash(
                {
                    key: value
                    for key, value in lock.items()
                    if key != "lock_payload_sha256"
                }
            )
            == lock["lock_payload_sha256"]
            and all(
                file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"]
                for key in dependency_keys
            )
        ),
        "all_population_assets_reconstruct_exactly": assets_exact,
        "audit_pass_and_decision_reconstruct_exactly": bool(
            reconstructed == result["population_audit"]
            and result["passed"] == reconstructed["passed"]
            and result["decision"] == expected_decision
        ),
        "factorization_identifiability_retention_and_renaming_hold": bool(
            reconstructed["target_retention_when_noncontradictory"] == 1.0
            and reconstructed["sufficient_expressibility_classification_accuracy"]
            == 1.0
            and reconstructed["evidence_status_classification_accuracy"] == 1.0
            and reconstructed["renaming_version_space_invariance"] == 1.0
            and reconstructed["public_hidden_field_leak_count"] == 0
        ),
        "evaluation_model_registration_authority_and_execution_boundary_holds": bool(
            result["access"]["population_build_count"] == 1
            and all(result["access"][key] == 0 for key in zero_access_keys)
        ),
    }
    integrity_passed = all(checks.values())
    audit = {
        "schema_version": "165-factored-ontology-identifiability-outcome-audit",
        "experiment": "v165_factored_ontology_identifiability_outcome_audit",
        "passed": integrity_passed,
        "scientific_population_passed": result["passed"],
        "decision": (
            "freeze_positive_V165_population_and_authorize_V166_preregistration"
            if integrity_passed and result["passed"]
            else "reject_V165_outcome"
        ),
        "checks": checks,
        "independent_audit": reconstructed,
        "additional_access": {
            "population_reconstruction_count": 1,
            **{key: 0 for key in zero_access_keys},
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not integrity_passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "population_lock": lock_path,
        "result": result_path,
        "verifier": verifier_path,
        "audit": audit_path,
        "results_document": doc_path,
    }
    for key, integrity in result["output_integrity"].items():
        dependencies[key] = PROJECT_ROOT / integrity["path"]
    outcome: dict[str, Any] = {
        "schema_version": "165-factored-ontology-identifiability-population-outcome-lock",
        "experiment": "v165_factored_ontology_identifiability_population_outcome_lock",
        "outcome": {
            "passed": True,
            "scientific_population_passed": result["passed"],
            "decision": result["decision"],
            "population_audit": reconstructed,
        },
        "authorization": {
            "modify_or_rerun_V165": False,
            "preregister_model_free_V166_deterministic_baselines": bool(
                result["passed"]
            ),
            "score_baselines_without_separate_lock": False,
            "create_or_open_evaluation_population": False,
            "load_or_run_local_or_API_model": False,
            "train_or_fit_learned_component": False,
            "register_provisional_primitive": False,
            "grant_candidate_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        outcome[key] = str(path.relative_to(PROJECT_ROOT))
        outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = hashlib.sha256(
        json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(outcome_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(outcome_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
