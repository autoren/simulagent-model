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
)


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v165r1-outcome-verifier-repair.json"
    plan_path = PROJECT_ROOT / "docs/v165r1-outcome-verifier-repair-plan.md"
    results_doc_path = PROJECT_ROOT / "docs/v165r1-outcome-verifier-repair-results.md"
    protocol_path = PROJECT_ROOT / "python/v165r1_outcome_verifier_repair.py"
    tests_path = PROJECT_ROOT / "python/test_v165r1_outcome_verifier_repair.py"
    auditor_path = (
        PROJECT_ROOT / "python/audit_and_freeze_v165r1_outcome_verifier_repair.py"
    )
    verifier_path = (
        PROJECT_ROOT / "python/verify_and_freeze_v165r1_outcome_verifier_repair.py"
    )
    audit_path = (
        PROJECT_ROOT / "outputs/v165r1-outcome-verifier-repair/design-audit.json"
    )
    lock_path = PROJECT_ROOT / "configs/v165r1-outcome-verifier-repair-lock.json"
    nominal_outcome = (
        PROJECT_ROOT
        / "configs/v165-factored-ontology-identifiability-population-outcome-lock.json"
    )
    repaired_outcome = (
        PROJECT_ROOT / "configs/v165r1-outcome-verifier-repair-outcome-lock.json"
    )
    if audit_path.exists() or lock_path.exists() or repaired_outcome.exists():
        raise RuntimeError("V165r1 is already preregistered or frozen")
    if nominal_outcome.exists():
        raise RuntimeError("nominal V165 outcome unexpectedly exists")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV165PopulationLock"]
    failed_audit_path = PROJECT_ROOT / config["failedV165OutcomeAudit"]
    result_path = PROJECT_ROOT / config["v165Result"]
    parent = json.loads(parent_path.read_text())
    failed = json.loads(failed_audit_path.read_text())
    result = json.loads(result_path.read_text())
    original_verifier_path = PROJECT_ROOT / parent["verifier"]
    parent_dependencies = [
        key
        for key in parent
        if not key.endswith("_sha256") and f"{key}_sha256" in parent
    ]
    population = build_population(parent["config_payload"])
    reconstructed = audit_population(population, parent["config_payload"])
    false_checks = sorted(key for key, value in failed["checks"].items() if not value)
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
    frozen = config["frozenScientificOutcome"]
    checks = {
        "parent_V165_lock_and_dependencies_are_unchanged": bool(
            valid_lock(parent)
            and all(
                file_sha256(PROJECT_ROOT / parent[key]) == parent[f"{key}_sha256"]
                for key in parent_dependencies
            )
        ),
        "original_failed_audit_is_preserved_with_one_expected_false_check": bool(
            not failed["passed"]
            and false_checks == sorted(config["diagnosis"]["expectedFailedChecks"])
            and all(
                value
                for key, value in failed["checks"].items()
                if key not in config["diagnosis"]["expectedFailedChecks"]
            )
        ),
        "sole_mismatch_is_population_build_count_alias_mutation": bool(
            sole_population_build_count_alias_mismatch(
                reconstructed, result["population_audit"]
            )
            and result["access"]["population_build_count"]
            == config["diagnosis"]["requiredResultAccessPopulationBuildCount"]
        ),
        "population_assets_and_scientific_outcome_remain_exact": bool(
            assets_exact
            and reconstructed["passed"] == frozen["passed"]
            and result["decision"] == frozen["decision"]
            and reconstructed["summary"]["record_count"] == frozen["recordCount"]
            and reconstructed["summary"]["candidate_truth_table_count"]
            == frozen["candidateTruthTableCount"]
            and reconstructed["target_retention_when_noncontradictory"]
            == frozen["targetRetention"]
            and reconstructed["evidence_status_classification_accuracy"]
            == frozen["evidenceStatusAccuracy"]
            and reconstructed["renaming_version_space_invariance"]
            == frozen["renamingInvariance"]
        ),
        "original_verifier_and_failed_artifacts_are_unmodified": bool(
            file_sha256(original_verifier_path) == parent["verifier_sha256"]
            and file_sha256(failed_audit_path)
            == file_sha256(PROJECT_ROOT / config["failedV165OutcomeAudit"])
        ),
        "repair_scope_has_zero_scientific_or_external_access": bool(
            all(value == 0 for value in config["accessGates"].values())
            and not config["authorization"]["modifyOriginalV165Artifacts"]
            and not config["authorization"]["rerunPopulationBuild"]
            and not config["authorization"][
                "changeScientificRecordsMetricsGatesDecisionOrClaims"
            ]
            and not config["authorization"][
                "runModelAPITrainingRegistrationAuthorityActionOrExecution"
            ]
        ),
        "required_files_exist": all(
            path.is_file()
            for path in (
                config_path,
                plan_path,
                results_doc_path,
                protocol_path,
                tests_path,
                auditor_path,
                verifier_path,
                parent_path,
                failed_audit_path,
                result_path,
                original_verifier_path,
            )
        ),
        "nominal_V165_outcome_remains_absent": not nominal_outcome.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "165r1-outcome-verifier-repair-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "diagnosis": {
            "false_original_checks": false_checks,
            "persisted_population_build_count": result["population_audit"]["access"].get(
                "population_build_count"
            ),
            "exact_after_locked_projection": sole_population_build_count_alias_mismatch(
                reconstructed, result["population_audit"]
            ),
        },
        "repair_access": {key: 0 for key in config["accessGates"]},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    paths = {
        "config": config_path,
        "plan": plan_path,
        "results_document": results_doc_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
        "parent_population_lock": parent_path,
        "failed_V165_outcome_audit": failed_audit_path,
        "V165_result": result_path,
        "original_V165_verifier": original_verifier_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "165r1-outcome-verifier-repair-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": config["authorization"],
    }
    for key, path in paths.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
