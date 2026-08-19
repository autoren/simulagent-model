#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v131_complete_clarification_realization_population import (
    build_catalog, evaluate_gates, excluded_identifiers, select_population,
)


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v131-complete-clarification-realization-population.json"
    plan_path = PROJECT_ROOT / "docs/v131-complete-clarification-realization-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v131_complete_clarification_realization_population.py"
    tests_path = PROJECT_ROOT / "python/test_v131_complete_clarification_realization_population.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v131_complete_clarification_realization_population_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v131_complete_clarification_realization_population.py"
    audit_path = PROJECT_ROOT / "outputs/v131-complete-clarification-realization-population/design-audit.json"
    catalog_path = PROJECT_ROOT / "outputs/v131-complete-clarification-realization-population/design/choice-catalog.json"
    population_path = PROJECT_ROOT / "outputs/v131-complete-clarification-realization-population/design/fixture-population.json"
    lock_path = PROJECT_ROOT / "configs/v131-complete-clarification-realization-population-lock.json"
    if any(path.exists() for path in (audit_path, catalog_path, population_path, lock_path)):
        raise RuntimeError("V131 already frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV130OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    inventory = json.loads(inventory_path.read_text())
    excluded = excluded_identifiers(config["excludedPopulations"], PROJECT_ROOT)
    catalog = build_catalog(inventory, config)
    population = select_population(inventory, catalog, excluded, config)
    gates = evaluate_gates(catalog, population, config)
    auth = parent["authorization"]
    checks = {
        "V130_is_valid_positive_and_authorizes_realization_audit": bool(
            valid_lock(parent) and valid_lock(parent_lock) and parent["outcome"]["passed"]
            and parent["outcome"]["audit_pass"] and parent["outcome"]["evidence_feasibility_pass"]
            and auth["preregister_evidence_realization_audit"]
            and not auth["run_language_human_model_or_protected"]
            and not auth["run_API_training_action_or_execution"]
        ),
        "catalog_and_population_pass_all_frozen_gates": all(gates.values()),
        "selection_is_structural_and_pre_language": bool(
            config["population"]["selectionUsesOnlyIdentifierPartitionClassDomainServiceAndIntent"]
            and config["population"]["selectionBeforeAnySelectedLanguageExtraction"]
            and not catalog["contains_language"] and not population["contains_language"]
        ),
        "success_authorizes_only_separate_protocol_lock": bool(
            config["decisionRule"]["passAuthorizesOnlyPreregisterLocalModelRealizationProtocol"]
            and not config["decisionRule"]["passAuthorizesImmediateLanguageExtractionOrModelRun"]
            and not config["decisionRule"]["passAuthorizesProtectedInductionAPITrainingActionOrExecution"]
        ),
        "code_and_outputs_hold": all(path.is_file() for path in (plan_path, protocol_path, tests_path, verifier_path, auditor_path)) and not catalog_path.exists() and not population_path.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "131-complete-clarification-realization-population-design-audit",
        "experiment": config["experiment"], "passed": passed, "checks": checks,
        "population_gates": gates,
        "decision": "freeze_text_free_complete_clarification_realization_population" if passed else "reject_V131_design",
        "summary": {
            "choice_count": catalog["choice_count"], "fixture_count": population["fixture_count"],
            "source_record_count": population["source_record_count"], "missing_control_count": population["missing_control_count"],
            "cell_count": population["cell_count"], "excluded_identifier_count": len(excluded),
            "known_pair_coverage": population["known_pair_coverage"], "novel_domain_coverage": population["novel_domain_coverage"],
            "unsupported_domain_coverage": population["unsupported_domain_coverage"],
        },
        "access": {"language_read_count": 0, "manual_language_inspection_count": 0, "model_load_count": 0, "model_generation_count": 0, "actual_execution_count": 0},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    write_json(catalog_path, catalog); write_json(population_path, population)
    deps = {
        "config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path,
        "source_inventory": inventory_path, "V125_population": PROJECT_ROOT / config["excludedPopulations"][0],
        "V127_population": PROJECT_ROOT / config["excludedPopulations"][1], "V128_population": PROJECT_ROOT / config["excludedPopulations"][2],
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "verifier": verifier_path,
        "auditor": auditor_path, "design_audit": audit_path, "choice_catalog": catalog_path,
        "fixture_population": population_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "131-complete-clarification-realization-population-lock",
        "experiment": config["experiment"], "config_payload": config,
        "authorization": {
            "freeze_text_free_catalog_and_population": True,
            "modify_catalog_population_salt_counts_gates_or_decision": False,
            "preregister_local_model_realization_protocol": True,
            "extract_language_or_run_model_before_protocol_lock": False,
            "grant_protected_induction_API_training_authority_or_execution": False,
        },
    }
    for key, path in deps.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
