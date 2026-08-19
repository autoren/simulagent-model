#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v125_sgd_catalog_population import build_catalog, evaluate_gates, select_populations


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v125-sgd-catalog-population.json"
    plan_path = PROJECT_ROOT / "docs/v125-sgd-catalog-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v125_sgd_catalog_population.py"
    tests_path = PROJECT_ROOT / "python/test_v125_sgd_catalog_population.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v125_sgd_catalog_population_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v125_sgd_catalog_population.py"
    audit_path = PROJECT_ROOT / "outputs/v125-sgd-catalog-population/design-audit.json"
    catalog_path = PROJECT_ROOT / "outputs/v125-sgd-catalog-population/design/choice-catalog.json"
    population_path = PROJECT_ROOT / "outputs/v125-sgd-catalog-population/design/selected-populations.json"
    lock_path = PROJECT_ROOT / "configs/v125-sgd-catalog-population-lock.json"
    if any(path.exists() for path in (audit_path, catalog_path, population_path, lock_path)):
        raise RuntimeError("V125 already frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV124OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    inventory = json.loads(inventory_path.read_text())
    catalog = build_catalog(inventory, config)
    populations = select_populations(inventory, config)
    population_checks = evaluate_gates(catalog, populations, config)
    auth = parent["authorization"]
    checks = {
        "V124_is_valid_positive_and_authorizes_only_text_free_design": bool(
            valid_lock(parent) and valid_lock(parent_lock)
            and parent["outcome"]["passed"] and parent["outcome"]["audit_pass"] and parent["outcome"]["source_pass"]
            and auth["preregister_text_free_SGD_catalog_and_population"]
            and not auth["extract_selected_language_or_evaluate_signal_trigger_model"]
            and not auth["run_API_training_action_or_execution"]
            and file_sha256(inventory_path) == parent["inventory_sha256"]
        ),
        "catalog_and_populations_pass_all_frozen_gates": all(population_checks.values()),
        "selection_is_structural_pre_language_and_complete_safe": bool(
            config["evaluationPopulation"]["selectionUsesOnlyFrozenIdentifierPartitionClassDomainServiceAndIntent"]
            and config["evaluationPopulation"]["selectionBeforeAnySelectedLanguageExtraction"]
            and config["catalog"]["completeSafeCompositeHypothesisUniverse"]
            and not catalog["contains_language"] and not populations["contains_language"]
        ),
        "success_authorizes_only_separate_retrieval_selectivity_design": bool(
            config["decisionRule"]["passAuthorizesOnlyPreregisterCrossDatasetRetrievalSelectivityDesign"]
            and not config["decisionRule"]["passAuthorizesImmediateLanguageExtractionSignalTriggerOrModelEvaluation"]
            and not config["decisionRule"]["passAuthorizesProtectedInductionAPITrainingActionOrExecution"]
        ),
        "code_and_outputs_hold": all(path.is_file() for path in (plan_path, protocol_path, tests_path, verifier_path, auditor_path)) and not catalog_path.exists() and not population_path.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "125-sgd-catalog-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "checks": checks,
        "population_gates": population_checks,
        "decision": "freeze_text_free_catalog_and_populations" if passed else "reject_V125_design",
        "summary": {
            "choice_count": catalog["choice_count"],
            "training_record_count": populations["training_record_count"],
            "evaluation_record_count": populations["evaluation_record_count"],
            "evaluation_class_counts": populations["evaluation_class_counts"],
            "known_pair_coverage": populations["known_pair_coverage"],
            "novel_domain_coverage": populations["novel_domain_coverage"],
            "unsupported_domain_coverage": populations["unsupported_domain_coverage"],
        },
        "access": {"language_read_count": 0, "manual_language_inspection_count": 0, "model_load_count": 0, "model_generation_count": 0, "actual_execution_count": 0},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    write_json(catalog_path, catalog)
    write_json(population_path, populations)
    deps = {
        "config": config_path,
        "parent_outcome": parent_path,
        "parent_analysis_lock": parent_lock_path,
        "source_inventory": inventory_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
        "choice_catalog": catalog_path,
        "selected_populations": population_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "125-sgd-catalog-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "freeze_text_free_catalog_and_population": True,
            "modify_catalog_population_counts_salt_gates_or_decision": False,
            "extract_or_inspect_language": False,
            "evaluate_retrieval_signal_trigger_or_model": False,
            "grant_protected_induction_authority_or_execution": False,
        },
    }
    for key, path in deps.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
