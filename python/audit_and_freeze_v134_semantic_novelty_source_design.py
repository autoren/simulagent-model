#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v134_semantic_novelty_source_design import build_catalog, derive_classes, evaluate_gates, schema_identifiability, select_population


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v134-semantic-novelty-source-design.json"
    plan_path = PROJECT_ROOT / "docs/v134-semantic-novelty-source-design-plan.md"
    protocol_path = PROJECT_ROOT / "python/v134_semantic_novelty_source_design.py"
    tests_path = PROJECT_ROOT / "python/test_v134_semantic_novelty_source_design.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v134_semantic_novelty_source_design_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v134_semantic_novelty_source_design.py"
    audit_path = PROJECT_ROOT / "outputs/v134-semantic-novelty-source-design/design-audit.json"
    catalog_path = PROJECT_ROOT / "outputs/v134-semantic-novelty-source-design/design/choice-catalog.json"
    population_path = PROJECT_ROOT / "outputs/v134-semantic-novelty-source-design/design/fixture-population.json"
    ident_path = PROJECT_ROOT / "outputs/v134-semantic-novelty-source-design/design/schema-identifiability.json"
    lock_path = PROJECT_ROOT / "configs/v134-semantic-novelty-source-design-lock.json"
    if any(path.exists() for path in (audit_path, catalog_path, population_path, ident_path, lock_path)): raise RuntimeError("V134 already frozen")
    config = json.loads(config_path.read_text()); parent_path = PROJECT_ROOT / config["parentV133OutcomeLock"]; parent = json.loads(parent_path.read_text()); parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]; parent_lock = json.loads(parent_lock_path.read_text())
    inventory_path = PROJECT_ROOT / config["sourceInventory"]; inventory = json.loads(inventory_path.read_text()); archive_path = PROJECT_ROOT / config["sourceArchive"]
    rows = derive_classes(inventory, config["population"]["sourcePartition"]); catalog = build_catalog(rows, config); population = select_population(rows, catalog, config); ident = schema_identifiability(archive_path.read_bytes(), catalog, population, config); gates = evaluate_gates(catalog, population, ident, config)
    auth = parent["authorization"]
    checks = {"V133_is_valid_negative_and_authorizes_only_text_free_source_design": bool(valid_lock(parent) and valid_lock(parent_lock) and parent["outcome"]["passed"] and parent["outcome"]["audit_pass"] and not parent["outcome"]["identifiability_pass"] and auth["preregister_text_free_semantically_noncolliding_source_design"] and not auth["rerun_model_revise_prompt_or_scale"]), "source_population_and_identifiability_gates_pass": all(gates.values()), "dev_language_remains_unopened": config["population"]["devLanguagePreviouslyUsedCount"] == 0 and config["sourceGates"]["maximumLanguageReadCount"] == 0, "pass_creates_asset_without_model_authority": config["decisionRule"]["passCreatesFutureBenchmarkAssetOnly"] and not config["decisionRule"]["passAuthorizesLanguageExtractionOrModelRun"] and not config["decisionRule"]["passAuthorizesProtectedInductionRicherPlanningAPITrainingActionOrExecution"], "code_and_outputs_hold": all(path.is_file() for path in (plan_path, protocol_path, tests_path, verifier_path, auditor_path)) and not catalog_path.exists()}
    passed = all(checks.values()); audit = {"schema_version": "134-semantic-novelty-source-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "source_gates": gates, "decision": config["decisionRule"]["ifEverySourceIdentifiabilityAndAccessGatePasses"] if passed else config["decisionRule"]["otherwise"], "summary": {"choice_count": catalog["choice_count"], "fixture_count": population["fixture_count"], "source_record_count": population["source_record_count"], "cell_count": population["cell_count"], "novel_name_collision_fraction": ident["selected_novel_exact_name_collision_fraction"], "novel_full_signature_collision_fraction": ident["selected_novel_full_signature_collision_fraction"]}, "access": {"source_archive_read_count": 1, "schema_file_read_count": 2, "dialogue_file_read_count": 0, "language_record_read_count": 0, "model_load_count": 0, "model_generation_count": 0, "actual_execution_count": 0}}
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    write_json(catalog_path, catalog); write_json(population_path, population); write_json(ident_path, ident)
    deps = {"config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path, "source_archive": archive_path, "source_inventory": inventory_path, "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path, "choice_catalog": catalog_path, "fixture_population": population_path, "schema_identifiability": ident_path}
    lock: dict[str, Any] = {"schema_version": "134-semantic-novelty-source-design-lock", "experiment": config["experiment"], "config_payload": config, "authorization": {"freeze_future_benchmark_asset": True, "modify_reselect_or_redefine_V134": False, "extract_language_or_run_model": False, "run_API_training_authority_or_execution": False}}
    for key, path in deps.items(): lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
