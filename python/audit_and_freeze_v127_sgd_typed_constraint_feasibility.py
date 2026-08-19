#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v127_sgd_typed_constraint_feasibility import population_gates, select_fresh_population


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v127-sgd-typed-constraint-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v127-sgd-typed-constraint-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v127_sgd_typed_constraint_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v127_sgd_typed_constraint_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v127_sgd_typed_constraint_feasibility.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v127_sgd_typed_constraint_feasibility_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v127_sgd_typed_constraint_feasibility.py"
    audit_path = PROJECT_ROOT / "outputs/v127-sgd-typed-constraint-feasibility/design-audit.json"
    population_path = PROJECT_ROOT / "outputs/v127-sgd-typed-constraint-feasibility/design/fresh-population.json"
    lock_path = PROJECT_ROOT / "configs/v127-sgd-typed-constraint-feasibility-lock.json"
    result_path = PROJECT_ROOT / "outputs/v127-sgd-typed-constraint-feasibility/evaluation/result.json"
    if any(path.exists() for path in (audit_path, population_path, lock_path, result_path)):
        raise RuntimeError("V127 already frozen or evaluated")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV126OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    archive_path = PROJECT_ROOT / config["sourceArchive"]
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    catalog_path = PROJECT_ROOT / config["choiceCatalog"]
    excluded_path = PROJECT_ROOT / config["excludedV125Population"]
    baseline_path = PROJECT_ROOT / config["baselineConfig"]
    v119_path = PROJECT_ROOT / config["V119Config"]
    inventory = json.loads(inventory_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    excluded = json.loads(excluded_path.read_text())
    population = select_fresh_population(inventory, excluded, catalog, config)
    population_checks = population_gates(population, config)
    auth = parent["authorization"]
    mechanism = config["typedConstraintMechanism"]
    checks = {
        "V126_is_valid_and_closed_only_the_retrieval_trigger": bool(
            valid_lock(parent) and valid_lock(parent_lock)
            and parent["outcome"]["passed"] and parent["outcome"]["audit_pass"]
            and not parent["outcome"]["experimental_pass"]
            and auth["close_current_prequery_trigger_inventory"]
            and not auth["fit_threshold_or_select_alternative_current_signal"]
            and not auth["run_language_model_or_protected"]
            and not auth["run_API_training_action_or_execution"]
        ),
        "new_mechanism_is_typed_structural_oracle_not_retrieval_retuning": bool(
            mechanism["oracleUpperBoundOnly"]
            and mechanism["id"] == "oracle_dialogue_state_slot_signature_unique_known_compatibility"
            and mechanism["fitCount"] == mechanism["selectionCount"] == mechanism["thresholdCount"] == 0
            and not mechanism["usesGroundTruthClassIntentServiceOrDomainAtDecisionTime"]
            and config["authorityBoundary"]["annotationsAreUnavailableOracleEvidenceNotRuntimeInputs"]
        ),
        "fresh_population_passes_every_frozen_gate": all(population_checks.values()),
        "archive_and_dependencies_are_present": all(path.is_file() for path in (archive_path, inventory_path, catalog_path, excluded_path, baseline_path, v119_path)),
        "language_model_authority_and_execution_remain_closed": bool(
            not config["freshPopulation"]["persistLanguage"]
            and not mechanism["utteranceFieldMayBeAccessed"]
            and not config["decisionRule"]["passAuthorizesImmediateLanguageOrModelRun"]
            and not config["decisionRule"]["passAuthorizesProtectedInductionOrRicherPlanning"]
            and not config["decisionRule"]["passAuthorizesAPITrainingActionOrExecution"]
            and config["authorityBoundary"]["actualExecutionCount"] == 0
        ),
        "code_exists_and_outputs_are_absent": all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)) and not result_path.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "127-sgd-typed-constraint-feasibility-design-audit",
        "experiment": config["experiment"], "passed": passed, "checks": checks,
        "population_gates": population_checks,
        "decision": "freeze_and_authorize_one_oracle_typed_constraint_feasibility_run" if passed else "reject_V127_design",
        "summary": {key: population[key] for key in ("record_count", "class_counts", "known_pair_coverage", "novel_domain_coverage", "unsupported_domain_coverage", "excluded_identifier_overlap_count")},
        "prelock_access": {"source_archive_read_count": 0, "utterance_field_access_count": 0, "slot_value_access_count": 0, "model_load_count": 0, "model_generation_count": 0, "actual_execution_count": 0},
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    write_json(population_path, population)
    deps = {
        "config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path,
        "source_archive": archive_path, "source_inventory": inventory_path, "choice_catalog": catalog_path,
        "excluded_population": excluded_path, "baseline_config": baseline_path, "V119_config": v119_path,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path,
        "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path,
        "fresh_population": population_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "127-sgd-typed-constraint-feasibility-lock", "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "run_one_oracle_typed_constraint_feasibility_evaluation": True,
            "modify_population_mechanism_channel_cost_gates_or_decision": False,
            "access_utterance_fields_slot_values_or_emit_individual_evidence": False,
            "fit_select_or_load_any_model": False,
            "grant_protected_induction_authority_or_execution": False,
        },
    }
    for key, path in deps.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
