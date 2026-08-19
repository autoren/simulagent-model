#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v133-sgd-capability-label-identifiability.json"
    plan_path = PROJECT_ROOT / "docs/v133-sgd-capability-label-identifiability-plan.md"
    protocol_path = PROJECT_ROOT / "python/v133_sgd_capability_label_identifiability.py"
    tests_path = PROJECT_ROOT / "python/test_v133_sgd_capability_label_identifiability.py"
    runner_path = PROJECT_ROOT / "python/run_v133_sgd_capability_label_identifiability.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v133_sgd_capability_label_identifiability_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v133_sgd_capability_label_identifiability.py"
    audit_path = PROJECT_ROOT / "outputs/v133-sgd-capability-label-identifiability/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v133-sgd-capability-label-identifiability-lock.json"
    if audit_path.exists() or lock_path.exists(): raise RuntimeError("V133 already frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV132OutcomeLock"]
    parent = json.loads(parent_path.read_text()); parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text())
    catalog = json.loads((PROJECT_ROOT / config["choiceCatalog"]).read_text())
    population = json.loads((PROJECT_ROOT / config["fixturePopulation"]).read_text())
    checks = {
        "V132_is_valid_frozen_negative": bool(valid_lock(parent) and valid_lock(parent_lock) and parent["outcome"]["passed"] and parent["outcome"]["audit_pass"] and not parent["outcome"]["realization_pass"] and not parent["authorization"]["modify_rerun_retry_retune_or_mine_V132"]),
        "audit_is_model_free_and_schema_only": bool(config["accessGates"]["maximumSchemaFileReadCount"] == 2 and config["accessGates"]["maximumDialogueFileReadCount"] == 0 and config["accessGates"]["maximumLanguageRecordReadCount"] == 0 and config["accessGates"]["maximumModelLoadCount"] == config["accessGates"]["maximumModelGenerationCount"] == 0),
        "selected_population_and_catalog_match_claim": bool(catalog["choice_count"] == 11 and population["truth_choice_counts"]["N01"] == population["truth_choice_counts"]["N02"] == population["truth_choice_counts"]["N03"] == 24),
        "collision_rules_and_gates_are_exactly_preregistered": bool(config["schemaAudit"]["primaryCollision"].startswith("Unicode-normalized") and config["identifiabilityGates"]["maximumSelectedNovelRecordExactNameCollisionFraction"] == 0.10 and config["identifiabilityGates"]["minimumNovelChoicesWithNoExactNameCollision"] == 3),
        "failure_cannot_reopen_model_or_authority": bool(config["decisionRule"]["failureAuthorizesOnlyTextFreeSemanticallyNoncollidingSourceDesign"] and not config["decisionRule"]["failureAuthorizesModelRerunPromptRevisionOrScaling"] and not config["decisionRule"]["eitherOutcomeAuthorizesProtectedInductionRicherPlanningAPITrainingActionOrExecution"]),
        "code_and_output_hold": all(path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)) and not (PROJECT_ROOT / "outputs/v133-sgd-capability-label-identifiability/evaluation/result.json").exists(),
    }
    passed = all(checks.values())
    audit = {"schema_version": "133-sgd-capability-label-identifiability-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "decision": "freeze_V133_schema_only_audit" if passed else "reject_V133_design", "pre_run_access": {"schema_file_read_count": 0, "dialogue_file_read_count": 0, "language_record_read_count": 0, "model_load_count": 0, "model_generation_count": 0, "actual_execution_count": 0}}
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path, "source_archive": PROJECT_ROOT / config["sourceArchive"], "choice_catalog": PROJECT_ROOT / config["choiceCatalog"], "fixture_population": PROJECT_ROOT / config["fixturePopulation"], "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "133-sgd-capability-label-identifiability-lock", "experiment": config["experiment"], "config_payload": config, "authorization": {"run_one_schema_only_identifiability_audit": True, "modify_collision_rules_gates_population_or_decision": False, "read_dialogue_language_or_model_output": False, "run_model_API_training_authority_or_execution": False}}
    for key, path in deps.items(): lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
