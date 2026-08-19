#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v196-protected-confirmation-role-binding.json"
    plan_path = PROJECT_ROOT / "docs/v196-protected-confirmation-role-binding-plan.md"
    protocol_path = PROJECT_ROOT / "python/v196_protected_confirmation_role_binding.py"
    tests_path = PROJECT_ROOT / "python/test_v196_protected_confirmation_role_binding.py"
    runner_path = PROJECT_ROOT / "python/run_v196_protected_confirmation_role_binding.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v196_protected_confirmation_role_binding_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v196_protected_confirmation_role_binding.py"
    audit_path = PROJECT_ROOT / "outputs/v196-protected-confirmation-role-binding/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v196-protected-confirmation-role-binding-lock.json"
    output_root = PROJECT_ROOT / "outputs/v196-protected-confirmation-role-binding/binding"
    outcome_path = PROJECT_ROOT / "configs/v196-protected-confirmation-role-binding-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V196 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV195OutcomeLock"]
    v183_path = PROJECT_ROOT / config["sourceV183OutcomeLock"]
    v184_path = PROJECT_ROOT / config["sourceV184OutcomeLock"]
    v191_path = PROJECT_ROOT / config["sourceV191OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v183 = json.loads(v183_path.read_text())
    v184 = json.loads(v184_path.read_text())
    v191 = json.loads(v191_path.read_text())
    v191_lock_path = PROJECT_ROOT / v191["population_lock"]
    v191_lock = json.loads(v191_lock_path.read_text())
    inputs = {
        "source_inventory": PROJECT_ROOT / config["sourceInventory"],
        "contract_catalog": PROJECT_ROOT / config["contractCatalog"],
        "V183_hidden_identifiability": PROJECT_ROOT / config["V183HiddenIdentifiability"],
        "V191_hidden_targets": PROJECT_ROOT / config["V191HiddenTargets"],
        "sealed_protected_language": PROJECT_ROOT / config["sealedProtectedLanguage"],
        "roadmap": PROJECT_ROOT / config["roadmap"],
    }
    decision = config["decisionRule"]
    checks = {
        "V195_passed_and_authorizes_separate_confirmation_design_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_qualification_gates_passed"]
            and parent["authorization"]["preregister_separate_confirmation_design_only"]
            and not parent["authorization"]["run_confirmation_API_additional_model_or_protected_access_without_new_lock"]
        ),
        "V183_V184_and_V191_sources_are_valid": bool(
            valid_lock(v183) and v183["outcome"]["passed"]
            and valid_lock(v184) and v184["outcome"]["passed"]
            and valid_lock(v191) and v191["outcome"]["passed"]
            and valid_lock(v191_lock)
        ),
        "metadata_and_hidden_inputs_match_frozen_sources": bool(
            file_sha256(inputs["source_inventory"]) == v191_lock["source_inventory_sha256"]
            and file_sha256(inputs["contract_catalog"]) == v183["contract_catalog_sha256"]
            and file_sha256(inputs["V183_hidden_identifiability"]) == v183["hidden_identifiability_sha256"]
            and file_sha256(inputs["V191_hidden_targets"]) == v191["hidden_targets_sha256"]
        ),
        "protected_language_artifact_is_still_exact_but_not_opened": bool(
            inputs["sealed_protected_language"].is_file()
            and file_sha256(inputs["sealed_protected_language"]) == v184["protected_language_sha256"]
            and config["preLockExposure"]["protectedUtteranceReadOrEmissionCount"] == 0
            and config["preLockExposure"]["manualLanguageInspectionCount"] == 0
        ),
        "selection_rule_is_metadata_only_and_dialogue_isolated": bool(
            config["sourceDecision"]["usePreV195SealedProtectedRole"]
            and config["sourceDecision"]["excludeEveryDialogueAppearingInV183Development"]
            and config["sourceDecision"]["excludeEveryDialogueAppearingInV191"]
            and config["sourceDecision"]["selectAtMostOneRecordPerDialogue"]
            and config["sourceDecision"]["selectionUsesOnlyFrozenMetadataIdentifiersAndSalt"]
            and config["sourceDecision"]["selectionUsesNoUtteranceSlotFramePredictionScoreOrOutcome"]
        ),
        "immediate_language_model_and_authority_paths_are_closed": bool(
            not decision["passAuthorizesImmediateProtectedLanguageReadOrModelRun"]
            and not decision["passAuthorizesPromptModelBudgetParserCostOrGateChange"]
            and not decision["passAuthorizesAPITrainingRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, plan_path, protocol_path, tests_path, runner_path, verifier_path,
                auditor_path, parent_path, v183_path, v184_path, v191_path, v191_lock_path,
                *inputs.values(),
            ))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "196-protected-confirmation-role-binding-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V196_text_free_binding" if passed else "reject_V196_design",
        "checks": checks,
        "prelock_exposure": config["preLockExposure"],
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_V195_outcome": parent_path,
        "source_V183_outcome": v183_path,
        "source_V184_outcome": v184_path,
        "source_V191_outcome": v191_path,
        "source_V191_lock": v191_lock_path,
        **inputs,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "196-protected-confirmation-role-binding-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_source_selection_counts_or_freshness_rules": False,
            "run_exact_text_free_binding_once": True,
            "open_or_score_protected_language_or_run_model": False,
            "run_API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
