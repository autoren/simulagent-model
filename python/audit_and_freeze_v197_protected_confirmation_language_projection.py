#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v197-protected-confirmation-language-projection.json"
    plan_path = PROJECT_ROOT / "docs/v197-protected-confirmation-language-projection-plan.md"
    protocol_path = PROJECT_ROOT / "python/v197_protected_confirmation_language_projection.py"
    tests_path = PROJECT_ROOT / "python/test_v197_protected_confirmation_language_projection.py"
    runner_path = PROJECT_ROOT / "python/run_v197_protected_confirmation_language_projection.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v197_protected_confirmation_language_projection_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v197_protected_confirmation_language_projection.py"
    audit_path = PROJECT_ROOT / "outputs/v197-protected-confirmation-language-projection/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v197-protected-confirmation-language-projection-lock.json"
    output_root = PROJECT_ROOT / "outputs/v197-protected-confirmation-language-projection/projection"
    outcome_path = PROJECT_ROOT / "configs/v197-protected-confirmation-language-projection-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V197 is already preregistered, run, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV196OutcomeLock"]
    v184_path = PROJECT_ROOT / config["sourceV184OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v184 = json.loads(v184_path.read_text())
    inputs = {
        "sealed_protected_language": PROJECT_ROOT / config["sealedProtectedLanguage"],
        "confirmation_identities": PROJECT_ROOT / config["confirmationIdentities"],
        "hidden_targets": PROJECT_ROOT / config["hiddenTargets"],
        "roadmap": PROJECT_ROOT / config["roadmap"],
    }
    decision = config["decisionRule"]
    pre = config["preLockExposure"]
    checks = {
        "V196_is_valid_and_authorizes_separate_unchanged_policy_confirmation_design": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_unchanged_V195_policy_confirmation_only"]
            and not parent["authorization"]["open_protected_language_or_run_model_without_separate_lock"]
        ),
        "V184_and_all_projection_inputs_are_exact": bool(
            valid_lock(v184) and v184["outcome"]["passed"]
            and file_sha256(inputs["sealed_protected_language"]) == v184["protected_language_sha256"]
            and file_sha256(inputs["confirmation_identities"]) == parent["confirmation_identities_sha256"]
            and file_sha256(inputs["hidden_targets"]) == parent["hidden_targets_sha256"]
        ),
        "projection_schema_and_counts_are_fixed": bool(
            config["projection"]["requiredInputRecordCount"] == 132
            and config["projection"]["requiredSelectedRecordCount"] == 125
            and config["projection"]["requiredSelectedObservedCount"] == 113
            and config["projection"]["requiredSelectedMissingCount"] == 12
            and config["projection"]["requiredUnselectedReadButNotEmittedCount"] == 7
            and config["projection"]["selectionUsesOnlyFrozenV196OpaqueRecordIds"]
            and config["projection"]["noRecordMayBeSelectedOrExcludedByLanguage"]
        ),
        "prelock_utterance_scoring_model_and_execution_access_is_zero": bool(
            pre["protectedUtteranceReadOrEmissionCount"] == 0
            and pre["manualLanguageInspectionCount"] == 0
            and pre["policyScoreCount"] == 0
            and pre["modelLoadCount"] == 0
            and pre["modelGenerationCount"] == 0
            and pre["APICallCount"] == 0
            and pre["trainingRunCount"] == 0
            and pre["actualExecutionCount"] == 0
        ),
        "immediate_model_changes_and_authority_remain_closed": bool(
            not decision["passAuthorizesImmediateModelRun"]
            and not decision["passAuthorizesPromptModelBudgetParserCostOrGateChange"]
            and not decision["passAuthorizesAPITrainingRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, plan_path, protocol_path, tests_path, runner_path, verifier_path,
                auditor_path, parent_path, v184_path, *inputs.values(),
            )) and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "197-protected-confirmation-language-projection-design-audit",
        "experiment": config["experiment"], "passed": passed,
        "decision": "freeze_and_authorize_one_V197_exact_projection" if passed else "reject_V197_design",
        "checks": checks, "prelock_exposure": pre,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_V196_outcome": parent_path, "source_V184_outcome": v184_path,
        **inputs, "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "197-protected-confirmation-language-projection-lock",
        "experiment": config["experiment"], "config_payload": config,
        "authorization": {
            "modify_selected_IDs_projection_schema_or_counts": False,
            "run_exact_projection_once": True,
            "score_language_or_run_model_API_training": False,
            "registration_authority_action_or_execution": False,
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
