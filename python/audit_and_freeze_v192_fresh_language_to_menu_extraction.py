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
    config_path = PROJECT_ROOT / "configs/v192-fresh-language-to-menu-extraction.json"
    plan_path = PROJECT_ROOT / "docs/v192-fresh-language-to-menu-extraction-plan.md"
    protocol_path = PROJECT_ROOT / "python/v192_fresh_language_to_menu_extraction.py"
    tests_path = PROJECT_ROOT / "python/test_v192_fresh_language_to_menu_extraction.py"
    runner_path = PROJECT_ROOT / "python/run_v192_fresh_language_to_menu_extraction.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v192_fresh_language_to_menu_extraction_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v192_fresh_language_to_menu_extraction.py"
    audit_path = PROJECT_ROOT / "outputs/v192-fresh-language-to-menu-extraction/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v192-fresh-language-to-menu-extraction-lock.json"
    output_root = PROJECT_ROOT / "outputs/v192-fresh-language-to-menu-extraction/extraction"
    outcome_path = PROJECT_ROOT / "configs/v192-fresh-language-to-menu-extraction-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V192 is already preregistered, evaluated, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV191OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    population_lock = json.loads((PROJECT_ROOT / parent["population_lock"]).read_text())
    inputs = {
        "source_archive": PROJECT_ROOT / config["sourceArchive"],
        "development_identities": PROJECT_ROOT / config["developmentIdentities"],
        "hidden_targets": PROJECT_ROOT / config["hiddenTargets"],
    }
    pre = config["preLockExposure"]
    decision = config["decisionRule"]
    observable = config["observableRecordContract"]
    checks = {
        "parent_V191_and_population_lock_are_valid": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_separate_exact_development_language_extraction_only"]
            and valid_lock(population_lock)
        ),
        "source_and_frozen_population_artifacts_are_exact": bool(
            file_sha256(inputs["source_archive"]) == population_lock["source_archive_sha256"]
            and file_sha256(inputs["development_identities"]) == parent["development_identities_sha256"]
            and file_sha256(inputs["hidden_targets"]) == parent["hidden_targets_sha256"]
        ),
        "observable_projection_excludes_gold_fields": bool(
            observable["recordFields"] == ["record_id", "role", "observation_available", "conversation"]
            and observable["conversationTurnFields"] == ["speaker", "utterance"]
            and "target_contract_id" in observable["forbiddenRecordFields"]
            and "source_candidate_id" in observable["forbiddenRecordFields"]
            and "truth_kind" in observable["forbiddenRecordFields"]
        ),
        "prelock_language_policy_model_and_execution_access_is_zero": all(value == 0 for value in pre.values()),
        "successor_authority_is_closed": bool(
            not decision["passAuthorizesImmediateInterfaceScoring"]
            and not decision["passAuthorizesModelOrAPIRun"]
            and not decision["passAuthorizesProtectedAccessRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    parent_path,
                    plan_path,
                    protocol_path,
                    tests_path,
                    runner_path,
                    verifier_path,
                    auditor_path,
                    *inputs.values(),
                )
            )
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "192-fresh-language-to-menu-extraction-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_exact_V192_extraction" if passed else "reject_V192_design",
        "checks": checks,
        "prelock_exposure": pre,
        "selected_conversation_read_count": 0,
        "protected_language_read_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_V191_outcome": parent_path,
        "parent_V191_population_lock": PROJECT_ROOT / parent["population_lock"],
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
        "schema_version": "192-fresh-language-to-menu-extraction-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_projection_records_gates_or_decision": False,
            "run_exact_unprotected_extraction_once": True,
            "read_protected_language_or_score_interface": False,
            "run_model_API_or_training": False,
            "register_prune_mutate_call_service_act_or_execute": False,
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
