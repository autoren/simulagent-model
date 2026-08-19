#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash(
        {key: value for key, value in payload.items() if key != "lock_payload_sha256"}
    ) == payload.get("lock_payload_sha256")


def main() -> None:
    repair_path = PROJECT_ROOT / "configs/v102r1-presto-parser-repair.json"
    parent_path = PROJECT_ROOT / "configs/v102-presto-context-source-technical-outcome-lock.json"
    scientific_path = PROJECT_ROOT / "configs/v102-presto-context-source.json"
    plan_path = PROJECT_ROOT / "docs/v102r1-presto-parser-repair-plan.md"
    protocol_path = PROJECT_ROOT / "python/v102r1_presto_context_source.py"
    tests_path = PROJECT_ROOT / "python/test_v102r1_presto_context_source.py"
    runner_path = PROJECT_ROOT / "python/run_v102r1_presto_source_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v102r1_presto_source_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v102r1_presto_source.py"
    audit_path = PROJECT_ROOT / "outputs/v102r1-presto-context-source/source-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v102r1-presto-context-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v102r1-presto-context-source/source"
    inventory_root = PROJECT_ROOT / "outputs/v102r1-presto-context-source/source-inventory"
    if audit_path.exists() or lock_path.exists() or source_root.exists() or inventory_root.exists():
        raise RuntimeError("V102r1 source stage is already frozen or materialized")

    repair = json.loads(repair_path.read_text())
    parent = json.loads(parent_path.read_text())
    scientific = json.loads(scientific_path.read_text())
    policy = repair["repairPolicy"]
    exposure = repair["preLockExposure"]
    checks = {
        "V102_technical_outcome_is_exact_and_authorizes_one_repair_retrieval": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_source_feasibility_evaluated"]
            and parent["outcome"]["failure_class"]
            == "technical_parser_schema_validation_failure"
            and parent["authorization"]["preregister_V102r1_parser_repair"]
            and parent["authorization"]["retain_exact_source_dependency_rule_and_scientific_gates"]
            and parent["authorization"]["authorize_one_fresh_archive_retrieval_because_no_artifact_persisted"]
        ),
        "scientific_config_is_byte_exact_and_unchanged": bool(
            file_sha256(scientific_path) == repair["unchangedScientificConfigSha256"]
            and scientific["archive"]["byteSize"] == 415_990_813
            and scientific["sourceGates"]["minimumEligibleTotalCandidateCount"] == 256
            and scientific["sourceGates"]["minimumPreviousTurnDependentCandidateCount"] == 64
            and scientific["sourceGates"]["minimumSeededStateDependentCandidateCount"] == 64
        ),
        "repair_is_limited_to_optional_context_leaf_tolerance": bool(
            policy["requireRecordMetadataAndContextContainers"]
            and policy["acceptNullContextContainerAsEmpty"]
            and policy["useOnlyStringContextLeaves"]
            and policy["ignoreNonStringOptionalContextLeaves"]
            and policy["neverCoerceNonStringLeavesToStrings"]
            and policy["neverEmitContextLeavesOrTheirHashes"]
            and policy["preserveTargetArgumentCurrentInputAndContiguousContextMatchRule"]
            and policy["preserveEveryScientificCountDiversityAndSplitGate"]
            and policy["persistVerifiedArchiveBeforeParsing"]
        ),
        "prior_access_is_honest_and_repair_has_no_prelock_payload_or_language_access": bool(
            exposure["priorV102ArchiveDownloadCount"] == 1
            and exposure["priorV102LanguageRecordAutomaticParseCompleted"]
            and exposure["priorV102EmittedLanguageRecordCount"] == 0
            and exposure["priorV102ManualUtteranceInspectionCount"] == 0
            and all(
                exposure[key] == 0
                for key in (
                    "V102r1ArchivePayloadDownloadCount", "V102r1LanguageRecordInspectionCount",
                    "modelLoadCount", "modelGenerationCount", "LLMAPICallCount",
                    "adapterTrainingRunCount",
                )
            )
        ),
        "pass_still_does_not_authorize_language_or_model_access": not (
            repair["decisionRule"]["passAuthorizesLanguageExtractionOrModelInference"]
        ),
        "plan_and_locked_code_exist": all(
            path.is_file()
            for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "102r1-presto-parser-repair-design-audit",
        "experiment": "v102r1_presto_parser_repair_design_audit",
        "passed": passed,
        "decision": (
            "freeze_parser_repair_and_authorize_one_fresh_archive_inventory"
            if passed else "reject_V102r1_parser_repair"
        ),
        "checks": checks,
        "prelock_access": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "repair_config": repair_path,
        "parent_technical_outcome": parent_path,
        "scientific_config": scientific_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "102r1-presto-context-source-lock",
        "experiment": "v102r1_presto_context_source_lock",
        "repair_config_payload": repair,
        "scientific_config_payload": scientific,
        "authorization": {
            "modify_source_dependency_rule_scientific_gates_or_repair_policy": False,
            "download_persist_and_inventory_pinned_archive_once": True,
            "emit_or_manually_inspect_language": False,
            "select_population_or_extract_selected_language": False,
            "load_local_or_API_model": False,
            "train_adapter_or_learn_likelihood": False,
            "grant_model_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({
        "lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "sha256": file_sha256(lock_path),
    }, indent=2))


if __name__ == "__main__":
    main()
