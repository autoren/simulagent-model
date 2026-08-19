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
    config_path = PROJECT_ROOT / "configs/v102-presto-context-source.json"
    parent_path = PROJECT_ROOT / "configs/v99-open-world-source-selection-lock.json"
    massive_path = PROJECT_ROOT / "configs/v101-massive-population-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v102-presto-context-source-plan.md"
    protocol_path = PROJECT_ROOT / "python/v102_presto_context_source.py"
    tests_path = PROJECT_ROOT / "python/test_v102_presto_context_source.py"
    runner_path = PROJECT_ROOT / "python/run_v102_presto_source_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v102_presto_source_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v102_presto_source.py"
    audit_path = PROJECT_ROOT / "outputs/v102-presto-context-source/source-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v102-presto-context-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v102-presto-context-source/source"
    inventory_root = PROJECT_ROOT / "outputs/v102-presto-context-source/source-inventory"
    if audit_path.exists() or lock_path.exists() or source_root.exists() or inventory_root.exists():
        raise RuntimeError("V102 PRESTO source stage is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    massive = json.loads(massive_path.read_text())
    archive = config["archive"]
    rule = config["dependencyRule"]
    gates = config["sourceGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V99_selection_is_exact_and_authorizes_PRESTO_inventory_preregistration": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["insufficient_evidence_source"]
            == "PRESTO_v1_en-US_human_context_pairs"
            and parent["authorization"]["preregisterPrestoArchiveInventory"]
            and not parent["authorization"]["downloadEitherPayloadBeforeItsOwnLock"]
        ),
        "V101_MASSIVE_population_is_frozen_without_language_or_model_access": bool(
            valid_lock(massive)
            and massive["outcome"]["passed"]
            and massive["outcome"]["scientific_population_feasibility_passed"]
            and massive["outcome"]["population_summary"]["selected_candidate_count"] == 512
            and not massive["authorization"]["reopen_archive_or_extract_language_before_extraction_lock"]
        ),
        "official_archive_identity_and_members_are_frozen": bool(
            archive["url"] == "https://storage.googleapis.com/gresearch/presto/presto_v1.zip"
            and archive["byteSize"] == 415_990_813
            and archive["etagMd5"] == "5fb5bd7e437a07fbae4991b5b4a573f4"
            and archive["generation"] == "1678604196509246"
            and archive["lastModified"] == "2023-03-12T06:56:36Z"
            and archive["requiredMemberBasenames"]
            == ["presto_dev.jsonl", "presto_test.jsonl"]
        ),
        "human_en_US_only_and_synthetic_context_is_prohibited": bool(
            config["locale"] == "en-US"
            and config["requiredContextProvenance"] == "human"
            and not rule["syntheticContextAllowed"]
            and gates["maximumSyntheticContextCandidateCount"] == 0
        ),
        "dependency_is_mechanical_and_current_input_is_ablated_only_by_context": bool(
            rule["targetArgumentDelimiters"] == ["«", "»"]
            and rule["minimumNormalizedArgumentCharacterCount"] >= 3
            and rule["maximumNormalizedArgumentTokenCount"] <= 8
            and rule["argumentMustBeAbsentFromCurrentInput"]
            and rule["argumentMustOccurAsContiguousNormalizedTokensInContext"]
            and rule["fullAndAblatedPairsShareExampleIdInputAndTarget"]
            and not rule["emitInputTargetArgumentContextOrSeededValues"]
            and set(rule["eligibleContextSources"]) == {
                "previous_turn_user_query", "previous_turn_response_text", "seeded_list_name",
                "seeded_list_item", "seeded_note_name", "seeded_note_text", "seeded_contact",
            }
        ),
        "count_diversity_and_split_gates_are_noncompensatory": bool(
            gates["minimumEligibleDevelopmentCandidateCount"] >= 64
            and gates["minimumEligibleProtectedTestCandidateCount"] >= 64
            and gates["minimumEligibleTotalCandidateCount"] >= 256
            and gates["minimumPreviousTurnDependentCandidateCount"] >= 64
            and gates["minimumSeededStateDependentCandidateCount"] >= 64
            and gates["minimumDependencySourceKindCount"] >= 2
            and gates["minimumSemanticRootFunctionCount"] >= 8
            and gates["requireDevelopmentTestIdentifierDisjointness"]
        ),
        "zero_payload_language_model_API_training_or_side_effect_access": bool(
            exposure["archiveHEADRequestCount"] == 1
            and exposure["officialREADMEAndPaperMetadataInspectionCount"] == 1
            and all(
                exposure[key] == 0
                for key in (
                    "archivePayloadDownloadCount", "languageRecordInspectionCount",
                    "modelLoadCount", "modelGenerationCount", "LLMAPICallCount",
                    "adapterTrainingRunCount",
                )
            )
            and all(
                gates[key] == 0
                for key in (
                    "maximumEmittedLanguageRecordCount", "maximumManualUtteranceInspectionCount",
                    "maximumModelLoadCount", "maximumModelGenerationCount",
                    "maximumLLMAPICallCount", "maximumAdapterTrainingRunCount",
                    "maximumRealServiceCallCount", "maximumExternalSideEffectCount",
                )
            )
        ),
        "pass_does_not_authorize_language_or_model_access": bool(
            not config["decisionRule"]["passAuthorizesSelectedLanguageExtraction"]
            and not config["decisionRule"]["passAuthorizesModelInference"]
            and not config["decisionRule"]["passAuthorizesAPITrainingPosteriorPlanningOrExecution"]
        ),
        "plan_and_locked_code_exist": all(
            path.is_file()
            for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "102-presto-context-source-design-audit",
        "experiment": "v102_presto_context_source_design_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_one_pinned_PRESTO_dev_test_inventory"
            if passed else "reject_V102_PRESTO_source_design"
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
        "config": config_path,
        "parent_source_selection_lock": parent_path,
        "massive_population_outcome": massive_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "102-presto-context-source-lock",
        "experiment": "v102_presto_context_source_lock",
        "config_payload": config,
        "authorization": {
            "modify_source_identity_dependency_rule_or_gates": False,
            "download_and_inventory_pinned_archive_once": True,
            "parse_only_locked_dev_and_test_members": True,
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
