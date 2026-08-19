#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v161-fresh-massive-transfer-population.json"
    plan_path = PROJECT_ROOT / "docs/v161-fresh-massive-transfer-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v161_fresh_massive_transfer_population.py"
    tests_path = PROJECT_ROOT / "python/test_v161_fresh_massive_transfer_population.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v161_fresh_massive_transfer_population.py"
    runner_path = PROJECT_ROOT / "python/run_v161_fresh_massive_transfer_population.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v161_fresh_massive_transfer_population_outcome.py"
    audit_path = PROJECT_ROOT / "outputs/v161-fresh-massive-transfer-population/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v161-fresh-massive-transfer-population-lock.json"
    population_dir = PROJECT_ROOT / "outputs/v161-fresh-massive-transfer-population/population"
    if audit_path.exists() or lock_path.exists() or population_dir.exists():
        raise RuntimeError("V161 already preregistered or materialized")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV160OutcomeLock"]
    source_outcome_path = PROJECT_ROOT / config["sourceOutcomeLock"]
    exclusion_outcome_path = PROJECT_ROOT / config["excludedPopulationOutcomeLock"]
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    exclusion_path = PROJECT_ROOT / config["excludedPopulation"]
    parent = json.loads(parent_path.read_text())
    source_outcome = json.loads(source_outcome_path.read_text())
    exclusion_outcome = json.loads(exclusion_outcome_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    exclusion = json.loads(exclusion_path.read_text())
    excluded_ids = {row["candidate_id"] for row in exclusion["selected_population"]}
    remaining = [
        row
        for row in inventory["candidate_index"]
        if row["candidate_id"] not in excluded_ids and row["partition"] in {"validation", "test"}
    ]
    availability = Counter((row["partition"], row["class_label"]) for row in remaining)
    forbidden = {
        "utt",
        "utterance",
        "annot_utt",
        "annotated_utterance",
        "tokens",
        "slot_values",
        "values",
        "text",
        "prompt",
    }
    selection = config["selection"]
    gates = config["populationGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V160_positive_controlled_grammar_outcome_authorizes_fresh_transfer_population_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["policy_qualified"]
            and parent["authorization"]["design_fresh_external_style_transfer_population_if_policy_qualified"]
            and not parent["authorization"]["run_transfer_policy_before_separate_preregistration"]
            and not parent["authorization"]["open_or_score_V159_evaluation"]
            and not parent["authorization"]["run_local_model_or_hybrid"]
            and not parent["authorization"]["run_API_training_induction_authority_action_or_execution"]
        ),
        "V100_source_and_V101_consumed_population_locks_are_exact": bool(
            valid_lock(source_outcome)
            and source_outcome["outcome"]["passed"]
            and source_outcome["outcome"]["scientific_source_feasibility_passed"]
            and valid_lock(exclusion_outcome)
            and exclusion_outcome["outcome"]["passed"]
            and exclusion_outcome["outcome"]["scientific_population_feasibility_passed"]
        ),
        "text_free_source_and_exclusion_identities_are_exact": bool(
            file_sha256(inventory_path) == config["sourceInventorySha256"]
            and inventory["candidate_index_sha256"] == config["sourceCandidateIndexSha256"]
            and not inventory["contains_raw_or_annotated_utterances_tokens_or_slot_values"]
            and not inventory["provenance"]["contains_source_language"]
            and file_sha256(exclusion_path) == config["excludedPopulationFileSha256"]
            and exclusion["selected_population_sha256"] == config["excludedPopulationPayloadSha256"]
            and len(excluded_ids) == gates["requiredExcludedCandidateCount"]
            and not exclusion["contains_language_tokens_slot_values_or_prompts"]
        ),
        "remaining_text_free_pools_satisfy_every_role_class_quota": all(
            availability[(role_spec["sourcePartition"], class_label)]
            >= gates["minimumRemainingCandidateCountPerClassPerRole"]
            for role_spec in selection["roles"].values()
            for class_label in config["requiredClasses"]
        ),
        "source_and_exclusion_rows_contain_no_language_fields": bool(
            all(not (forbidden & set(row)) for row in inventory["candidate_index"])
            and all(not (forbidden & set(row)) for row in exclusion["selected_population"])
        ),
        "roles_hash_exclusion_and_population_gates_are_frozen": bool(
            selection["roles"]["development_transfer"]["sourcePartition"] == "validation"
            and selection["roles"]["protected_transfer"]["sourcePartition"] == "test"
            and selection["excludeEveryV101CandidateBeforeHashSelection"]
            and selection["hashSelectionUsesOnlyCandidateIdentifierClassRoleAndScenario"]
            and selection["selectionBeforeAnyNewSelectedLanguageExtraction"]
            and selection["selectedCandidateCountPerClassPerRole"] == 48
            and len(selection["baseSalt"]) >= 16
            and config["requiredClasses"]
            == ["known_familiar", "known_unfamiliar", "novel_valid", "unsupported"]
            and gates["requiredCandidateCountPerRole"] == 192
            and gates["requiredTotalCandidateCount"] == 384
            and gates["requiredOverlapWithExcludedPopulation"] == 0
            and gates["requireRoleIdentifierDisjointness"]
        ),
        "coverage_gates_are_nontrivial_and_source_compatible": bool(
            gates["requiredKnownScenarioCoverage"] == 3
            and gates["requiredNovelScenarioCoverage"] == 2
            and gates["requiredUnsupportedScenarioCoverage"] == 1
            and gates["minimumIntentCoveragePerClass"]
            == {
                "known_familiar": 6,
                "known_unfamiliar": 8,
                "novel_valid": 2,
                "unsupported": 4,
            }
            and selection["scenarioMinimumPerClass"]["known_familiar"] >= 1
            and selection["scenarioMinimumPerClass"]["known_unfamiliar"] >= 1
            and selection["scenarioMinimumPerClass"]["novel_valid"] >= 1
        ),
        "zero_language_archive_model_API_training_or_execution_access": bool(
            exposure["textFreeCandidateIndexAggregateInspectionCount"] == 1
            and exposure["excludedPopulationAggregateInspectionCount"] == 1
            and exposure["individualCandidateIdentifierInspectionCount"] == 0
            and all(
                exposure[key] == 0
                for key in (
                    "sourceArchiveReopenCount",
                    "selectedLanguageRecordExtractionCount",
                    "manualUtteranceInspectionCount",
                    "modelLoadCount",
                    "modelGenerationCount",
                    "LLMAPICallCount",
                    "trainingRunCount",
                    "actualExecutionCount",
                )
            )
            and all(
                gates[key] == 0
                for key in (
                    "maximumEmittedLanguageRecordCount",
                    "maximumManualUtteranceInspectionCount",
                    "maximumSourceArchiveReopenCount",
                    "maximumModelLoadCount",
                    "maximumModelGenerationCount",
                    "maximumLLMAPICallCount",
                    "maximumTrainingRunCount",
                    "maximumRealServiceCallCount",
                    "maximumExternalSideEffectCount",
                    "maximumActualExecutionCount",
                )
            )
        ),
        "passing_requires_separate_language_extraction_lock": bool(
            config["decisionRule"]["passingAuthorizesOnlySeparateLanguageExtractionPreregistration"]
            and config["decisionRule"]["passingDoesNotAuthorizeImmediateLanguageExtractionInterfaceScoringModelAPITrainingInductionAuthorityActionOrExecution"]
        ),
        "required_files_exist": all(
            path.is_file()
            for path in (
                config_path,
                plan_path,
                protocol_path,
                tests_path,
                auditor_path,
                runner_path,
                verifier_path,
                parent_path,
                source_outcome_path,
                exclusion_outcome_path,
                inventory_path,
                exclusion_path,
            )
        ),
    }
    passed = all(checks.values())
    decision = (
        "freeze_and_authorize_one_text_free_fresh_transfer_population_selection"
        if passed
        else "reject_V161_transfer_population_design"
    )
    audit = {
        "schema_version": "161-fresh-massive-transfer-population-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": decision,
        "checks": checks,
        "remaining_pool_counts": {
            f"{partition}::{class_label}": count
            for (partition, class_label), count in sorted(availability.items())
        },
        "prelock_access": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_V160_outcome": parent_path,
        "source_outcome": source_outcome_path,
        "exclusion_outcome": exclusion_outcome_path,
        "source_inventory": inventory_path,
        "excluded_population": exclusion_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "auditor": auditor_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "161-fresh-massive-transfer-population-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "select_and_emit_one_text_free_fresh_transfer_population": True,
            "modify_salt_exclusions_quotas_roles_or_gates": False,
            "reopen_source_archive_or_extract_selected_language": False,
            "read_protected_transfer_language": False,
            "manually_inspect_language": False,
            "run_interface_policy_model_hybrid_API_training_induction_authority_action_or_execution": False,
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
