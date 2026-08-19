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
    config_path = PROJECT_ROOT / "configs/v101-massive-population.json"
    parent_path = PROJECT_ROOT / "configs/v100-massive-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v101-massive-population-plan.md"
    protocol_path = PROJECT_ROOT / "python/v101_massive_population.py"
    tests_path = PROJECT_ROOT / "python/test_v101_massive_population.py"
    runner_path = PROJECT_ROOT / "python/run_v101_population_selection.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v101_population_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v101_population.py"
    audit_path = PROJECT_ROOT / "outputs/v101-massive-population/population-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v101-massive-population-lock.json"
    population_root = PROJECT_ROOT / "outputs/v101-massive-population/population"
    if audit_path.exists() or lock_path.exists() or population_root.exists():
        raise RuntimeError("V101 population stage is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    inventory_path = PROJECT_ROOT / config["sourceInventory"]
    inventory = json.loads(inventory_path.read_text())
    selection = config["selection"]
    gates = config["populationGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V100_positive_source_outcome_is_exact_and_authorizes_population_selection": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_source_feasibility_passed"]
            and parent["authorization"]["preregister_hash_selected_validation_and_test_population"]
            and parent["authorization"]["select_population_before_language_extraction"]
            and not parent["authorization"]["extract_selected_language_before_population_lock"]
        ),
        "text_free_source_inventory_identity_is_exact": bool(
            file_sha256(inventory_path) == config["sourceInventorySha256"]
            and inventory["candidate_index_sha256"] == config["sourceCandidateIndexSha256"]
            and not inventory["contains_raw_or_annotated_utterances_tokens_or_slot_values"]
            and not inventory["provenance"]["contains_source_language"]
        ),
        "two_source_partitions_and_hash_selection_are_frozen": bool(
            selection["roles"]["development"]["sourcePartition"] == "validation"
            and selection["roles"]["protected_test"]["sourcePartition"] == "test"
            and selection["hashSelectionUsesOnlyCandidateIdentifierClassSplitAndScenario"]
            and selection["selectionBeforeAnySelectedLanguageExtraction"]
            and selection["selectedCandidateCountPerClassPerSplit"] == 64
            and len(selection["baseSalt"]) >= 16
        ),
        "scenario_balancing_and_four_classes_are_nontrivial": bool(
            config["requiredClasses"]
            == ["known_familiar", "known_unfamiliar", "novel_valid", "unsupported"]
            and selection["scenarioMinimumPerClass"]["known_familiar"] >= 12
            and selection["scenarioMinimumPerClass"]["known_unfamiliar"] >= 12
            and selection["scenarioMinimumPerClass"]["novel_valid"] >= 16
        ),
        "population_size_coverage_and_disjointness_gates_are_frozen": bool(
            gates["requiredCandidateCountPerClassPerSplit"] == 64
            and gates["requiredCandidateCountPerSplit"] == 256
            and gates["requiredTotalCandidateCount"] == 512
            and gates["requiredKnownScenarioCoverage"] == 3
            and gates["requiredNovelScenarioCoverage"] == 2
            and gates["requiredUnsupportedScenarioCoverage"] == 1
            and gates["requireDevelopmentTestIdentifierDisjointness"]
            and gates["maximumTrainPartitionCandidateCount"] == 0
            and gates["minimumIntentCoveragePerClass"] == {
                "known_familiar": 6, "known_unfamiliar": 8,
                "novel_valid": 2, "unsupported": 4,
            }
        ),
        "zero_language_archive_model_API_training_or_side_effect_access": bool(
            exposure["textFreeCandidateIndexAggregateInspectionCount"] == 1
            and exposure["individualCandidateIdentifierInspectionCount"] == 0
            and all(
                exposure[key] == 0
                for key in (
                    "sourceArchiveReopenCount", "selectedLanguageRecordExtractionCount",
                    "manualUtteranceInspectionCount", "modelLoadCount", "modelGenerationCount",
                    "LLMAPICallCount", "adapterTrainingRunCount",
                )
            )
            and all(
                gates[key] == 0
                for key in (
                    "maximumEmittedLanguageRecordCount", "maximumManualUtteranceInspectionCount",
                    "maximumSourceArchiveReopenCount", "maximumModelLoadCount",
                    "maximumModelGenerationCount", "maximumLLMAPICallCount",
                    "maximumAdapterTrainingRunCount", "maximumRealServiceCallCount",
                    "maximumExternalSideEffectCount",
                )
            )
        ),
        "pass_requires_a_separate_language_and_model_design": bool(
            not config["decisionRule"]["passAuthorizesImmediateLanguageExtraction"]
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
        "schema_version": "101-massive-population-design-audit",
        "experiment": "v101_massive_population_design_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_one_text_free_population_selection"
            if passed else "reject_V101_population_design"
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
        "parent_source_outcome": parent_path,
        "source_inventory": inventory_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "101-massive-population-lock",
        "experiment": "v101_massive_population_lock",
        "config_payload": config,
        "authorization": {
            "modify_salt_quotas_roles_or_gates": False,
            "select_and_emit_text_free_population_once": True,
            "reopen_source_archive_or_extract_selected_language": False,
            "manually_inspect_language": False,
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
