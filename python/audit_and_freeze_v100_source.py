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
    config_path = PROJECT_ROOT / "configs/v100-massive-source.json"
    parent_path = PROJECT_ROOT / "configs/v99-open-world-source-selection-lock.json"
    plan_path = PROJECT_ROOT / "docs/v100-massive-source-plan.md"
    protocol_path = PROJECT_ROOT / "python/v100_massive_source.py"
    tests_path = PROJECT_ROOT / "python/test_v100_massive_source.py"
    runner_path = PROJECT_ROOT / "python/run_v100_source_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v100_source_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v100_source.py"
    audit_path = PROJECT_ROOT / "outputs/v100-massive-open-set/source-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v100-massive-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v100-massive-open-set/source"
    inventory_root = PROJECT_ROOT / "outputs/v100-massive-open-set/source-inventory"
    if audit_path.exists() or lock_path.exists() or source_root.exists() or inventory_root.exists():
        raise RuntimeError("V100 source stage is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    archive = config["archive"]
    roles = config["servicePartition"]
    classes = config["classConstruction"]
    gates = config["sourceGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V99_source_selection_is_exact_and_authorizes_MASSIVE_inventory_preregistration": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["core_source"] == "MASSIVE_1.1_en-US"
            and parent["authorization"]["preregisterMassiveArchiveInventory"]
            and not parent["authorization"]["downloadEitherPayloadBeforeItsOwnLock"]
        ),
        "official_archive_identity_is_frozen": bool(
            archive["url"]
            == "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
            and archive["byteSize"] == 40_251_390
            and archive["etag"] == "51e0da2a3ff7a016f109e1d1b4306e93-3"
            and archive["lastModified"] == "2022-11-07T16:55:04Z"
            and archive["expectedLocaleMemberSuffix"] == "en-US.jsonl"
            and config["locale"] == "en-US"
        ),
        "roles_are_structural_and_selected_before_language_features": bool(
            roles["partitionBeforeUtteranceDerivedFeatures"]
            and roles["eligibleScenarioMinimumRecordCount"] >= 200
            and roles["eligibleIntentMinimumRecordCount"] >= 64
            and roles["unsupportedScenarioCount"] == 1
            and roles["catalogScenarioCount"] == 3
            and roles["hiddenScenarioCount"] == 2
            and roles["hiddenIntentCountPerSelectedScenario"] == 1
            and roles["minimumDeclaredIntentCount"] >= 3
        ),
        "four_classes_and_noncompensatory_partition_gates_are_frozen": bool(
            classes["knownFamiliarMinimumCurrentUtteranceOverlapCount"] == 1
            and classes["knownUnfamiliarMaximumCurrentUtteranceOverlapCount"] == 0
            and classes["novelRequiresHiddenIntentInsideCatalogScenario"]
            and classes["unsupportedRequiresCompletelyWithheldScenario"]
            and not classes["emitRawOrAnnotatedUtteranceTokensOrSlotValues"]
            and gates["minimumClassCandidateCount"] >= 64
            and gates["minimumValidationCandidateCountPerClass"] >= 16
            and gates["minimumTestCandidateCountPerClass"] >= 16
            and gates["minimumKnownClassScenarioCoverage"] >= 2
            and gates["requiredNovelScenarioCoverage"] == 2
            and gates["requiredUnsupportedScenarioCoverage"] == 1
        ),
        "typed_ontology_gates_are_nontrivial": bool(
            gates["minimumScenarioCount"] >= 18
            and gates["minimumIntentCount"] >= 60
            and gates["minimumSlotTypeCount"] >= 20
            and gates["minimumEligibleScenarioCount"] >= 4
        ),
        "zero_payload_language_model_API_training_or_side_effect_access": bool(
            exposure["archiveHEADRequestCount"] == 1
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
        "schema_version": "100-massive-source-design-audit",
        "experiment": "v100_massive_source_design_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_one_pinned_MASSIVE_archive_inventory"
            if passed else "reject_V100_source_design"
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
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "100-massive-source-lock",
        "experiment": "v100_massive_source_lock",
        "config_payload": config,
        "authorization": {
            "modify_source_identity_roles_classes_or_gates": False,
            "download_and_inventory_pinned_archive_once": True,
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
