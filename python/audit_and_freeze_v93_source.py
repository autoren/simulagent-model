#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    unhashed = {key: value for key, value in payload.items() if key != "lock_payload_sha256"}
    return payload_hash(unhashed) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v93-controlled-open-set-source.json"
    parent_path = PROJECT_ROOT / "configs/v92-structured-llm-architecture-outcome-lock.json"
    authority_path = PROJECT_ROOT / "configs/v87-external-language-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v93-controlled-open-set-source-plan.md"
    protocol_path = PROJECT_ROOT / "python/v93_open_set_source.py"
    tests_path = PROJECT_ROOT / "python/test_v93_open_set_source.py"
    runner_path = PROJECT_ROOT / "python/run_v93_source_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v93_source_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v93_source.py"
    audit_path = PROJECT_ROOT / "outputs/v93-controlled-open-set/source-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v93-controlled-open-set-source-lock.json"
    materialized_root = PROJECT_ROOT / "outputs/v93-controlled-open-set/source"
    if audit_path.exists() or lock_path.exists() or materialized_root.exists():
        raise RuntimeError("V93 source design is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    authority = json.loads(authority_path.read_text())
    schema = config["schemaDependency"]
    shard = config["newDialogueShard"]
    exposure = config["preLockExposure"]
    class_rules = config["classConstruction"]
    source_gates = config["sourceGates"]
    prior_shards = {
        "dev/dialogues_001.json",
        "dev/dialogues_002.json",
        "dev/dialogues_003.json",
    }
    schema_path = PROJECT_ROOT / schema["localPath"]
    checks = {
        "parent_architecture_lock_is_exact_and_model_free": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["runtime_architecture"]["localLLMRuntimeRole"] == "disabled"
            and not parent["authorization"]["use_any_local_or_API_model_in_runtime_decision_path"]
        ),
        "source_authority_and_schema_are_exact": bool(
            valid_lock(authority)
            and authority["outcome"]["passed"]
            and schema["path"] == "dev/schema.json"
            and schema["gitBlobSha1"] == authority["source_files"]["dev/schema.json"]["git_blob_sha1"]
            and schema_path.is_file()
            and file_sha256(schema_path) == schema["localSha256"]
        ),
        "new_shard_metadata_is_exact_fresh_and_unmaterialized": bool(
            shard["path"] == "dev/dialogues_004.json"
            and shard["path"] not in prior_shards
            and shard["byteSize"] == 2794855
            and shard["gitBlobSha1"] == "81329efc5ef7169f455cd5f1a3a47d1de6fc419b"
            and config["revision"] in shard["rawUrl"]
            and not materialized_root.exists()
        ),
        "five_classes_and_hash_rules_are_frozen": bool(
            class_rules["minimumTypedIntentCountPerService"] >= 3
            and class_rules["minimumDeclaredIntentCountPerService"] >= 2
            and class_rules["minimumSourceIntentRecordCountForHiding"] >= 8
            and class_rules["noneMapsOnlyToInsufficientEvidence"]
            and class_rules["unsupportedRequiresDifferentSourceAndTargetService"]
            and class_rules["unsupportedRequiresSourceIntentAbsentFromCompleteTargetSchema"]
            and not class_rules["emitLanguageTokensSlotValuesOrHistories"]
        ),
        "previously_exposed_services_are_excluded": set(config["excludedServices"]) == {
            "Alarm_1", "Buses_1", "Flights_3", "Homes_1", "RentalCars_1",
            "Restaurants_2", "RideSharing_1", "Services_4", "Weather_1",
        },
        "source_gates_are_noncompensatory": bool(
            source_gates["minimumEligibleServiceCount"] >= 3
            and min(
                source_gates["minimumKnownFamiliarCandidateCount"],
                source_gates["minimumKnownUnfamiliarCandidateCount"],
                source_gates["minimumNovelValidCandidateCount"],
                source_gates["minimumUnsupportedCandidateCount"],
            ) >= 32
            and source_gates["minimumInsufficientEvidenceCandidateCount"] >= 24
            and source_gates["minimumClassServiceCoverage"] >= 2
        ),
        "zero_payload_manual_model_API_or_training_access": all(
            exposure[key] == 0
            for key in (
                "dialoguePayloadAccessCount", "individualUtteranceAccessCount",
                "manualUtteranceInspectionCount", "modelLoadCount", "modelGenerationCount",
                "LLMAPICallCount", "adapterTrainingRunCount",
            )
        ),
        "plan_and_locked_code_exist": all(
            path.is_file()
            for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "93-controlled-open-set-source-design-audit",
        "experiment": "v93_controlled_open_set_source_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_text_free_source_inventory" if passed else "reject_V93_source_design",
        "checks": checks,
        "prelock_access": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    lock = {
        "schema_version": "93-controlled-open-set-source-lock",
        "experiment": "v93_controlled_open_set_source_lock",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "config_payload": config,
        "parent_architecture_lock": str(parent_path.relative_to(PROJECT_ROOT)),
        "parent_architecture_lock_sha256": file_sha256(parent_path),
        "source_authority_lock": str(authority_path.relative_to(PROJECT_ROOT)),
        "source_authority_lock_sha256": file_sha256(authority_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "protocol": str(protocol_path.relative_to(PROJECT_ROOT)),
        "protocol_sha256": file_sha256(protocol_path),
        "tests": str(tests_path.relative_to(PROJECT_ROOT)),
        "tests_sha256": file_sha256(tests_path),
        "runner": str(runner_path.relative_to(PROJECT_ROOT)),
        "runner_sha256": file_sha256(runner_path),
        "verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "verifier_sha256": file_sha256(verifier_path),
        "auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_source_rules_metadata_or_gates": False,
            "download_and_inventory_pinned_shard_once": True,
            "emit_or_manually_inspect_language": False,
            "select_population_or_extract_language": False,
            "load_local_or_API_model": False,
            "train_adapter_or_learn_likelihood": False,
            "grant_model_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
