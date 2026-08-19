#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v98_test_schema_feasibility import service_family


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v98-test-schema-feasibility.json"
    parent_path = PROJECT_ROOT / "configs/v97-aggregate-open-set-source-outcome-lock.json"
    authority_path = PROJECT_ROOT / "configs/v87-external-language-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v98-test-schema-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v98_test_schema_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v98_test_schema_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v98_schema_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v98_schema_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v98_schema.py"
    audit_path = PROJECT_ROOT / "outputs/v98-test-schema-feasibility/schema-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v98-test-schema-feasibility-lock.json"
    source_root = PROJECT_ROOT / "outputs/v98-test-schema-feasibility/source"
    if audit_path.exists() or lock_path.exists() or source_root.exists():
        raise RuntimeError("V98 schema stage is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    authority = json.loads(authority_path.read_text())
    dev_path = PROJECT_ROOT / config["developmentSchemaDependency"]["localPath"]
    development = json.loads(dev_path.read_text())
    development_families = {
        service_family(service["service_name"]) for service in development
    }
    schema = config["testSchema"]
    rule = config["familyRule"]
    gates = config["schemaGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V97_negative_outcome_is_exact_and_proves_dev_service_exhaustion": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_source_feasibility_passed"]
            and parent["outcome"]["inventory_summary"]["eligible_fresh_service_count"] == 0
            and not parent["authorization"]["preregister_dialogue_disjoint_population"]
        ),
        "source_authority_and_development_schema_are_exact": bool(
            valid_lock(authority)
            and authority["outcome"]["passed"]
            and file_sha256(dev_path) == config["developmentSchemaDependency"]["localSha256"]
            and len(development_families) >= 16
        ),
        "test_schema_metadata_is_exact_and_unmaterialized": bool(
            schema["path"] == "test/schema.json"
            and schema["byteSize"] == 54864
            and schema["gitBlobSha1"] == "6cb0fc131da22b43bb4672ec89755f19e7d278ff"
            and config["revision"] in schema["rawUrl"]
            and config["testDialogueShardMetadataCount"] == 34
            and not source_root.exists()
        ),
        "family_freshness_is_stricter_than_versioned_service_identity": bool(
            rule["familyIsPrefixBeforeFinalNumericSuffix"]
            and rule["excludeEveryFamilyPresentInDevelopmentSchema"]
            and rule["minimumTypedIntentCountPerEligibleService"] >= 2
            and rule["minimumSlotCountPerEligibleService"] >= 1
        ),
        "schema_viability_gates_are_nontrivial": bool(
            gates["minimumNovelServiceFamilyCount"] >= 4
            and gates["minimumEligibleNovelServiceCount"] >= 4
            and gates["minimumTestDialogueShardMetadataCount"] >= 34
            and gates["maximumEmittedIntentNameCount"] == 0
            and gates["maximumEmittedIntentDescriptionCount"] == 0
            and gates["maximumEmittedSlotNameCount"] == 0
        ),
        "zero_schema_or_dialogue_payload_model_API_or_training_access": all(
            exposure[key] == 0 for key in (
                "testSchemaPayloadAccessCount", "testDialoguePayloadAccessCount",
                "manualSchemaLanguageInspectionCount", "modelLoadCount", "modelGenerationCount",
                "LLMAPICallCount", "adapterTrainingRunCount",
            )
        ),
        "plan_and_locked_code_exist": all(
            path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "98-test-schema-feasibility-design-audit",
        "experiment": "v98_test_schema_feasibility_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_test_schema_inventory" if passed else "reject_V98_schema_design",
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
        "source_authority_lock": authority_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "98-test-schema-feasibility-lock",
        "experiment": "v98_test_schema_feasibility_lock",
        "config_payload": config,
        "authorization": {
            "modify_schema_metadata_family_rule_or_gates": False,
            "download_and_inventory_pinned_test_schema_once": True,
            "download_test_dialogue_payload": False,
            "emit_intent_slot_or_description_language": False,
            "manually_inspect_schema_or_dialogue_language": False,
            "select_population_or_extract_language": False,
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
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
