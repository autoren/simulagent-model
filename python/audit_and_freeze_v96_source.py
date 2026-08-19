#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v93_open_set_source import compile_schema


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash({key: value for key, value in payload.items() if key != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v96-two-source-open-set-source.json"
    parent_path = PROJECT_ROOT / "configs/v95-activation-open-set-source-outcome-lock.json"
    architecture_path = PROJECT_ROOT / "configs/v92-structured-llm-architecture-outcome-lock.json"
    authority_path = PROJECT_ROOT / "configs/v87-external-language-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v96-two-source-open-set-source-plan.md"
    protocol_path = PROJECT_ROOT / "python/v96_two_source_open_set_source.py"
    tests_path = PROJECT_ROOT / "python/test_v96_two_source_open_set_source.py"
    runner_path = PROJECT_ROOT / "python/run_v96_source_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v96_source_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v96_source.py"
    audit_path = PROJECT_ROOT / "outputs/v96-two-source-open-set/source-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v96-two-source-open-set-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v96-two-source-open-set/source"
    if audit_path.exists() or lock_path.exists() or source_root.exists():
        raise RuntimeError("V96 source stage is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    architecture = json.loads(architecture_path.read_text())
    authority = json.loads(authority_path.read_text())
    schema_path = PROJECT_ROOT / config["schemaDependency"]["localPath"]
    schema = compile_schema(json.loads(schema_path.read_text()))
    catalog = config["catalogDialogueShard"]
    unsupported = config["unsupportedDialogueShard"]
    catalog_partition = config["catalogPartition"]
    unsupported_partition = config["unsupportedPartition"]
    classes = config["classConstruction"]
    gates = config["sourceGates"]
    exposure = config["preLockExposure"]
    V95_summary = parent["outcome"]["inventory_summary"]
    checks = {
        "V95_negative_outcome_is_exact_and_supports_material_two_source_successor": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_source_feasibility_passed"]
            and V95_summary["class_counts"]["known_unfamiliar"] >= 16
            and V95_summary["class_counts"]["novel_valid"] >= 16
            and V95_summary["unsupported_service_count"] == 0
        ),
        "architecture_and_source_authority_are_exact": bool(
            valid_lock(architecture)
            and valid_lock(authority)
            and architecture["outcome"]["runtime_architecture"]["localLLMRuntimeRole"] == "disabled"
            and authority["outcome"]["passed"]
            and file_sha256(schema_path) == config["schemaDependency"]["localSha256"]
        ),
        "two_fresh_shard_metadata_records_are_exact_and_unmaterialized": bool(
            catalog["role"] == "catalog_only"
            and catalog["path"] == "dev/dialogues_007.json"
            and catalog["byteSize"] == 1381473
            and catalog["gitBlobSha1"] == "5e6e1bb204621cf530228c8a7bd1537b12fc35b2"
            and unsupported["role"] == "unsupported_only"
            and unsupported["path"] == "dev/dialogues_008.json"
            and unsupported["byteSize"] == 4040385
            and unsupported["gitBlobSha1"] == "c80db910d22e1f7db9d576652e9f1e5537a01c58"
            and config["revision"] in catalog["rawUrl"]
            and config["revision"] in unsupported["rawUrl"]
            and not source_root.exists()
        ),
        "catalog_partition_retains_V95_successful_rules": bool(
            catalog_partition["partitionBeforeLanguageDerivedFeatures"]
            and catalog_partition["eligibleServiceMinimumActivationCount"] >= 12
            and catalog_partition["eligiblePairMinimumActivationCount"] >= 8
            and catalog_partition["catalogServiceCount"] == 3
            and catalog_partition["hiddenServiceCount"] == 2
            and catalog_partition["hiddenPairCountPerSelectedService"] == 1
            and catalog_partition["minimumDeclaredPairCount"] >= 3
            and classes["nonNoneClassesRequireSourceIntentActivation"]
            and classes["useCurrentUserTurnOnlyForLexicalSeparation"]
        ),
        "unsupported_partition_is_separate_structural_and_nontrivial": bool(
            unsupported_partition["partitionBeforeLanguageDerivedFeatures"]
            and unsupported_partition["requireDisjointFromCatalogServices"]
            and unsupported_partition["eligibleServiceMinimumActivationCount"] >= 16
            and unsupported_partition["eligiblePairMinimumActivationCount"] >= 8
            and unsupported_partition["unsupportedServiceCount"] == 1
        ),
        "V95_services_are_excluded_from_both_new_roles": set(V95_summary["eligible_fresh_services"]) <= set(config["previouslyExposedServices"]),
        "five_class_count_and_coverage_gates_are_unchanged": bool(
            min(
                gates["minimumKnownFamiliarCandidateCount"],
                gates["minimumKnownUnfamiliarCandidateCount"],
                gates["minimumNovelValidCandidateCount"],
                gates["minimumUnsupportedCandidateCount"],
                gates["minimumInsufficientEvidenceCandidateCount"],
            ) >= 16
            and gates["requiredCatalogServiceCount"] == 3
            and gates["requiredUnsupportedServiceCount"] == 1
            and gates["requiredHiddenPairCount"] == 2
            and gates["requiredNovelServiceCoverage"] == 2
        ),
        "schema_is_typed_and_global": bool(
            len(schema) >= 17
            and all(service["intent_names"] and service["slot_names"] for service in schema.values())
        ),
        "zero_dialogue_manual_model_API_or_training_access": all(
            exposure[key] == 0 for key in (
                "dialoguePayloadAccessCount", "individualUtteranceAccessCount",
                "manualUtteranceInspectionCount", "modelLoadCount", "modelGenerationCount",
                "LLMAPICallCount", "adapterTrainingRunCount",
            )
        ),
        "plan_and_locked_code_exist": all(
            path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "96-two-source-open-set-source-design-audit",
        "experiment": "v96_two_source_open_set_source_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_two_source_inventory" if passed else "reject_V96_source_design",
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
        "parent_architecture_lock": architecture_path,
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
        "schema_version": "96-two-source-open-set-source-lock",
        "experiment": "v96_two_source_open_set_source_lock",
        "config_payload": config,
        "authorization": {
            "modify_source_roles_partition_metadata_or_gates": False,
            "download_and_inventory_both_pinned_shards_once": True,
            "emit_or_manually_inspect_language": False,
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
