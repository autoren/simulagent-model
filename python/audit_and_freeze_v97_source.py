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
    config_path = PROJECT_ROOT / "configs/v97-aggregate-open-set-source.json"
    parent_path = PROJECT_ROOT / "configs/v96-two-source-open-set-source-outcome-lock.json"
    architecture_path = PROJECT_ROOT / "configs/v92-structured-llm-architecture-outcome-lock.json"
    authority_path = PROJECT_ROOT / "configs/v87-external-language-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v97-aggregate-open-set-source-plan.md"
    protocol_path = PROJECT_ROOT / "python/v97_aggregate_open_set_source.py"
    tests_path = PROJECT_ROOT / "python/test_v97_aggregate_open_set_source.py"
    runner_path = PROJECT_ROOT / "python/run_v97_source_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v97_source_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v97_source.py"
    audit_path = PROJECT_ROOT / "outputs/v97-aggregate-open-set/source-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v97-aggregate-open-set-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v97-aggregate-open-set/source"
    if audit_path.exists() or lock_path.exists() or source_root.exists():
        raise RuntimeError("V97 source stage is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    architecture = json.loads(architecture_path.read_text())
    authority = json.loads(authority_path.read_text())
    schema_path = PROJECT_ROOT / config["schemaDependency"]["localPath"]
    schema = compile_schema(json.loads(schema_path.read_text()))
    shards = config["aggregateDialogueShards"]
    partition = config["servicePartition"]
    classes = config["classConstruction"]
    gates = config["sourceGates"]
    exposure = config["preLockExposure"]
    parent_summary = parent["outcome"]["inventory_summary"]
    expected_paths = [f"dev/dialogues_{index:03d}.json" for index in range(9, 21)]
    checks = {
        "V96_negative_outcome_is_exact_and_identifies_shard_concentration": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_source_feasibility_passed"]
            and parent_summary["catalog_service_count"] == 1
            and parent_summary["unsupported_service_count"] == 0
            and not parent["authorization"]["preregister_dialogue_disjoint_population"]
        ),
        "architecture_and_source_authority_are_exact": bool(
            valid_lock(architecture)
            and valid_lock(authority)
            and architecture["outcome"]["runtime_architecture"]["localLLMRuntimeRole"] == "disabled"
            and authority["outcome"]["passed"]
            and file_sha256(schema_path) == config["schemaDependency"]["localSha256"]
        ),
        "complete_remaining_untouched_pool_is_exact_and_unmaterialized": bool(
            config["aggregateDialogueShardCount"] == 12
            and len(shards) == 12
            and [shard["path"] for shard in shards] == expected_paths
            and len({shard["path"] for shard in shards}) == 12
            and len({shard["gitBlobSha1"] for shard in shards}) == 12
            and sum(shard["byteSize"] for shard in shards) == config["aggregateDialogueByteSize"] == 45767156
            and all(config["revision"] in shard["rawUrl"] for shard in shards)
            and not source_root.exists()
        ),
        "service_level_partition_is_structural_and_nontrivial": bool(
            partition["partitionBeforeLanguageDerivedFeatures"]
            and partition["eligibleServiceMinimumActivationCount"] >= 12
            and partition["eligiblePairMinimumActivationCount"] >= 8
            and partition["unsupportedServiceCount"] == 1
            and partition["catalogServiceCount"] == 3
            and partition["hiddenServiceCount"] == 2
            and partition["hiddenPairCountPerSelectedService"] == 1
            and partition["minimumDeclaredPairCount"] >= 3
            and classes["nonNoneClassesRequireSourceIntentActivation"]
            and classes["useCurrentUserTurnOnlyForLexicalSeparation"]
        ),
        "V96_catalog_service_is_excluded": set(parent_summary["catalog_services"]) <= set(config["previouslyExposedServices"]),
        "five_class_count_and_coverage_gates_are_retained": bool(
            min(
                gates["minimumKnownFamiliarCandidateCount"],
                gates["minimumKnownUnfamiliarCandidateCount"],
                gates["minimumNovelValidCandidateCount"],
                gates["minimumUnsupportedCandidateCount"],
                gates["minimumInsufficientEvidenceCandidateCount"],
            ) >= 16
            and gates["minimumEligibleFreshServiceCount"] >= 4
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
        "schema_version": "97-aggregate-open-set-source-design-audit",
        "experiment": "v97_aggregate_open_set_source_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_aggregate_source_inventory" if passed else "reject_V97_source_design",
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
        "schema_version": "97-aggregate-open-set-source-lock",
        "experiment": "v97_aggregate_open_set_source_lock",
        "config_payload": config,
        "authorization": {
            "modify_pool_partition_metadata_or_gates": False,
            "download_and_inventory_all_pinned_shards_once": True,
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
