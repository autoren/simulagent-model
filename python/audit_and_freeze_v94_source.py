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
    return payload_hash({k: v for k, v in payload.items() if k != "lock_payload_sha256"}) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v94-global-open-set-source.json"
    closure_path = PROJECT_ROOT / "configs/v93-controlled-open-set-source-closure-lock.json"
    architecture_path = PROJECT_ROOT / "configs/v92-structured-llm-architecture-outcome-lock.json"
    authority_path = PROJECT_ROOT / "configs/v87-external-language-source-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v94-global-open-set-source-plan.md"
    protocol_path = PROJECT_ROOT / "python/v94_global_open_set_source.py"
    tests_path = PROJECT_ROOT / "python/test_v94_global_open_set_source.py"
    runner_path = PROJECT_ROOT / "python/run_v94_source_inventory.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v94_source_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v94_source.py"
    audit_path = PROJECT_ROOT / "outputs/v94-global-open-set/source-design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v94-global-open-set-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v94-global-open-set/source"
    if audit_path.exists() or lock_path.exists() or source_root.exists():
        raise RuntimeError("V94 source stage is already frozen or materialized")

    config = json.loads(config_path.read_text())
    closure = json.loads(closure_path.read_text())
    architecture = json.loads(architecture_path.read_text())
    authority = json.loads(authority_path.read_text())
    schema_path = PROJECT_ROOT / config["schemaDependency"]["localPath"]
    schema = compile_schema(json.loads(schema_path.read_text()))
    shard = config["newDialogueShard"]
    partition = config["catalogPartition"]
    gates = config["sourceGates"]
    exposure = config["preLockExposure"]
    checks = {
        "V93_negative_closure_is_exact_and_authorizes_material_successor": bool(
            valid_lock(closure)
            and closure["outcome"]["passed"]
            and not closure["outcome"]["scientific_source_feasibility_passed"]
            and closure["authorization"]["preregister_fresh_global_capability_catalog_successor"]
            and not closure["authorization"]["reuse_V93_source_for_successor_outcomes"]
        ),
        "architecture_and_source_authority_are_exact": bool(
            valid_lock(architecture) and valid_lock(authority)
            and architecture["outcome"]["runtime_architecture"]["localLLMRuntimeRole"] == "disabled"
            and authority["outcome"]["passed"]
            and file_sha256(schema_path) == config["schemaDependency"]["localSha256"]
        ),
        "schema_fact_justifies_global_not_per_service_unit": bool(
            max(len(service["intent_names"]) for service in schema.values()) <= 2
            and len(schema) >= 17
        ),
        "fresh_shard_metadata_is_exact_and_unmaterialized": bool(
            shard["path"] == "dev/dialogues_005.json"
            and shard["byteSize"] == 1512309
            and shard["gitBlobSha1"] == "80ddca2c6ed633e49e2f827c05615e514b62ca17"
            and config["revision"] in shard["rawUrl"]
            and not source_root.exists()
        ),
        "partition_is_hash_based_structural_and_nontrivial": bool(
            partition["partitionBeforeLanguageDerivedFeatures"]
            and partition["eligibleServiceMinimumActiveRecordCount"] >= 12
            and partition["eligiblePairMinimumActiveRecordCount"] >= 8
            and partition["minimumCatalogServiceCount"] >= 3
            and partition["minimumUnsupportedServiceCount"] >= 1
            and partition["minimumHiddenPairCount"] >= 2
            and partition["minimumDeclaredPairCount"] >= 4
        ),
        "five_class_count_and_coverage_gates_are_frozen": bool(
            min(
                gates["minimumKnownFamiliarCandidateCount"], gates["minimumKnownUnfamiliarCandidateCount"],
                gates["minimumNovelValidCandidateCount"], gates["minimumUnsupportedCandidateCount"],
            ) >= 24
            and gates["minimumInsufficientEvidenceCandidateCount"] >= 16
            and gates["minimumEligibleFreshServiceCount"] >= 4
        ),
        "zero_dialogue_manual_model_API_or_training_access": all(
            exposure[key] == 0 for key in (
                "dialoguePayloadAccessCount", "individualUtteranceAccessCount", "manualUtteranceInspectionCount",
                "modelLoadCount", "modelGenerationCount", "LLMAPICallCount", "adapterTrainingRunCount",
            )
        ),
        "plan_and_locked_code_exist": all(
            path.is_file() for path in (plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "94-global-open-set-source-design-audit",
        "experiment": "v94_global_open_set_source_design_audit",
        "passed": passed,
        "decision": "freeze_and_authorize_one_global_catalog_source_inventory" if passed else "reject_V94_source_design",
        "checks": checks,
        "prelock_access": exposure,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_source_closure": closure_path,
        "parent_architecture_lock": architecture_path, "source_authority_lock": authority_path,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "94-global-open-set-source-lock",
        "experiment": "v94_global_open_set_source_lock",
        "config_payload": config,
        "authorization": {
            "modify_partition_metadata_or_gates": False,
            "download_and_inventory_pinned_shard_once": True,
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
