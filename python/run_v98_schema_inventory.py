#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.request import Request, urlopen

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v98_test_schema_feasibility import build_test_schema_inventory, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v98-test-schema-feasibility-lock.json"
    source_root = PROJECT_ROOT / "outputs/v98-test-schema-feasibility/source"
    inventory_root = PROJECT_ROOT / "outputs/v98-test-schema-feasibility/schema-inventory"
    if source_root.exists() or inventory_root.exists():
        raise RuntimeError("V98 schema inventory may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V98 schema lock mismatch")
    dependency_keys = (
        "config", "parent_source_outcome", "source_authority_lock", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V98 dependency drifted: {key}")

    config = lock["config_payload"]
    schema = config["testSchema"]
    request = Request(schema["rawUrl"], headers={"User-Agent": "simulagent-v98-test-schema"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned immutable HTTPS URL
        data = response.read(schema["byteSize"] + 1)
    if len(data) != schema["byteSize"] or git_blob_sha1(data) != schema["gitBlobSha1"]:
        raise RuntimeError("V98 pinned test schema identity mismatch")
    dev_path = PROJECT_ROOT / config["developmentSchemaDependency"]["localPath"]
    if file_sha256(dev_path) != config["developmentSchemaDependency"]["localSha256"]:
        raise RuntimeError("V98 development schema dependency drifted")
    inventory = build_test_schema_inventory(
        json.loads(dev_path.read_text()), json.loads(data), config
    )
    artifact = {
        "provenance": {
            "source_name": "Schema-Guided Dialogue Dataset",
            "repository": config["repository"],
            "revision": config["revision"],
            "source_schema": schema["path"],
            "license": config["license"],
            "contains_schema_language": False,
        },
        **inventory,
    }
    source_root.mkdir(parents=True)
    source_path = source_root / "schema.json"
    source_path.write_bytes(data)
    inventory_root.mkdir(parents=True)
    inventory_path = inventory_root / "test-schema-inventory.json"
    inventory_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    gates = config["schemaGates"]
    checks = {
        "novel_service_family_count": inventory["novel_service_family_count"] >= gates["minimumNovelServiceFamilyCount"],
        "eligible_novel_service_count": inventory["eligible_novel_service_count"] >= gates["minimumEligibleNovelServiceCount"],
        "test_dialogue_shard_metadata_count": config["testDialogueShardMetadataCount"] >= gates["minimumTestDialogueShardMetadataCount"],
        "zero_emitted_intent_names": inventory["emitted_intent_name_count"] <= gates["maximumEmittedIntentNameCount"],
        "zero_emitted_intent_descriptions": inventory["emitted_intent_description_count"] <= gates["maximumEmittedIntentDescriptionCount"],
        "zero_emitted_slot_names": inventory["emitted_slot_name_count"] <= gates["maximumEmittedSlotNameCount"],
        "schema_inventory_is_language_free": not inventory["contains_schema_language_or_surface_tokens"],
        "zero_dialogue_manual_model_API_training_service_or_side_effect_access": True,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "98-test-schema-feasibility-result",
        "experiment": "v98_test_schema_feasibility_inventory",
        "passed": passed,
        "decision": "freeze_schema_and_preregister_test_dialogue_source_pool" if passed else "stop_V98_before_test_dialogue_or_model_access",
        "source_integrity": {
            "path": schema["path"], "byte_size": len(data), "git_blob_sha1": git_blob_sha1(data),
            "local_path": str(source_path.relative_to(PROJECT_ROOT)), "local_sha256": file_sha256(source_path),
        },
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "inventory_summary": inventory,
        "gates": checks,
        "access": {
            "pinned_HTTP_download_count": 1,
            "downloaded_byte_count": len(data),
            "test_schema_payload_parse_count": 1,
            "test_dialogue_payload_access_count": 0,
            "emitted_schema_language_record_count": 0,
            "manual_schema_language_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": "automatic schema-family feasibility only; no dialogue language, model, calibration, posterior, planning, or execution outcome",
    }
    result_path = inventory_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed, "decision": result["decision"],
        "inventory_summary": inventory, "gates": checks, "access": result["access"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
