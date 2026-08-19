#!/usr/bin/env python3
"""Download the pinned untouched V91 SGD shard and emit only a structural inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v87_external_source_inventory import build_structural_inventory, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v91-rank-only-source-lock.json"
    output_root = PROJECT_ROOT / "outputs/v91-rank-only"
    source_root = output_root / "source"
    inventory_root = output_root / "source-inventory"
    if source_root.exists() or inventory_root.exists():
        raise RuntimeError("V91 source acquisition and inventory may run only once")
    lock = json.loads(lock_path.read_text())
    lock_payload = {
        key: value for key, value in lock.items() if key != "lock_payload_sha256"
    }
    if payload_hash(lock_payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V91 source lock payload mismatch")
    for key in (
        "config",
        "source_authority_lock",
        "parent_model_decision_lock",
        "previous_source_outcome_lock",
        "plan",
        "inventory_module",
        "runner",
        "verifier",
        "auditor",
    ):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V91 locked dependency drifted: {key}")
    if not lock["authorization"]["download_and_inventory_pinned_shard_once"]:
        raise RuntimeError("V91 source acquisition is not authorized")

    config = lock["config_payload"]
    shard = config["newDialogueShard"]
    schema_spec = config["schemaDependency"]
    request = Request(
        shard["rawUrl"], headers={"User-Agent": "simulagent-v91-rank-only-source"}
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - immutable locked HTTPS URL
        dialogue_bytes = response.read(shard["byteSize"] + 1)
    if len(dialogue_bytes) != shard["byteSize"]:
        raise RuntimeError("V91 source shard byte size mismatch")
    if git_blob_sha1(dialogue_bytes) != shard["gitBlobSha1"]:
        raise RuntimeError("V91 source shard Git blob mismatch")

    schema_path = PROJECT_ROOT / schema_spec["localPath"]
    if file_sha256(schema_path) != schema_spec["localSha256"]:
        raise RuntimeError("V91 pinned schema dependency drifted")
    schema_payload = json.loads(schema_path.read_bytes())
    dialogue_payload = json.loads(dialogue_bytes)
    inventory = build_structural_inventory(
        schema_payload,
        dialogue_payload,
        excluded_service_prefixes=tuple(
            config["inventoryProtocol"]["excludedServicePrefixes"]
        ),
    )
    inventory_artifact = {
        "provenance": {
            "source_name": "Schema-Guided Dialogue Dataset",
            "repository": config["repository"],
            "revision": config["revision"],
            "source_shard": shard["path"],
            "license": config["license"],
            "adapted_inventory_license": config["license"],
            "contains_source_utterance_text": False,
        },
        **inventory,
    }
    source_root.mkdir(parents=True)
    source_path = source_root / Path(shard["path"]).name
    source_path.write_bytes(dialogue_bytes)
    inventory_root.mkdir(parents=True)
    inventory_path = inventory_root / "structural-inventory.json"
    inventory_path.write_text(
        json.dumps(inventory_artifact, indent=2, sort_keys=True) + "\n"
    )

    gates = config["sourceGates"]
    counts = inventory["counts"]
    active_intents = {
        key
        for key in inventory["eligible_intent_counts"]
        if not key.endswith("::NONE")
    }
    gate_checks = {
        "eligible_record_count": counts["eligible_record_count"]
        >= gates["minimumEligibleRecordCount"],
        "eligible_active_record_count": counts["eligible_active_record_count"]
        >= gates["minimumEligibleActiveRecordCount"],
        "eligible_none_record_count": counts["eligible_none_record_count"]
        >= gates["minimumEligibleNoneRecordCount"],
        "eligible_service_count": len(inventory["eligible_service_counts"])
        >= gates["minimumEligibleServiceCount"],
        "eligible_active_intent_count": len(active_intents)
        >= gates["minimumEligibleActiveIntentCount"],
        "text_free_inventory": not inventory["contains_utterance_or_text_fields"],
        "zero_manual_model_API_service_or_side_effect_access": True,
    }
    passed = all(gate_checks.values())
    result = {
        "schema_version": "91-rank-only-source-result",
        "experiment": "v91_rank_only_source_inventory",
        "passed": passed,
        "decision": (
            "freeze_source_and_preregister_fresh_rank_only_population"
            if passed
            else "stop_V91_before_population_or_model_access"
        ),
        "source_integrity": {
            "path": shard["path"],
            "byte_size": len(dialogue_bytes),
            "git_blob_sha1": git_blob_sha1(dialogue_bytes),
            "local_path": str(source_path.relative_to(PROJECT_ROOT)),
            "local_sha256": file_sha256(source_path),
        },
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "inventory_summary": {
            "counts": counts,
            "eligible_service_counts": inventory["eligible_service_counts"],
            "eligible_intent_counts": inventory["eligible_intent_counts"],
            "record_index_sha256": inventory["record_index_sha256"],
            "contains_utterance_or_text_fields": inventory[
                "contains_utterance_or_text_fields"
            ],
        },
        "gates": gate_checks,
        "access": {
            "pinned_HTTP_download_count": 1,
            "downloaded_byte_count": len(dialogue_bytes),
            "dialogue_payload_parse_count": 1,
            "manual_utterance_inspection_count": 0,
            "new_model_weight_download_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": (
            "source integrity and text-free structural inventory only; no language score, "
            "population selection, model access, training, pruning, belief, action, or execution"
        ),
    }
    result_path = inventory_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "decision": result["decision"],
                "inventory_summary": result["inventory_summary"],
                "gates": gate_checks,
                "access": result["access"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
