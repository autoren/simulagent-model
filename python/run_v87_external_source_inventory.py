#!/usr/bin/env python3
"""Acquire the two pinned SGD files once and write a text-free V87 inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v87_external_source_inventory import build_structural_inventory, fetch_pinned_file


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v87-external-language-source-implementation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v87-external-language-source-audit"
    source_root = output_root / "source"
    evaluation_root = output_root / "inventory"
    if source_root.exists() or evaluation_root.exists():
        raise RuntimeError("V87 pinned acquisition and inventory may run only once")
    lock = json.loads(lock_path.read_text())
    lock_payload = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(lock_payload) != lock["lock_payload_sha256"]:
        raise RuntimeError("V87 implementation lock payload mismatch")
    for key in ("module", "test", "runner", "implementation_auditor"):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V87 locked {key} changed")
    if not lock["authorization"]["acquire_pinned_source_and_inventory_once"]:
        raise RuntimeError("V87 source acquisition is not authorized")

    config = lock["config_payload"]
    source = config["selectedSource"]
    downloaded: dict[str, bytes] = {}
    for spec in source["files"]:
        downloaded[spec["path"]] = fetch_pinned_file(
            spec["path"],
            expected_size=spec["byteSize"],
            expected_blob_sha1=spec["gitBlobSha1"],
        )
    if sum(len(data) for data in downloaded.values()) > source["maximumAcquisitionBytes"]:
        raise RuntimeError("V87 acquisition byte budget exceeded")

    schema_payload = json.loads(downloaded["dev/schema.json"])
    dialogue_payload = json.loads(downloaded["dev/dialogues_001.json"])
    inventory = build_structural_inventory(
        schema_payload,
        dialogue_payload,
        excluded_service_prefixes=tuple(config["postLockStructuralInventoryProtocol"]["excludedServicePrefixes"]),
    )
    provenance = {
        "source_name": "Schema-Guided Dialogue Dataset",
        "repository": next(item["repository"] for item in config["candidates"] if item["selected"]),
        "revision": next(item["revision"] for item in config["candidates"] if item["selected"]),
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "adapted_inventory_license": "CC-BY-SA-4.0",
        "contains_source_utterance_text": False,
    }
    inventory_artifact = {"provenance": provenance, **inventory}

    source_root.mkdir(parents=True)
    for path, data in downloaded.items():
        destination = source_root / Path(path).name
        destination.write_bytes(data)
    evaluation_root.mkdir(parents=True)
    inventory_path = evaluation_root / "structural-inventory.json"
    inventory_path.write_text(json.dumps(inventory_artifact, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": "87-external-language-source-inventory-result",
        "experiment": "v87_external_language_source_inventory",
        "passed": True,
        "decision": "freeze_source_integrity_and_structural_inventory_then_preregister_external_shadow_subset",
        "source_integrity": {
            spec["path"]: {
                "byte_size": spec["byteSize"],
                "git_blob_sha1": spec["gitBlobSha1"],
                "local_sha256": file_sha256(source_root / Path(spec["path"]).name),
            }
            for spec in source["files"]
        },
        "inventory_summary": {
            "schema_service_count": inventory["schema_service_count"],
            "counts": inventory["counts"],
            "ineligibility_reason_counts": inventory["ineligibility_reason_counts"],
            "eligible_service_counts": inventory["eligible_service_counts"],
            "eligible_intent_count": len(inventory["eligible_intent_counts"]),
            "record_index_sha256": inventory["record_index_sha256"],
            "contains_utterance_or_text_fields": inventory["contains_utterance_or_text_fields"],
        },
        "inventory_artifact": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_artifact_sha256": file_sha256(inventory_path),
        "access": {
            "pinned_HTTP_download_count": 2,
            "downloaded_byte_count": sum(len(data) for data in downloaded.values()),
            "schema_payload_parse_count": 1,
            "dialogue_payload_parse_count": 1,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": (
            "This establishes source integrity and a text-free structural inventory only. It does not "
            "measure language understanding, authorize model access, or permit execution."
        ),
    }
    result_path = evaluation_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": result["passed"],
        "decision": result["decision"],
        "inventory_summary": result["inventory_summary"],
        "access": result["access"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
