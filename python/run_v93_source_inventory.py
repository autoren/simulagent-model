#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v93_open_set_source import build_open_set_inventory, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v93-controlled-open-set-source-lock.json"
    output_root = PROJECT_ROOT / "outputs/v93-controlled-open-set"
    source_root = output_root / "source"
    inventory_root = output_root / "source-inventory"
    if source_root.exists() or inventory_root.exists():
        raise RuntimeError("V93 source acquisition may run only once")
    lock = json.loads(lock_path.read_text())
    unhashed = {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    if payload_hash(unhashed) != lock["lock_payload_sha256"]:
        raise RuntimeError("V93 source lock payload mismatch")
    for key in (
        "config", "parent_architecture_lock", "source_authority_lock", "plan",
        "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    ):
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V93 source dependency drifted: {key}")
    if not lock["authorization"]["download_and_inventory_pinned_shard_once"]:
        raise RuntimeError("V93 source download is not authorized")

    config = lock["config_payload"]
    shard = config["newDialogueShard"]
    request = Request(shard["rawUrl"], headers={"User-Agent": "simulagent-v93-open-set-source"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - exact pinned HTTPS URL
        dialogue_bytes = response.read(shard["byteSize"] + 1)
    if len(dialogue_bytes) != shard["byteSize"]:
        raise RuntimeError("V93 shard byte size mismatch")
    if git_blob_sha1(dialogue_bytes) != shard["gitBlobSha1"]:
        raise RuntimeError("V93 shard Git blob mismatch")

    schema_spec = config["schemaDependency"]
    schema_path = PROJECT_ROOT / schema_spec["localPath"]
    if file_sha256(schema_path) != schema_spec["localSha256"]:
        raise RuntimeError("V93 schema dependency drifted")
    inventory = build_open_set_inventory(
        json.loads(schema_path.read_text()),
        json.loads(dialogue_bytes),
        config,
    )
    inventory_artifact = {
        "provenance": {
            "source_name": "Schema-Guided Dialogue Dataset",
            "repository": config["repository"],
            "revision": config["revision"],
            "source_shard": shard["path"],
            "license": config["license"],
            "contains_source_language": False,
            "contains_derived_language_tokens": False,
        },
        **inventory,
    }
    source_root.mkdir(parents=True)
    source_path = source_root / Path(shard["path"]).name
    source_path.write_bytes(dialogue_bytes)
    inventory_root.mkdir(parents=True)
    inventory_path = inventory_root / "open-set-inventory.json"
    inventory_path.write_text(json.dumps(inventory_artifact, indent=2, sort_keys=True) + "\n")

    counts = inventory["class_counts"]
    coverage = inventory["class_service_counts"]
    gates = config["sourceGates"]
    class_gate_map = {
        "known_familiar": "minimumKnownFamiliarCandidateCount",
        "known_unfamiliar": "minimumKnownUnfamiliarCandidateCount",
        "novel_valid": "minimumNovelValidCandidateCount",
        "unsupported": "minimumUnsupportedCandidateCount",
        "insufficient_evidence": "minimumInsufficientEvidenceCandidateCount",
    }
    gate_checks = {
        "eligible_service_count": inventory["eligible_service_count"] >= gates["minimumEligibleServiceCount"],
        **{
            f"{label}_candidate_count": counts.get(label, 0) >= gates[gate_key]
            for label, gate_key in class_gate_map.items()
        },
        **{
            f"{label}_service_coverage": coverage.get(label, 0) >= gates["minimumClassServiceCoverage"]
            for label in class_gate_map
        },
        "text_free_inventory": not inventory["contains_language_tokens_slot_values_or_histories"],
        "zero_manual_model_API_training_service_or_side_effect_access": True,
    }
    passed = all(gate_checks.values())
    result = {
        "schema_version": "93-controlled-open-set-source-result",
        "experiment": "v93_controlled_open_set_source_inventory",
        "passed": passed,
        "decision": "freeze_source_and_preregister_open_set_population" if passed else "stop_V93_before_population_or_model_access",
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
            "eligible_service_count": inventory["eligible_service_count"],
            "service_splits": inventory["service_splits"],
            "source_record_count": inventory["source_record_count"],
            "eligible_source_record_count": inventory["eligible_source_record_count"],
            "candidate_count": inventory["candidate_count"],
            "class_counts": counts,
            "class_service_counts": coverage,
            "candidate_index_sha256": inventory["candidate_index_sha256"],
            "contains_language_tokens_slot_values_or_histories": False,
        },
        "gates": gate_checks,
        "access": {
            "pinned_HTTP_download_count": 1,
            "downloaded_byte_count": len(dialogue_bytes),
            "dialogue_payload_parse_count": 1,
            "automatic_language_tokenization_count": inventory["source_record_count"],
            "emitted_language_record_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": "automatic source feasibility and text-free class inventory only; no language inspection, population, model, calibration, posterior, planning, or execution result",
    }
    result_path = inventory_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed,
        "decision": result["decision"],
        "inventory_summary": result["inventory_summary"],
        "gates": gate_checks,
        "access": result["access"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
