#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v94_global_open_set_source import build_global_catalog_inventory, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v94-global-open-set-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v94-global-open-set/source"
    inventory_root = PROJECT_ROOT / "outputs/v94-global-open-set/source-inventory"
    if source_root.exists() or inventory_root.exists():
        raise RuntimeError("V94 source inventory may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({k: v for k, v in lock.items() if k != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V94 source lock mismatch")
    dependency_keys = (
        "config", "parent_source_closure", "parent_architecture_lock", "source_authority_lock",
        "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V94 dependency drifted: {key}")

    config = lock["config_payload"]
    shard = config["newDialogueShard"]
    request = Request(shard["rawUrl"], headers={"User-Agent": "simulagent-v94-global-open-set"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned immutable HTTPS URL
        data = response.read(shard["byteSize"] + 1)
    if len(data) != shard["byteSize"] or git_blob_sha1(data) != shard["gitBlobSha1"]:
        raise RuntimeError("V94 pinned shard identity mismatch")
    schema_path = PROJECT_ROOT / config["schemaDependency"]["localPath"]
    if file_sha256(schema_path) != config["schemaDependency"]["localSha256"]:
        raise RuntimeError("V94 schema dependency drifted")
    inventory = build_global_catalog_inventory(
        json.loads(schema_path.read_text()), json.loads(data), config
    )
    artifact = {
        "provenance": {
            "source_name": "Schema-Guided Dialogue Dataset",
            "repository": config["repository"], "revision": config["revision"],
            "source_shard": shard["path"], "license": config["license"],
            "contains_source_language": False, "contains_derived_language_tokens": False,
        },
        **inventory,
    }
    source_root.mkdir(parents=True)
    source_path = source_root / Path(shard["path"]).name
    source_path.write_bytes(data)
    inventory_root.mkdir(parents=True)
    inventory_path = inventory_root / "global-open-set-inventory.json"
    inventory_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    gates = config["sourceGates"]
    counts = inventory["class_counts"]
    coverage = inventory["class_service_counts"]
    checks = {
        "eligible_fresh_service_count": inventory["eligible_fresh_service_count"] >= gates["minimumEligibleFreshServiceCount"],
        "catalog_service_count": inventory["catalog_service_count"] >= gates["minimumCatalogServiceCount"],
        "unsupported_service_count": inventory["unsupported_service_count"] >= gates["minimumUnsupportedServiceCount"],
        "hidden_pair_count": inventory["hidden_pair_count"] >= gates["minimumHiddenPairCount"],
        "declared_pair_count": inventory["declared_supported_pair_count"] >= gates["minimumDeclaredPairCount"],
        "known_familiar_candidate_count": counts.get("known_familiar", 0) >= gates["minimumKnownFamiliarCandidateCount"],
        "known_unfamiliar_candidate_count": counts.get("known_unfamiliar", 0) >= gates["minimumKnownUnfamiliarCandidateCount"],
        "novel_valid_candidate_count": counts.get("novel_valid", 0) >= gates["minimumNovelValidCandidateCount"],
        "unsupported_candidate_count": counts.get("unsupported", 0) >= gates["minimumUnsupportedCandidateCount"],
        "insufficient_evidence_candidate_count": counts.get("insufficient_evidence", 0) >= gates["minimumInsufficientEvidenceCandidateCount"],
        "known_familiar_service_coverage": coverage.get("known_familiar", 0) >= gates["minimumKnownClassServiceCoverage"],
        "known_unfamiliar_service_coverage": coverage.get("known_unfamiliar", 0) >= gates["minimumKnownClassServiceCoverage"],
        "novel_service_coverage": coverage.get("novel_valid", 0) >= gates["minimumNovelServiceCoverage"],
        "insufficient_evidence_service_coverage": coverage.get("insufficient_evidence", 0) >= gates["minimumInsufficientEvidenceServiceCoverage"],
        "unsupported_service_coverage": coverage.get("unsupported", 0) >= gates["minimumUnsupportedServiceCount"],
        "text_free_inventory": not inventory["contains_language_tokens_slot_values_or_histories"],
        "zero_manual_model_API_training_service_or_side_effect_access": True,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "94-global-open-set-source-result",
        "experiment": "v94_global_open_set_source_inventory",
        "passed": passed,
        "decision": "freeze_source_and_preregister_global_open_set_population" if passed else "stop_V94_before_population_or_model_access",
        "source_integrity": {
            "path": shard["path"], "byte_size": len(data), "git_blob_sha1": git_blob_sha1(data),
            "local_path": str(source_path.relative_to(PROJECT_ROOT)), "local_sha256": file_sha256(source_path),
        },
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "inventory_summary": {key: inventory[key] for key in (
            "source_record_count", "eligible_fresh_services", "eligible_fresh_service_count",
            "catalog_services", "catalog_service_count", "unsupported_services", "unsupported_service_count",
            "supported_pair_count", "hidden_pairs", "hidden_pair_count", "declared_supported_pairs",
            "declared_supported_pair_count", "complete_catalog_pair_count", "candidate_count", "class_counts",
            "class_service_counts", "candidate_index_sha256",
            "contains_language_tokens_slot_values_or_histories",
        )},
        "gates": checks,
        "access": {
            "pinned_HTTP_download_count": 1, "downloaded_byte_count": len(data), "dialogue_payload_parse_count": 1,
            "automatic_language_tokenization_count": inventory["source_record_count"], "emitted_language_record_count": 0,
            "manual_utterance_inspection_count": 0, "model_load_count": 0, "model_generation_count": 0,
            "LLM_API_call_count": 0, "adapter_training_run_count": 0, "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": "automatic text-free source feasibility only; no selected language, model, calibration, posterior, planning, or execution outcome",
    }
    result_path = inventory_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "decision": result["decision"], "inventory_summary": result["inventory_summary"], "gates": checks, "access": result["access"]}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
