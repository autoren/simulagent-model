#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v96_two_source_open_set_source import build_two_source_open_set_inventory, git_blob_sha1


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def download(shard: dict[str, Any], user_agent: str) -> bytes:
    request = Request(shard["rawUrl"], headers={"User-Agent": user_agent})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned immutable HTTPS URL
        data = response.read(shard["byteSize"] + 1)
    if len(data) != shard["byteSize"] or git_blob_sha1(data) != shard["gitBlobSha1"]:
        raise RuntimeError(f"V96 pinned shard identity mismatch: {shard['path']}")
    return data


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v96-two-source-open-set-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v96-two-source-open-set/source"
    inventory_root = PROJECT_ROOT / "outputs/v96-two-source-open-set/source-inventory"
    if source_root.exists() or inventory_root.exists():
        raise RuntimeError("V96 source inventory may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V96 source lock mismatch")
    dependency_keys = (
        "config", "parent_source_outcome", "parent_architecture_lock", "source_authority_lock",
        "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V96 dependency drifted: {key}")

    config = lock["config_payload"]
    catalog_shard = config["catalogDialogueShard"]
    unsupported_shard = config["unsupportedDialogueShard"]
    catalog_data = download(catalog_shard, "simulagent-v96-catalog-source")
    unsupported_data = download(unsupported_shard, "simulagent-v96-unsupported-source")
    schema_path = PROJECT_ROOT / config["schemaDependency"]["localPath"]
    if file_sha256(schema_path) != config["schemaDependency"]["localSha256"]:
        raise RuntimeError("V96 schema dependency drifted")
    inventory = build_two_source_open_set_inventory(
        json.loads(schema_path.read_text()),
        json.loads(catalog_data),
        json.loads(unsupported_data),
        config,
    )
    artifact = {
        "provenance": {
            "source_name": "Schema-Guided Dialogue Dataset",
            "repository": config["repository"],
            "revision": config["revision"],
            "catalog_source_shard": catalog_shard["path"],
            "unsupported_source_shard": unsupported_shard["path"],
            "license": config["license"],
            "contains_source_language": False,
            "contains_derived_language_tokens": False,
        },
        **inventory,
    }
    source_root.mkdir(parents=True)
    catalog_path = source_root / f"catalog-{Path(catalog_shard['path']).name}"
    unsupported_path = source_root / f"unsupported-{Path(unsupported_shard['path']).name}"
    catalog_path.write_bytes(catalog_data)
    unsupported_path.write_bytes(unsupported_data)
    inventory_root.mkdir(parents=True)
    inventory_path = inventory_root / "two-source-open-set-inventory.json"
    inventory_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    gates = config["sourceGates"]
    counts = inventory["class_counts"]
    coverage = inventory["class_service_counts"]
    checks = {
        "catalog_service_count": inventory["catalog_service_count"] == gates["requiredCatalogServiceCount"],
        "unsupported_service_count": inventory["unsupported_service_count"] == gates["requiredUnsupportedServiceCount"],
        "source_roles_are_disjoint": inventory["source_roles_are_disjoint"],
        "hidden_pair_count": inventory["hidden_pair_count"] == gates["requiredHiddenPairCount"],
        "hidden_pairs_are_service_stratified": len(inventory["hidden_services"]) == gates["requiredNovelServiceCoverage"],
        "declared_pair_count": inventory["declared_supported_pair_count"] >= gates["minimumDeclaredPairCount"],
        "known_familiar_candidate_count": counts.get("known_familiar", 0) >= gates["minimumKnownFamiliarCandidateCount"],
        "known_unfamiliar_candidate_count": counts.get("known_unfamiliar", 0) >= gates["minimumKnownUnfamiliarCandidateCount"],
        "novel_valid_candidate_count": counts.get("novel_valid", 0) >= gates["minimumNovelValidCandidateCount"],
        "unsupported_candidate_count": counts.get("unsupported", 0) >= gates["minimumUnsupportedCandidateCount"],
        "insufficient_evidence_candidate_count": counts.get("insufficient_evidence", 0) >= gates["minimumInsufficientEvidenceCandidateCount"],
        "known_familiar_service_coverage": coverage.get("known_familiar", 0) >= gates["minimumKnownClassServiceCoverage"],
        "known_unfamiliar_service_coverage": coverage.get("known_unfamiliar", 0) >= gates["minimumKnownClassServiceCoverage"],
        "novel_service_coverage": coverage.get("novel_valid", 0) == gates["requiredNovelServiceCoverage"],
        "insufficient_evidence_service_coverage": coverage.get("insufficient_evidence", 0) >= gates["minimumInsufficientEvidenceServiceCoverage"],
        "unsupported_service_coverage": coverage.get("unsupported", 0) == gates["requiredUnsupportedServiceCoverage"],
        "activation_and_current_turn_boundaries_hold": bool(
            inventory["all_non_none_candidates_are_source_intent_activations"]
            and inventory["lexical_separation_uses_current_turn_only"]
        ),
        "text_free_inventory": not inventory["contains_language_tokens_slot_values_or_histories"],
        "zero_manual_model_API_training_service_or_side_effect_access": True,
    }
    passed = all(checks.values())
    summary_keys = (
        "catalog_source_record_count", "catalog_source_intent_activation_count",
        "unsupported_source_record_count", "unsupported_source_intent_activation_count",
        "eligible_catalog_services", "eligible_catalog_service_count", "catalog_services",
        "catalog_service_count", "eligible_unsupported_services",
        "eligible_unsupported_service_count", "unsupported_services", "unsupported_service_count",
        "unsupported_supported_pairs", "supported_pairs", "supported_pair_count", "hidden_services",
        "hidden_pairs", "hidden_pair_count", "declared_supported_pairs",
        "declared_supported_pair_count", "complete_catalog_pair_count", "candidate_count",
        "class_counts", "class_service_counts", "candidate_index_sha256", "source_roles_are_disjoint",
        "all_non_none_candidates_are_source_intent_activations",
        "lexical_separation_uses_current_turn_only",
        "contains_language_tokens_slot_values_or_histories",
    )
    source_integrity = {}
    for role, shard, path, data in (
        ("catalog", catalog_shard, catalog_path, catalog_data),
        ("unsupported", unsupported_shard, unsupported_path, unsupported_data),
    ):
        source_integrity[role] = {
            "path": shard["path"], "byte_size": len(data), "git_blob_sha1": git_blob_sha1(data),
            "local_path": str(path.relative_to(PROJECT_ROOT)), "local_sha256": file_sha256(path),
        }
    result = {
        "schema_version": "96-two-source-open-set-source-result",
        "experiment": "v96_two_source_open_set_source_inventory",
        "passed": passed,
        "decision": "freeze_source_and_preregister_two_source_open_set_population" if passed else "stop_V96_before_population_or_model_access",
        "source_integrity": source_integrity,
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "inventory_summary": {key: inventory[key] for key in summary_keys},
        "gates": checks,
        "access": {
            "pinned_HTTP_download_count": 2,
            "downloaded_byte_count": len(catalog_data) + len(unsupported_data),
            "dialogue_payload_parse_count": 2,
            "automatic_current_turn_tokenization_count": inventory["catalog_source_record_count"] + inventory["unsupported_source_record_count"],
            "emitted_language_record_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": "automatic text-free two-source feasibility only; no selected language, model, calibration, posterior, planning, or execution outcome",
    }
    result_path = inventory_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed,
        "decision": result["decision"],
        "inventory_summary": result["inventory_summary"],
        "gates": checks,
        "access": result["access"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
