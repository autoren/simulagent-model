#!/usr/bin/env python3
from __future__ import annotations

from email.utils import parsedate_to_datetime
import hashlib
import json
from typing import Any
from urllib.request import Request, urlopen

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v102r1_presto_context_source import (
    build_repaired_presto_context_inventory,
    evaluate_repaired_presto_source_gates,
    parse_presto_archive,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def normalized_header(value: str | None) -> str:
    return (value or "").strip().strip('"')


def normalized_last_modified(value: str | None) -> str:
    if not value:
        return ""
    return parsedate_to_datetime(value).isoformat().replace("+00:00", "Z")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v102r1-presto-context-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v102r1-presto-context-source/source"
    inventory_root = PROJECT_ROOT / "outputs/v102r1-presto-context-source/source-inventory"
    if source_root.exists() or inventory_root.exists():
        raise RuntimeError("V102r1 source inventory may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash(
        {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    ) != lock["lock_payload_sha256"]:
        raise RuntimeError("V102r1 source lock mismatch")
    dependency_keys = (
        "repair_config", "parent_technical_outcome", "scientific_config", "plan",
        "protocol", "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V102r1 dependency drifted: {key}")

    config = lock["scientific_config_payload"]
    archive = config["archive"]
    request = Request(archive["url"], headers={
        "User-Agent": "simulagent-v102r1-presto-source",
        "Accept-Encoding": "identity",
    })
    with urlopen(request, timeout=300) as response:  # noqa: S310 - locked official HTTPS source
        response_etag = normalized_header(response.headers.get("ETag"))
        response_generation = normalized_header(response.headers.get("x-goog-generation"))
        response_last_modified = normalized_last_modified(response.headers.get("Last-Modified"))
        response_content_length = response.headers.get("Content-Length")
        data = response.read(archive["byteSize"] + 1)
    if (
        len(data) != archive["byteSize"]
        or md5_hex(data) != archive["etagMd5"]
        or response_etag != archive["etagMd5"]
        or response_generation != archive["generation"]
        or response_last_modified != archive["lastModified"]
        or (response_content_length is not None and int(response_content_length) != archive["byteSize"])
    ):
        raise RuntimeError("V102r1 pinned PRESTO archive identity mismatch")

    source_root.mkdir(parents=True)
    source_path = source_root / "presto_v1.zip"
    source_path.write_bytes(data)
    source_sha256 = file_sha256(source_path)
    source_records, members = parse_presto_archive(
        data, archive["requiredMemberBasenames"]
    )
    inventory = build_repaired_presto_context_inventory(source_records, config)
    artifact = {
        "provenance": {
            "source_name": config["sourceName"],
            "source_role": config["sourceRole"],
            "official_repository": config["officialRepository"],
            "official_paper": config["officialPaper"],
            "archive_url": archive["url"],
            "archive_sha256": source_sha256,
            "parsed_members": members,
            "license": config["license"],
            "parser_repair": "ignore_non_string_optional_context_leaves_without_coercion",
            "contains_source_language": False,
            "contains_target_arguments_context_tokens_or_seeded_values": False,
        },
        **inventory,
    }
    inventory_root.mkdir(parents=True)
    inventory_path = inventory_root / "presto-context-dependency-inventory.json"
    inventory_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    checks = evaluate_repaired_presto_source_gates(inventory, config)
    checks["repair_encountered_non_string_optional_context_leaves"] = (
        inventory["ignored_non_string_optional_context_leaf_count"] > 0
    )
    checks["zero_manual_model_API_training_service_or_side_effect_access"] = True
    passed = all(checks.values())
    summary_keys = (
        "source_record_count", "source_member_record_counts", "source_split_record_counts",
        "en_us_human_context_record_counts", "eligible_candidate_count",
        "role_candidate_counts", "dependency_source_kind_counts", "dependency_source_kind_count",
        "previous_turn_dependent_candidate_count", "seeded_state_dependent_candidate_count",
        "semantic_root_function_count", "synthetic_context_candidate_count",
        "ignored_non_string_optional_context_leaf_count",
        "development_test_identifiers_are_disjoint", "candidate_index_sha256",
        "pairs_share_source_id_input_and_target_by_construction",
        "contains_input_target_argument_context_tokens_seeded_values_or_prompts",
    )
    result = {
        "schema_version": "102r1-presto-context-source-result",
        "experiment": "v102r1_presto_context_source_inventory",
        "passed": passed,
        "decision": (
            "freeze_PRESTO_source_and_preregister_paired_context_population"
            if passed else "stop_V102r1_before_population_language_or_model_access"
        ),
        "source_integrity": {
            "url": archive["url"], "byte_size": len(data), "etag_md5": response_etag,
            "generation": response_generation, "last_modified": response_last_modified,
            "archive_sha256": source_sha256, "parsed_members": members,
            "local_path": str(source_path.relative_to(PROJECT_ROOT)),
            "local_sha256": source_sha256,
        },
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "inventory_summary": {key: inventory[key] for key in summary_keys},
        "gates": checks,
        "access": {
            "pinned_HTTP_download_count": 1,
            "downloaded_byte_count": len(data),
            "archive_persist_before_parse_count": 1,
            "archive_payload_parse_count": 1,
            "language_record_automatic_parse_count": len(source_records),
            "emitted_language_record_count": 0,
            "manual_utterance_inspection_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
            "LLM_API_call_count": 0,
            "adapter_training_run_count": 0,
            "real_service_call_count": 0,
            "external_side_effect_count": 0,
        },
        "claim_boundary": (
            "automatic text-free repaired PRESTO source feasibility only; no selected language, "
            "model, calibration, posterior, planning, or execution outcome"
        ),
    }
    result_path = inventory_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed, "decision": result["decision"],
        "source_integrity": result["source_integrity"],
        "inventory_summary": result["inventory_summary"],
        "gates": checks, "access": result["access"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
