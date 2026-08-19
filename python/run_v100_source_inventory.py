#!/usr/bin/env python3
from __future__ import annotations

from email.utils import parsedate_to_datetime
import hashlib
import json
from typing import Any
from urllib.request import Request, urlopen

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v100_massive_source import (
    build_massive_source_inventory,
    evaluate_massive_source_gates,
    parse_massive_archive,
    sha256_bytes,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalized_etag(value: str | None) -> str:
    return (value or "").strip().strip('"')


def normalized_last_modified(value: str | None) -> str:
    if not value:
        return ""
    return parsedate_to_datetime(value).isoformat().replace("+00:00", "Z")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v100-massive-source-lock.json"
    source_root = PROJECT_ROOT / "outputs/v100-massive-open-set/source"
    inventory_root = PROJECT_ROOT / "outputs/v100-massive-open-set/source-inventory"
    if source_root.exists() or inventory_root.exists():
        raise RuntimeError("V100 source inventory may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash(
        {key: value for key, value in lock.items() if key != "lock_payload_sha256"}
    ) != lock["lock_payload_sha256"]:
        raise RuntimeError("V100 source lock mismatch")
    dependency_keys = (
        "config", "parent_source_selection_lock", "plan", "protocol", "tests",
        "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V100 dependency drifted: {key}")

    config = lock["config_payload"]
    archive = config["archive"]
    request = Request(archive["url"], headers={
        "User-Agent": "simulagent-v100-massive-source",
        "Accept-Encoding": "identity",
    })
    with urlopen(request, timeout=120) as response:  # noqa: S310 - locked official HTTPS source
        response_etag = normalized_etag(response.headers.get("ETag"))
        response_last_modified = normalized_last_modified(response.headers.get("Last-Modified"))
        response_content_length = response.headers.get("Content-Length")
        data = response.read(archive["byteSize"] + 1)
    if (
        len(data) != archive["byteSize"]
        or response_etag != archive["etag"]
        or response_last_modified != archive["lastModified"]
        or (response_content_length is not None and int(response_content_length) != archive["byteSize"])
    ):
        raise RuntimeError("V100 pinned MASSIVE archive identity mismatch")

    records, locale_member = parse_massive_archive(
        data, archive["expectedLocaleMemberSuffix"]
    )
    inventory = build_massive_source_inventory(records, config)
    artifact = {
        "provenance": {
            "source_name": config["sourceName"],
            "source_role": config["sourceRole"],
            "official_dataset": config["officialDataset"],
            "official_paper": config["officialPaper"],
            "archive_url": archive["url"],
            "archive_sha256": sha256_bytes(data),
            "locale_member": locale_member,
            "license": config["license"],
            "contains_source_language": False,
            "contains_derived_language_tokens_or_slot_values": False,
        },
        **inventory,
    }
    source_root.mkdir(parents=True)
    source_path = source_root / "amazon-massive-dataset-1.1.tar.gz"
    source_path.write_bytes(data)
    inventory_root.mkdir(parents=True)
    inventory_path = inventory_root / "massive-open-set-inventory.json"
    inventory_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    checks = evaluate_massive_source_gates(inventory, config)
    checks["zero_manual_model_API_training_service_or_side_effect_access"] = True
    passed = all(checks.values())
    summary_keys = (
        "source_record_count", "partition_counts", "scenario_count", "intent_count",
        "slot_type_count", "eligible_scenarios", "eligible_scenario_count",
        "catalog_scenarios", "catalog_scenario_count", "unsupported_scenarios",
        "unsupported_scenario_count", "hidden_scenarios", "hidden_intents",
        "hidden_intent_count", "declared_intents", "declared_intent_count",
        "candidate_count", "class_counts", "class_scenario_counts",
        "class_partition_counts", "candidate_index_sha256",
        "roles_selected_before_utterance_features",
        "contains_raw_or_annotated_utterances_tokens_or_slot_values",
    )
    result = {
        "schema_version": "100-massive-source-result",
        "experiment": "v100_massive_source_inventory",
        "passed": passed,
        "decision": (
            "freeze_MASSIVE_source_and_preregister_open_set_population"
            if passed else "stop_V100_before_population_language_or_model_access"
        ),
        "source_integrity": {
            "url": archive["url"],
            "byte_size": len(data),
            "etag": response_etag,
            "last_modified": response_last_modified,
            "archive_sha256": sha256_bytes(data),
            "locale_member": locale_member,
            "local_path": str(source_path.relative_to(PROJECT_ROOT)),
            "local_sha256": file_sha256(source_path),
        },
        "inventory": str(inventory_path.relative_to(PROJECT_ROOT)),
        "inventory_sha256": file_sha256(inventory_path),
        "inventory_summary": {key: inventory[key] for key in summary_keys},
        "gates": checks,
        "access": {
            "pinned_HTTP_download_count": 1,
            "downloaded_byte_count": len(data),
            "archive_payload_parse_count": 1,
            "locale_language_record_parse_count": len(records),
            "automatic_utterance_tokenization_count": len(records),
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
            "automatic text-free MASSIVE source feasibility only; no selected language, "
            "model, calibration, posterior, planning, or execution outcome"
        ),
    }
    result_path = inventory_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": passed,
        "decision": result["decision"],
        "source_integrity": result["source_integrity"],
        "inventory_summary": result["inventory_summary"],
        "gates": checks,
        "access": result["access"],
    }, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
