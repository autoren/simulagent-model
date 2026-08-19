#!/usr/bin/env python3
from __future__ import annotations

import json
from urllib.request import Request, urlopen

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v124_sgd_source_feasibility import build_inventory


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v124-sgd-source-feasibility-lock.json"
    archive_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/source/sgd-pinned.tar.gz"
    inventory_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/source-inventory/sgd-open-set-inventory.json"
    access_path = PROJECT_ROOT / "outputs/v124-sgd-source-feasibility/source-inventory/access.json"
    if any(path.exists() for path in (archive_path, inventory_path, access_path)):
        raise RuntimeError("V124 download and inventory may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V124 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V124 dependency drifted: {key}")
    config = lock["config_payload"]
    request = Request(config["archiveUrl"], headers={"User-Agent": "simulagent-source-audit/1.0"})
    with urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"V124 archive status {response.status}")
        archive_bytes = response.read()
    if not archive_bytes:
        raise RuntimeError("V124 archive is empty")
    inventory = build_inventory(archive_bytes, config)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(archive_bytes)
    write_json(inventory_path, inventory)
    access = {
        "archive_payload_download_count": 1,
        "source_archive_read_count": 1,
        "automatic_language_parse_count": 1,
        "emitted_language_record_count": 0,
        "manual_language_or_raw_response_inspection_count": 0,
        "protected_test_language_read_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
        "actual_execution_count": 0,
    }
    write_json(access_path, access)
    print(json.dumps({
        "decision": inventory["decision"],
        "source_pass": inventory["source_pass"],
        "archive_sha256": inventory["archive_sha256"],
        "dialogue_count": inventory["dialogue_count"],
        "candidate_count": inventory["candidate_count"],
        "test_open_set_class_counts": inventory["test_open_set_class_counts"],
        "test_open_set_domain_coverage": inventory["test_open_set_domain_coverage"],
        "source_gates": inventory["source_gates"],
        "access": access,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
