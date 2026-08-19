#!/usr/bin/env python3
import json

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v122_prequery_signal_inventory import build_inventory
from v22r2_grounding import PROJECT_ROOT


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v122-prequery-signal-inventory-lock.json"
    output_path = PROJECT_ROOT / "outputs/v122-prequery-signal-inventory/audit/result.json"
    if output_path.exists():
        raise RuntimeError("V122 inventory may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V122 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V122 dependency drifted: {key}")
    summary = build_inventory(lock["config_payload"])
    access = {
        "fresh_language_read_count": 0,
        "protected_test_language_read_count": 0,
        "manual_language_or_raw_response_inspection_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "LLM_API_call_count": 0,
        "adapter_training_run_count": 0,
        "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    result = {
        "schema_version": "122-prequery-signal-inventory-result",
        "experiment": lock["config_payload"]["experiment"],
        "passed": summary["outcome_pass"],
        "decision": summary["decision"],
        "summary": summary,
        "access": access,
        "claim_boundary": lock["config_payload"]["claimBoundary"],
    }
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
