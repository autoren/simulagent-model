#!/usr/bin/env python3
import json

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v123_fresh_population_availability_audit import run_audit


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v123-fresh-population-availability-audit-lock.json"
    output_path = PROJECT_ROOT / "outputs/v123-fresh-population-availability-audit/audit/result.json"
    if output_path.exists():
        raise RuntimeError("V123 may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V123 lock mismatch")
    dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V123 dependency drifted: {key}")
    config = lock["config_payload"]
    inventory = json.loads((PROJECT_ROOT / lock["source_inventory"]).read_text())
    excluded = [
        json.loads((PROJECT_ROOT / lock[key]).read_text())
        for key in sorted(key for key in lock if key.startswith("excluded_population_") and not key.endswith("_sha256"))
    ]
    summary = run_audit(inventory, excluded, config)
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
        "schema_version": "123-fresh-population-availability-result",
        "experiment": config["experiment"],
        "passed": summary["outcome_pass"],
        "decision": summary["decision"],
        "summary": summary,
        "access": access,
        "claim_boundary": config["claimBoundary"],
    }
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
