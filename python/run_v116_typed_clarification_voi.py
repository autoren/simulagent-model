#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v116_typed_clarification_voi import run_audit
from audit_and_freeze_v116_typed_clarification_voi import payload_hash, write_json


def access_gates(access: dict[str, int], config: dict[str, Any]) -> dict[str, bool]:
    limits = config["accessGates"]
    mapping = {
        "fresh_language_read_count": "maximumFreshLanguageReadCount",
        "protected_test_language_read_count": "maximumProtectedTestLanguageReadCount",
        "manual_language_or_raw_response_inspection_count": "maximumManualLanguageOrRawResponseInspectionCount",
        "model_load_count": "maximumModelLoadCount", "model_generation_count": "maximumModelGenerationCount",
        "LLM_API_call_count": "maximumLLMAPICallCount", "adapter_training_run_count": "maximumAdapterTrainingRunCount",
        "real_service_call_count": "maximumRealServiceCallCount", "external_side_effect_count": "maximumExternalSideEffectCount",
    }
    return {key: access[key] <= limits[limit] for key, limit in mapping.items()}


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v116-typed-clarification-voi-lock.json"
    output_path = PROJECT_ROOT / "outputs/v116-typed-clarification-voi/audit/result.json"
    if output_path.exists():
        raise RuntimeError("V116 audit may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]:
        raise RuntimeError("V116 lock mismatch")
    dependency_keys = (
        "config", "parent_outcome", "parent_analysis_lock", "historical_population",
        "historical_model_result", "choice_catalog", "baseline_lock", "plan", "protocol",
        "tests", "runner", "verifier", "auditor", "design_audit",
    )
    for key in dependency_keys:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]:
            raise RuntimeError(f"V116 dependency drifted: {key}")
    population = json.loads((PROJECT_ROOT / lock["historical_population"]).read_text())
    historical_result = json.loads((PROJECT_ROOT / lock["historical_model_result"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    access = {
        "fresh_language_read_count": 0, "protected_test_language_read_count": 0,
        "manual_language_or_raw_response_inspection_count": 0, "model_load_count": 0,
        "model_generation_count": 0, "LLM_API_call_count": 0,
        "adapter_training_run_count": 0, "real_service_call_count": 0,
        "external_side_effect_count": 0,
    }
    summary = run_audit(
        population, historical_result, catalog, lock["baseline_config_payload"], lock["config_payload"],
    )
    gates = access_gates(access, lock["config_payload"])
    result = {
        "schema_version": "116-typed-clarification-voi-result",
        "experiment": lock["config_payload"]["experiment"], "passed": all(gates.values()),
        "summary": summary, "access": access, "access_gates": gates,
        "decision": summary["decision"],
        "claim_boundary": lock["config_payload"]["claimBoundary"],
        "individual_record_emission_count": 0,
    }
    write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
