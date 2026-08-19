#!/usr/bin/env python3
from __future__ import annotations

import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v119_asymmetric_adaptive_evidence import run_simulator
from audit_and_freeze_v119_asymmetric_adaptive_evidence import payload_hash, write_json


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v119-asymmetric-adaptive-evidence-lock.json"
    output_path = PROJECT_ROOT / "outputs/v119-asymmetric-adaptive-evidence/simulator/result.json"
    if output_path.exists(): raise RuntimeError("V119 simulator may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]: raise RuntimeError("V119 lock mismatch")
    dependencies = ("config", "parent_outcome", "parent_analysis_lock", "historical_population", "historical_model_result", "choice_catalog", "baseline_lock", "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit")
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V119 dependency drifted: {key}")
    population = json.loads((PROJECT_ROOT / lock["historical_population"]).read_text()); historical = json.loads((PROJECT_ROOT / lock["historical_model_result"]).read_text()); catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    access = {"fresh_language_read_count": 0, "protected_test_language_read_count": 0, "manual_language_or_raw_response_inspection_count": 0, "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0}
    summary = run_simulator(population, historical, catalog, lock["baseline_config_payload"], lock["config_payload"])
    result = {"schema_version": "119-asymmetric-adaptive-evidence-result", "experiment": lock["config_payload"]["experiment"], "passed": summary["outcome_pass"], "decision": summary["decision"], "summary": summary, "access": access, "claim_boundary": lock["config_payload"]["claimBoundary"]}
    write_json(output_path, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
