#!/usr/bin/env python3
from __future__ import annotations

import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v118_evidence_identifiability_audit import run_audit
from audit_and_freeze_v118_evidence_identifiability_audit import payload_hash, write_json


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v118-evidence-identifiability-audit-lock.json"
    output_path = PROJECT_ROOT / "outputs/v118-evidence-identifiability-audit/audit/result.json"
    if output_path.exists(): raise RuntimeError("V118 audit may run only once")
    lock = json.loads(lock_path.read_text())
    if payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) != lock["lock_payload_sha256"]: raise RuntimeError("V118 lock mismatch")
    dependencies = ("config", "parent_outcome", "parent_analysis_lock", "parent_result", "historical_population", "historical_model_result", "choice_catalog", "baseline_lock", "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit")
    for key in dependencies:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V118 dependency drifted: {key}")
    population = json.loads((PROJECT_ROOT / lock["historical_population"]).read_text())
    historical = json.loads((PROJECT_ROOT / lock["historical_model_result"]).read_text())
    catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    v117_result = json.loads((PROJECT_ROOT / lock["parent_result"]).read_text())
    access = {"fresh_language_read_count": 0, "protected_test_language_read_count": 0, "manual_language_or_raw_response_inspection_count": 0, "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0}
    summary = run_audit(population, historical, catalog, lock["baseline_config_payload"], lock["v117_config_payload"], lock["config_payload"], v117_result)
    result = {"schema_version": "118-evidence-identifiability-audit-result", "experiment": lock["config_payload"]["experiment"], "passed": summary["outcome_pass"], "decision": summary["decision"], "summary": summary, "access": access, "claim_boundary": lock["config_payload"]["claimBoundary"]}
    write_json(output_path, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
