#!/usr/bin/env python3
import json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v121_prequery_value_selectivity_envelope import run_audit
from audit_and_freeze_v121_prequery_value_selectivity_envelope import payload_hash, write_json


def main():
    lock_path = PROJECT_ROOT / "configs/v121-prequery-value-selectivity-envelope-lock.json"; output_path = PROJECT_ROOT / "outputs/v121-prequery-value-selectivity-envelope/audit/result.json"
    if output_path.exists(): raise RuntimeError("V121 audit may run only once")
    lock = json.loads(lock_path.read_text()); deps = ("config", "parent_outcome", "parent_analysis_lock", "parent_result", "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit")
    if payload_hash({k: v for k, v in lock.items() if k != "lock_payload_sha256"}) != lock["lock_payload_sha256"]: raise RuntimeError("V121 lock mismatch")
    for key in deps:
        if file_sha256(PROJECT_ROOT / lock[key]) != lock[f"{key}_sha256"]: raise RuntimeError(f"V121 dependency drifted: {key}")
    parent = json.loads((PROJECT_ROOT / lock["parent_result"]).read_text()); summary = run_audit(parent, lock["config_payload"]); access = {"fresh_language_read_count": 0, "protected_test_language_read_count": 0, "manual_language_or_raw_response_inspection_count": 0, "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0}
    result = {"schema_version": "121-prequery-value-selectivity-envelope-result", "experiment": lock["config_payload"]["experiment"], "passed": summary["outcome_pass"], "decision": summary["decision"], "summary": summary, "access": access, "claim_boundary": lock["config_payload"]["claimBoundary"]}; write_json(output_path, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
