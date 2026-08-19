#!/usr/bin/env python3
import json
from typing import Any
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v121_prequery_value_selectivity_envelope import run_audit
from audit_and_freeze_v121_prequery_value_selectivity_envelope import payload_hash


def main():
    lock_path = PROJECT_ROOT / "configs/v121-prequery-value-selectivity-envelope-lock.json"; result_path = PROJECT_ROOT / "outputs/v121-prequery-value-selectivity-envelope/audit/result.json"; doc_path = PROJECT_ROOT / "docs/v121-prequery-value-selectivity-envelope-results.md"; audit_path = PROJECT_ROOT / "outputs/v121-prequery-value-selectivity-envelope/outcome-audit.json"; outcome_path = PROJECT_ROOT / "configs/v121-prequery-value-selectivity-envelope-outcome-lock.json"; verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v121_prequery_value_selectivity_envelope_outcome.py"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V121 outcome already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V121 result document first")
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text()); parent = json.loads((PROJECT_ROOT / lock["parent_result"]).read_text()); summary = run_audit(parent, lock["config_payload"]); deps = ("config", "parent_outcome", "parent_analysis_lock", "parent_result", "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit")
    checks = {"lock_and_dependencies_exact": payload_hash({k: v for k, v in lock.items() if k != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[k]) == lock[f"{k}_sha256"] for k in deps), "summary_exact": summary == result["summary"] and result["passed"] == summary["outcome_pass"], "zero_access_and_execution": all(v == 0 for v in result["access"].values()) and summary["actual_execution_count"] == 0, "aggregate_only_and_no_trigger_claim": summary["individual_record_read_count"] == 0 and summary["individual_record_emission_count"] == 0 and not summary["trigger_certified"]}
    passed = all(checks.values()); audit = {"schema_version": "121-prequery-value-selectivity-outcome-audit", "experiment": lock["config_payload"]["experiment"], "passed": passed, "checks": checks, "independent_summary": summary}; audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed: raise SystemExit(1)
    paths = {"analysis_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}; outcome: dict[str, Any] = {"schema_version": "121-prequery-value-selectivity-outcome-lock", "experiment": "v121_prequery_value_selectivity_outcome_lock", "outcome": {"passed": True, "audit_pass": summary["outcome_pass"], "decision": summary["decision"], "summary": summary}, "authorization": {"modify_rerun_or_retune_V121": False, "inventory_independently_available_prequery_signals": bool(summary["outcome_pass"]), "evaluate_or_fit_trigger": False, "run_language_model_or_protected": False, "retune_cost_or_V119": False, "begin_induction_or_richer_planning": False, "run_API_training_action_or_execution": False}}
    for key, path in paths.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n"); print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
