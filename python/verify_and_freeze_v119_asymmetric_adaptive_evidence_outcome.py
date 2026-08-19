#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v119_asymmetric_adaptive_evidence import run_simulator
from audit_and_freeze_v119_asymmetric_adaptive_evidence import payload_hash


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v119-asymmetric-adaptive-evidence-lock.json"; result_path = PROJECT_ROOT / "outputs/v119-asymmetric-adaptive-evidence/simulator/result.json"; doc_path = PROJECT_ROOT / "docs/v119-asymmetric-adaptive-evidence-results.md"; audit_path = PROJECT_ROOT / "outputs/v119-asymmetric-adaptive-evidence/outcome-audit.json"; outcome_path = PROJECT_ROOT / "configs/v119-asymmetric-adaptive-evidence-outcome-lock.json"; verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v119_asymmetric_adaptive_evidence_outcome.py"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V119 outcome already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V119 result document before freezing")
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text()); population = json.loads((PROJECT_ROOT / lock["historical_population"]).read_text()); historical = json.loads((PROJECT_ROOT / lock["historical_model_result"]).read_text()); catalog = json.loads((PROJECT_ROOT / lock["choice_catalog"]).read_text())
    summary = run_simulator(population, historical, catalog, lock["baseline_config_payload"], lock["config_payload"])
    dependencies = ("config", "parent_outcome", "parent_analysis_lock", "historical_population", "historical_model_result", "choice_catalog", "baseline_lock", "plan", "protocol", "tests", "runner", "verifier", "auditor", "design_audit")
    checks = {"lock_and_dependencies_exact": payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies), "summary_decision_and_pass_reconstruct": summary == result["summary"] and result["decision"] == summary["decision"] and result["passed"] == summary["outcome_pass"], "zero_language_model_API_training_service_effect_and_execution": all(value == 0 for value in result["access"].values()) and summary["actual_execution_count"] == 0, "aggregate_only_and_all_hypotheses_retained": summary["individual_record_emission_count"] == 0 and summary["true_hypothesis_retention"] == 1.0}
    passed = all(checks.values()); audit = {"schema_version": "119-asymmetric-adaptive-evidence-outcome-audit", "experiment": lock["config_payload"]["experiment"], "passed": passed, "checks": checks, "independent_summary": summary}; audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"analysis_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {"schema_version": "119-asymmetric-adaptive-evidence-outcome-lock", "experiment": "v119_asymmetric_adaptive_evidence_outcome_lock", "outcome": {"passed": True, "simulator_pass": summary["outcome_pass"], "decision": summary["decision"], "summary": summary}, "authorization": {"modify_rerun_or_retune_V119": False, "preregister_real_mechanism_realization_audit": bool(summary["outcome_pass"]), "run_language_or_model": False, "open_protected_test": False, "begin_schema_induction_or_richer_planning": False, "run_API_or_training": False, "grant_capability_belief_action_or_execution_authority": False}}
    for key, path in deps.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); outcome_path.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n"); print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
