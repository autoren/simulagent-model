#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v201r1_elapsed_time_verification_repair import evaluate_repair
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v201r1-elapsed-time-verification-repair-lock.json"; lock = json.loads(lock_path.read_text())
    output_root = PROJECT_ROOT / "outputs/v201r1-elapsed-time-verification-repair/repair"; audit_path = PROJECT_ROOT / "outputs/v201r1-elapsed-time-verification-repair/outcome-audit.json"
    outcome_path = PROJECT_ROOT / "configs/v201r1-elapsed-time-verification-repair-outcome-lock.json"; results_path = PROJECT_ROOT / "docs/v201r1-elapsed-time-verification-repair-results.md"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V201r1 outcome already frozen")
    dependencies_exact = valid_lock(lock) and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock)
    config = lock["config_payload"]; source_lock = json.loads((PROJECT_ROOT / lock["source_V201_lock"]).read_text())
    repair = evaluate_repair(json.loads((PROJECT_ROOT / lock["source_failed_outcome_audit"]).read_text()), json.loads((PROJECT_ROOT / lock["source_result"]).read_text()), json.loads((PROJECT_ROOT / lock["source_evaluation_summary"]).read_text()), json.loads((PROJECT_ROOT / lock["source_access"]).read_text()), source_lock["config_payload"], config)
    decision = config["decisionRule"]["ifExactSingleFieldRepairAndAllOtherVerificationChecksPass" if repair["passed"] else "otherwise"]
    result_path = output_root / "result.json"; result = json.loads(result_path.read_text())
    result_exact = bool(result["passed"] == repair["passed"] and result["checks"] == repair["checks"] and result["different_summary_keys"] == repair["different_summary_keys"] and result["elapsed_seconds_delta"] == repair["elapsed_seconds_delta"] and result["qualified"] == repair["qualified"] and result["decision"] == decision and result["source_artifact_mutation_count"] == 0 and result["model_or_policy_rerun_count"] == 0)
    checks = {"design_lock_and_dependencies_are_exact": dependencies_exact, "repair_reconstructs_exactly": result_exact, "single_field_repair_passes": repair["passed"], "source_hashes_remain_exact": all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in ("source_result", "source_evaluation_summary", "source_scored_records", "source_access", "source_census_result")), "results_document_exists": results_path.is_file(), "zero_model_raw_API_mutation_and_execution": result["model_or_policy_rerun_count"] == result["raw_model_response_read_count"] == result["source_artifact_mutation_count"] == result["API_call_count"] == result["actual_execution_count"] == 0}
    passed = all(checks.values()); audit = {"schema_version": "201r1-elapsed-time-verification-repair-outcome-audit", "experiment": lock["experiment"], "passed": passed, "decision": "freeze_verified_V201r1_repair" if passed else "freeze_failed_V201r1_verification", "checks": checks, "repair": repair}
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"repair_lock": lock_path, "audit": audit_path, "result": result_path, "source_V201_lock": PROJECT_ROOT / lock["source_V201_lock"], "source_V201_result": PROJECT_ROOT / lock["source_result"], "source_V201_summary": PROJECT_ROOT / lock["source_evaluation_summary"], "source_V201_scored_records": PROJECT_ROOT / lock["source_scored_records"], "source_V201_access": PROJECT_ROOT / lock["source_access"], "source_V201_census": PROJECT_ROOT / lock["source_census_result"], "source_V201_results_document": PROJECT_ROOT / lock["source_results_document"], "results_document": results_path, "verifier": PROJECT_ROOT / lock["verifier"]}
    source_summary = json.loads((PROJECT_ROOT / lock["source_evaluation_summary"]).read_text())
    outcome: dict[str, Any] = {"schema_version": "201r1-elapsed-time-verification-repair-outcome-lock", "experiment": lock["experiment"], "outcome": {"passed": True, "V201_scientific_qualification_gates_passed": False, "decision": decision, "V201_summary": source_summary}, "authorization": {"update_roadmap_and_preregister_separate_model_free_decision_sufficiency_design": True, "run_paired_protected_robustness": False, "run_API_additional_model_synthetic_language_registration_authority_action_or_execution": False}}
    for key, path in deps.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); write_json(outcome_path, outcome)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"outcome_lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
