#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def independent_checks(result, config):
    gates = config["outcomeGates"]; required = f"{config['completeClarificationChannel']['requiredReliability']:.2f}"
    conditions = result["summary"]["conditions"]; comparator = result["summary"]["candidate_specific_comparator"]
    aware = [(prior["id"], regime, conditions[prior["id"]][required][regime]["channel_aware"]) for prior in config["priorRegimes"] for regime in config["completeClarificationChannel"]["errorRegimes"]]
    misspecified = [conditions[prior["id"]][required][regime]["symmetric_assumed"] for prior in config["priorRegimes"] for regime in ("candidate_attraction", "abstention_attraction")]
    perfect = [conditions[prior["id"]]["1.00"][regime]["channel_aware"] for prior in config["priorRegimes"] for regime in config["completeClarificationChannel"]["errorRegimes"]]
    return {
        "aware_regret_every_prior_and_bias": all(row["mean_regret"] <= gates["maximumAwareMeanRegretEveryPriorAndBiasAtRequiredReliability"] for _, _, row in aware),
        "aware_known_exact_every_prior_and_bias": all(row["known_exact_probability"] >= gates["minimumAwareKnownExactEveryPriorAndBiasAtRequiredReliability"] for _, _, row in aware),
        "aware_unsupported_every_prior_and_bias": all(row["unsupported_correct_probability"] >= gates["minimumAwareUnsupportedEveryPriorAndBiasAtRequiredReliability"] for _, _, row in aware),
        "aware_false_known_every_prior_and_bias": all(row["false_known_probability"] <= gates["maximumAwareFalseKnownEveryPriorAndBiasAtRequiredReliability"] for _, _, row in aware),
        "aware_no_worse_regret_than_candidate_specific": all(row["mean_regret"] <= comparator[prior_id]["mean_regret"] for prior_id, _, row in aware),
        "aware_no_worse_known_than_candidate_specific": all(row["known_exact_probability"] >= comparator[prior_id]["known_exact_probability"] for prior_id, _, row in aware),
        "symmetric_assumed_regret_under_bias": all(row["mean_regret"] <= gates["maximumSymmetricAssumedRegretEveryPriorAndBiasedRegime"] for row in misspecified),
        "symmetric_assumed_false_known_under_bias": all(row["false_known_probability"] <= gates["maximumSymmetricAssumedFalseKnownEveryPriorAndBiasedRegime"] for row in misspecified),
        "perfect_answer_known_exact": all(row["known_exact_probability"] == gates["requiredPerfectAnswerKnownExact"] for row in perfect),
        "perfect_answer_unsupported": all(row["unsupported_correct_probability"] == gates["requiredPerfectAnswerUnsupported"] for row in perfect),
        "complete_hypothesis_retention": result["summary"]["true_hypothesis_retention"] == gates["requiredTrueHypothesisRetention"],
        "zero_individual_pair_emission": result["summary"]["individual_pair_emission_count"] == gates["maximumIndividualPairEmissionCount"],
        "zero_execution": result["summary"]["actual_execution_count"] == gates["maximumActualExecutionCount"],
    }


def main() -> None:
    lock_path = PROJECT_ROOT / "configs/v129-complete-clarification-interface-lock.json"; result_path = PROJECT_ROOT / "outputs/v129-complete-clarification-interface/evaluation/result.json"; doc_path = PROJECT_ROOT / "docs/v129-complete-clarification-interface-results.md"; audit_path = PROJECT_ROOT / "outputs/v129-complete-clarification-interface/outcome-audit.json"; outcome_path = PROJECT_ROOT / "configs/v129-complete-clarification-interface-outcome-lock.json"; verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v129_complete_clarification_interface_outcome.py"
    if audit_path.exists() or outcome_path.exists(): raise RuntimeError("V129 outcome already frozen")
    if not doc_path.is_file(): raise RuntimeError("write V129 result document first")
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text()); dependencies = [key for key in lock if not key.endswith("_sha256") and f"{key}_sha256" in lock]; independent = independent_checks(result, lock["config_payload"])
    checks = {"lock_and_dependencies_exact": payload_hash({key: value for key, value in lock.items() if key != "lock_payload_sha256"}) == lock["lock_payload_sha256"] and all(file_sha256(PROJECT_ROOT / lock[key]) == lock[f"{key}_sha256"] for key in dependencies), "aggregate_gate_classification_exact": independent == result["summary"]["outcome_gates"] and result["passed"] == all(independent.values()), "deterministic_recomputation_recorded": result["deterministic_recomputation_exact"], "zero_language_record_model_and_execution_access": all(value == 0 for value in result["access"].values()) and result["summary"]["actual_execution_count"] == 0}
    passed = all(checks.values()); audit = {"schema_version": "129-complete-clarification-interface-outcome-audit", "experiment": lock["config_payload"]["experiment"], "passed": passed, "checks": checks, "independent_aggregate_gate_checks": independent}; write_json(audit_path, audit)
    if not passed: raise SystemExit(1)
    experimental_pass = bool(result["passed"]); paths = {"analysis_lock": lock_path, "result": result_path, "verifier": verifier_path, "audit": audit_path, "results_document": doc_path}
    outcome: dict[str, Any] = {"schema_version": "129-complete-clarification-interface-outcome-lock", "experiment": "v129_complete_clarification_interface_outcome_lock", "outcome": {"passed": True, "audit_pass": True, "experimental_pass": experimental_pass, "decision": result["decision"], "summary": result["summary"]}, "authorization": {"modify_rerun_retune_or_mine_V129": False, "preregister_complete_clarification_realization_audit": experimental_pass, "keep_language_model_and_human_channel_closed": True, "begin_induction_or_richer_planning": False, "run_API_training_action_or_execution": False}}
    for key, path in paths.items(): outcome[key] = str(path.relative_to(PROJECT_ROOT)); outcome[f"{key}_sha256"] = file_sha256(path)
    outcome["lock_payload_sha256"] = payload_hash(outcome); write_json(outcome_path, outcome); print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(outcome_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(outcome_path)}, indent=2))


if __name__ == "__main__": main()
