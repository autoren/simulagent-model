#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from audit_and_freeze_v122_prequery_signal_inventory import payload_hash, valid_lock, write_json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v130_clarification_evidence_strength import reliability_grid


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v130-clarification-evidence-strength.json"
    plan_path = PROJECT_ROOT / "docs/v130-clarification-evidence-strength-plan.md"
    protocol_path = PROJECT_ROOT / "python/v130_clarification_evidence_strength.py"
    tests_path = PROJECT_ROOT / "python/test_v130_clarification_evidence_strength.py"
    runner_path = PROJECT_ROOT / "python/run_v130_clarification_evidence_strength.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v130_clarification_evidence_strength_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v130_clarification_evidence_strength.py"
    audit_path = PROJECT_ROOT / "outputs/v130-clarification-evidence-strength/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v130-clarification-evidence-strength-lock.json"
    result_path = PROJECT_ROOT / "outputs/v130-clarification-evidence-strength/evaluation/result.json"
    if any(path.exists() for path in (audit_path, lock_path, result_path)): raise RuntimeError("V130 already frozen or evaluated")
    config = json.loads(config_path.read_text()); parent_path = PROJECT_ROOT / config["parentV129OutcomeLock"]
    parent = json.loads(parent_path.read_text()); parent_lock_path = PROJECT_ROOT / parent["analysis_lock"]
    parent_lock = json.loads(parent_lock_path.read_text()); auth = parent["authorization"]
    catalog_path = PROJECT_ROOT / config["choiceCatalog"]; baseline_path = PROJECT_ROOT / config["baselineConfig"]
    checks = {
        "V129_is_valid_negative_and_authorizes_no_realization": bool(
            valid_lock(parent) and valid_lock(parent_lock) and parent["outcome"]["passed"]
            and parent["outcome"]["audit_pass"] and not parent["outcome"]["experimental_pass"]
            and auth["keep_language_model_and_human_channel_closed"]
            and not auth["preregister_complete_clarification_realization_audit"]
            and not auth["run_API_training_action_or_execution"]
        ),
        "reliability_and_multi_answer_grids_are_exact": bool(
            len(reliability_grid(config["singleAnswerReliabilityGrid"])) == 101
            and config["multiAnswerGrid"]["answerCounts"] == [1, 2, 3]
            and config["multiAnswerGrid"]["commonShockCorrelations"] == [0.0, 0.25, 0.5]
            and config["multiAnswerGrid"]["costPerAnswer"] == config["completeClarificationChannel"]["totalCost"] == 0.3
            and config["multiAnswerGrid"]["repeatedSamplesFromOneModelAreNotDeclaredIndependent"]
        ),
        "feasibility_rule_is_noncompensatory_and_frozen": bool(
            config["feasibilityRule"]["maximumSingleAnswerReliability"] == 0.99
            and config["feasibilityRule"]["maximumIndependentAnswerCount"] == 2
            and config["feasibilityRule"]["maximumRequiredCommonShockCorrelation"] == 0.25
            and config["feasibilityRule"]["eitherRouteMayAuthorizeRealizationAudit"]
        ),
        "zero_language_model_authority_and_execution": all(value == 0 for value in config["accessGates"].values()) and not config["decisionRule"]["passAuthorizesLanguageHumanOrModelRun"] and not config["decisionRule"]["passAuthorizesProtectedInductionOrRicherPlanning"] and not config["decisionRule"]["passAuthorizesAPITrainingActionOrExecution"],
        "dependencies_code_and_outputs_hold": all(path.is_file() for path in (catalog_path, baseline_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)) and not result_path.exists(),
    }
    passed = all(checks.values()); audit = {"schema_version": "130-clarification-evidence-strength-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "decision": "freeze_and_authorize_one_model_free_evidence_strength_audit" if passed else "reject_V130_design", "prelock_access": {key: 0 for key in config["accessGates"]} | {"actual_execution_count": 0}}
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"config": config_path, "parent_outcome": parent_path, "parent_analysis_lock": parent_lock_path, "choice_catalog": catalog_path, "baseline_config": baseline_path, "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "130-clarification-evidence-strength-lock", "experiment": config["experiment"], "config_payload": config, "authorization": {"run_one_model_free_evidence_strength_audit": True, "modify_grids_channel_cost_gates_or_decision": False, "read_language_records_or_load_model": False, "grant_protected_induction_authority_or_execution": False}}
    for key, path in deps.items(): lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
