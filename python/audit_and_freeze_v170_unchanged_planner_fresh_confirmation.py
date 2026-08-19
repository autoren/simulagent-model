#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v170-unchanged-planner-fresh-confirmation.json"
    parent_path = PROJECT_ROOT / "configs/v169r1-json-key-normalization-repair-outcome-lock.json"
    source_planner_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    source_repair_path = PROJECT_ROOT / "configs/v167r1-history-action-metric-repair-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v170-unchanged-planner-fresh-confirmation-plan.md"
    protocol_path = PROJECT_ROOT / "python/v170_unchanged_planner_fresh_confirmation.py"
    tests_path = PROJECT_ROOT / "python/test_v170_unchanged_planner_fresh_confirmation.py"
    runner_path = PROJECT_ROOT / "python/run_v170_unchanged_planner_fresh_confirmation.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v170_unchanged_planner_fresh_confirmation_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v170_unchanged_planner_fresh_confirmation.py"
    audit_path = PROJECT_ROOT / "outputs/v170-unchanged-planner-fresh-confirmation/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v170-unchanged-planner-fresh-confirmation-lock.json"
    output_root = PROJECT_ROOT / "outputs/v170-unchanged-planner-fresh-confirmation/scoring"
    outcome_path = PROJECT_ROOT / "configs/v170-unchanged-planner-fresh-confirmation-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists(): raise RuntimeError("V170 already exists")
    config = json.loads(config_path.read_text()); parent = json.loads(parent_path.read_text())
    planner = json.loads(source_planner_path.read_text()); repair = json.loads(source_repair_path.read_text())
    contract = config["frozenPlannerContract"]; original = planner["config_payload"]
    states_path = PROJECT_ROOT / parent["constraint_states"]; eligible_path = PROJECT_ROOT / parent["eligible_state_ids"]
    exposure = config["preLockExposure"]; authority = config["authorityBoundary"]
    checks = {
        "V169r1_is_frozen_unscored_and_authorizes_separate_unchanged_planner": bool(
            valid_lock(parent) and parent["outcome"]["passed"]
            and parent["outcome"]["summary"]["planner_eligible_state_count"] == 58
            and parent["authorization"]["preregister_unchanged_V167_planner_on_all_eligible_states"]
            and not parent["authorization"]["score_planner_without_separate_lock"]
        ),
        "V167_and_corrected_metric_sources_are_exact": bool(
            valid_lock(planner) and valid_lock(repair) and repair["outcome"]["passed"]
            and repair["outcome"]["corrected_summary"]["history_dependent_second_action_case_count"] == 28
        ),
        "planner_contract_is_byte_for_semantic_field_unchanged": bool(
            contract["priorKind"] == original["prior"]["kind"]
            and contract["classMass"] == original["prior"]["classMass"]
            and contract["queryCost"] == original["queryModel"]["queryCost"]
            and contract["maximumQueries"] == original["queryModel"]["maximumQueries"]
            and contract["outcomeNoise"] == original["queryModel"]["outcomeNoise"]
            and contract["terminalDecisions"] == original["terminalDecisions"]
            and contract["terminalLoss"] == original["terminalLoss"]
            and set(contract["policies"]) == set(original["policies"])
        ),
        "all_frozen_eligible_membership_is_selected_without_policy_scores": bool(
            states_path.is_file() and eligible_path.is_file()
            and file_sha256(states_path) == parent["constraint_states_sha256"]
            and file_sha256(eligible_path) == parent["eligible_state_ids_sha256"]
            and config["caseSelection"]["useEveryV169PlannerEligibleState"]
            and config["caseSelection"]["requiredCaseCount"] == 58
            and not config["caseSelection"]["selectionMayUsePolicyOrRiskScore"]
        ),
        "integrity_and_non_tuning_outcome_rules_are_frozen": bool(
            config["integrityGates"]["requiredBayesNoWorseThanEveryNonOracleBaselineCaseRate"] == 1.0
            and not config["strongConfirmationThresholds"]["thresholdFailureInvalidatesIntegrity"]
            and config["strongConfirmationThresholds"]["mixedOrNegativeOutcomeMustBeRetainedWithoutTuning"]
            and not config["decisionRule"]["passAuthorizesPlannerTuning"]
        ),
        "prelock_and_authority_boundaries_hold": bool(
            exposure["implementationUnitPolicyScoreCount"] == 1
            and exposure["formalFreshPolicyScoreCount"] == 0
            and exposure["aggregateFreshMetricInspectionCount"] == 0
            and all(value == 0 for key, value in exposure.items() if key not in {"implementationUnitPolicyScoreCount", "formalFreshPolicyScoreCount", "aggregateFreshMetricInspectionCount"})
            and authority["allBeliefsQueriesAndDecisionsShadowOnly"]
            and authority["authoritativeOntologyAndStateImmutable"]
            and not authority["modelUseAllowed"] and not authority["provisionalRegistrationAllowed"]
            and not authority["trustedStateMutationAllowed"] and not authority["realServiceActionOrExecutionAllowed"]
        ),
        "required_locked_files_exist": all(path.is_file() for path in (config_path, parent_path, source_planner_path, source_repair_path, plan_path, protocol_path, tests_path, runner_path, verifier_path, auditor_path)),
        "formal_output_absent": not output_root.exists(),
    }
    passed = all(checks.values()); audit = {"schema_version": "170-unchanged-planner-fresh-confirmation-design-audit", "experiment": config["experiment"], "passed": passed, "checks": checks, "prelock_exposure": exposure}
    write_json(audit_path, audit)
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    deps = {"config": config_path, "parent_V169r1_outcome": parent_path, "source_V167_planner_lock": source_planner_path, "source_V167r1_outcome": source_repair_path, "constraint_states": states_path, "eligible_state_ids": eligible_path, "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path, "design_audit": audit_path}
    lock: dict[str, Any] = {"schema_version": "170-unchanged-planner-fresh-confirmation-lock", "experiment": config["experiment"], "config_payload": config, "V167_config_payload": original, "authorization": {"modify_planner_or_membership": False, "score_all_58_states_once": True, "tune_on_fresh_scores": False, "run_model_register_mutate_state_act_or_execute": False}}
    for key, path in deps.items(): lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock); write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
