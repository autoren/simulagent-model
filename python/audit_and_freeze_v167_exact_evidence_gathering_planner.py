#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner.json"
    parent_path = PROJECT_ROOT / "configs/v166-model-free-factored-ontology-baselines-outcome-lock.json"
    roadmap_path = PROJECT_ROOT / "docs/research-roadmap-after-v166.md"
    plan_path = PROJECT_ROOT / "docs/v167-exact-evidence-gathering-planner-plan.md"
    protocol_path = PROJECT_ROOT / "python/v167_exact_evidence_gathering_planner.py"
    tests_path = PROJECT_ROOT / "python/test_v167_exact_evidence_gathering_planner.py"
    runner_path = PROJECT_ROOT / "python/run_v167_exact_evidence_gathering_planner.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v167_exact_evidence_gathering_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v167_exact_evidence_gathering_planner.py"
    audit_path = PROJECT_ROOT / "outputs/v167-exact-evidence-gathering/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-lock.json"
    output_root = PROJECT_ROOT / "outputs/v167-exact-evidence-gathering/planner"
    outcome_path = PROJECT_ROOT / "configs/v167-exact-evidence-gathering-planner-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V167 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    disclosure = config["developmentDesignDisclosure"]
    gates = config["plannerGates"]
    authority = config["authorityBoundary"]
    exposure = config["formalRunPreLockAccess"]
    input_paths = {
        "baseline_predictions": PROJECT_ROOT / parent["baseline_predictions"],
        "hidden_records": PROJECT_ROOT / parent["hidden_records"],
    }
    checks = {
        "V166_is_exact_zero_residual_and_authorizes_separate_planner_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["model_eligible_residual_count"] == 0
            and parent["outcome"]["intentionally_ambiguous_record_count"] == 48
            and parent["outcome"]["intentionally_ambiguous_candidate_count"] == 64
            and parent["authorization"]["preregister_evidence_gathering_planner_on_frozen_ambiguous_states"]
            and not parent["authorization"]["run_evidence_planner_or_sandbox_without_separate_lock"]
            and not parent["authorization"]["run_local_or_API_model_on_zero_residual"]
        ),
        "frozen_inputs_match_parent": all(
            path.is_file() and file_sha256(path) == parent[f"{key}_sha256"]
            for key, path in input_paths.items()
        ),
        "development_informed_design_is_explicitly_disclosed": bool(
            disclosure["projectAuthoredDevelopmentOnly"]
            and not disclosure["protectedOrEvaluationPopulationExists"]
            and disclosure["hiddenDevelopmentTruthReadCount"] == 2
            and disclosure["oracleFeasibilityAnalysisRunCount"] == 2
            and disclosure["costsHorizonAndLossesChosenAfterDevelopmentFeasibilityInspection"]
            and not disclosure["formalPolicyScoresWrittenBeforeLock"]
            and not disclosure["confirmatoryClaimAllowed"]
        ),
        "prior_query_loss_and_policy_contract_are_complete": bool(
            config["prior"]["kind"] == "class_balanced_then_uniform_within_class"
            and set(config["prior"]["classMass"].values()) == {"1/3"}
            and config["queryModel"]["outcomeNoise"] == 0.0
            and config["queryModel"]["queryCost"] == "1/10"
            and config["queryModel"]["maximumQueries"] == 2
            and config["queryModel"]["earlyStopAllowed"]
            and len(config["policies"]) == 7
            and config["terminalLoss"]["alias"]["provisional_primitive"] == 12
            and config["terminalLoss"]["provisional_primitive"]["defer"] == 2
        ),
        "planner_gates_require_information_value_adaptation_and_control_dominance": bool(
            gates["requiredCaseCount"] == 48
            and gates["requiredCandidatesPerCase"] == 64
            and gates["requiredPositiveValueOfInformationCaseCount"] == 48
            and gates["minimumUniqueBayesRootQueryCount"] >= 2
            and gates["minimumHistoryDependentSecondActionCaseCount"] >= 1
            and gates["minimumStrictBayesImprovementOverOpenLoopCaseCount"] >= 1
            and gates["requiredBayesNoWorseThanEveryNonOracleBaselineCaseRate"] == 1.0
            and gates["requiredRenamingRiskInvariance"] == 1.0
        ),
        "authority_and_formal_prelock_boundaries_are_closed": bool(
            authority["candidateBeliefsAndTerminalDecisionsAreShadowOnly"]
            and authority["authoritativeOntologyImmutable"]
            and not authority["provisionalRegistrationAllowed"]
            and not authority["trustedStateMutationAllowed"]
            and not authority["realQueryOrServiceCallAllowed"]
            and not authority["modelUseAllowed"]
            and not authority["actionOrExecutionAllowed"]
            and authority["realExecutionCount"] == 0
            and all(value == 0 for value in exposure.values())
            and not config["decisionRule"]["passAuthorizesImmediateSandboxRun"]
            and not config["decisionRule"]["passAuthorizesModelOrEvaluationPopulation"]
            and not config["decisionRule"]["passAuthorizesRegistrationTrustedStateActionOrExecution"]
        ),
        "required_locked_files_exist": all(path.is_file() for path in (
            config_path, parent_path, roadmap_path, plan_path, protocol_path,
            tests_path, runner_path, verifier_path, auditor_path,
        )),
        "formal_output_absent_before_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "167-exact-evidence-gathering-planner-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_formal_development_planner_run" if passed else "reject_V167_design",
        "checks": checks,
        "development_design_disclosure": disclosure,
        "formal_prelock_access": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V166_outcome": parent_path,
        "roadmap": roadmap_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
        **input_paths,
    }
    lock: dict[str, Any] = {
        "schema_version": "167-exact-evidence-gathering-planner-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_prior_queries_costs_losses_horizon_policies_metrics_gates_or_decision": False,
            "read_frozen_predictions_and_hidden_development_truth_once": True,
            "run_formal_exact_development_planner_once": True,
            "create_or_open_evaluation_population": False,
            "load_or_run_local_or_API_model": False,
            "register_provisional_primitive": False,
            "mutate_trusted_state_or_call_real_service": False,
            "grant_candidate_state_belief_action_or_execution_authority": False,
            "perform_external_side_effect_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
