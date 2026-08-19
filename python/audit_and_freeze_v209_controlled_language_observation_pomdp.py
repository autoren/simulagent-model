#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v209-controlled-language-observation-pomdp.json"
    plan_path = PROJECT_ROOT / "docs/v209-controlled-language-observation-pomdp-plan.md"
    protocol_path = PROJECT_ROOT / "python/v209_controlled_language_observation_pomdp.py"
    tests_path = PROJECT_ROOT / "python/test_v209_controlled_language_observation_pomdp.py"
    runner_path = PROJECT_ROOT / "python/run_v209_controlled_language_observation_pomdp.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v209_controlled_language_observation_pomdp_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v209_controlled_language_observation_pomdp.py"
    audit_path = PROJECT_ROOT / "outputs/v209-controlled-language-observation-pomdp/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v209-controlled-language-observation-pomdp-lock.json"
    output_root = PROJECT_ROOT / "outputs/v209-controlled-language-observation-pomdp/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v209-controlled-language-observation-pomdp-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V209 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV208OutcomeLock"]
    reference_path = PROJECT_ROOT / config["referenceV205OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    reference = json.loads(reference_path.read_text())
    grammar = config["grammar"]
    channel = config["channel"]
    process = config["decisionProcess"]
    gates = config["oracleGates"]
    access = config["accessGates"]
    prelock = config["preLockExposure"]
    checks = {
        "V208_is_valid_frozen_negative_and_Track_F_is_parked": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_feasibility_passed"]
            and parent["authorization"]["park_external_behavioral_abstention_if_negative"]
            and not parent["authorization"]["preregister_exact_identifier_selection_and_text_extraction_only"]
        ),
        "V205_is_valid_positive_terminally_proper_reference_only": bool(
            valid_lock(reference)
            and reference["outcome"]["passed"]
            and reference["outcome"]["scientific_oracle_passed"]
            and not reference["authorization"]["external_candidate_language_or_model_run"]
        ),
        "V209_is_a_separate_controlled_language_process": bool(
            config["schemaVersion"] == "209-controlled-language-observation-POMDP-design"
            and process["stageNames"] == ["PRE_REFERENCE", "POST_REFERENCE", "POST_TARGET"]
            and grammar["clarificationActions"] == ["ask_reference", "ask_target"]
            and len(grammar["semanticObservationIds"]) == 3
            and process["maximumControllableDecisionCount"] == 3
        ),
        "three_regimes_include_outside_semantics_and_full_retention": bool(
            config["hypotheses"]["semanticRegimes"] == ["CANONICAL", "ALTERNATIVE", "OUTSIDE_UNKNOWN"]
            and len(config["hypotheses"]["semanticRegimePrior"]) == gates["requiredLatentRegimeCount"] == 3
            and abs(sum(config["hypotheses"]["semanticRegimePrior"]) - 1.0) <= 1e-12
            and abs(sum(config["hypotheses"]["taskStatePrior"]) - 1.0) <= 1e-12
            and not config["hypotheses"]["priorComesFromModelScoresOrRanks"]
            and config["hypotheses"]["fullHypothesisUniverseAlwaysRetained"]
        ),
        "finite_grammar_counterfactual_and_renaming_contract_is_fixed": bool(
            set(grammar["surfaceFamilies"]) == {"DIRECT", "MATCHED_PARAPHRASE"}
            and grammar["matchedSurfaceCounterfactualFamilies"] == ["DIRECT", "MATCHED_PARAPHRASE"]
            and sorted(grammar["observationRenamingPermutation"]) == [0, 1, 2]
            and grammar["surfaceProjectionIsDeterministicAndNonAuthoritative"]
            and grammar["surfaceTextNeverEntersPlanner"]
            and all(
                set(grammar["surfaceFamilies"][family][action]) == set(grammar["semanticObservationIds"])
                for family in grammar["surfaceFamilies"]
                for action in grammar["clarificationActions"]
            )
        ),
        "normalized_common_support_history_dependent_channel_is_frozen": bool(
            channel["everyActionRegimeStateHistoryDistributionIsNormalized"]
            and channel["commonPositiveSupport"]
            and 0.0 < channel["postReferenceHistoryMixWeight"] < 1.0
            and set(channel["postReferenceHistoryAnchors"]) == set(grammar["semanticObservationIds"])
            and min(
                channel["knownLogicalA"]
                + channel["knownLogicalB"]
                + channel["outsideEitherState"]
                + sum(channel["postReferenceHistoryAnchors"].values(), [])
            ) > 0.0
        ),
        "terminal_contract_is_structural_and_nonoptional": bool(
            process["controlAlwaysTransitionsThroughAutomaticSettlement"]
            and process["automaticSettlementConsumesNoControllableDecision"]
            and process["unfinishedClarificationAlwaysTerminatesBySafeDeferral"]
            and process["unfinishedClarificationTerminalValue"] == process["safeDeferralReward"]
            and process["controlImmediateReward"] == gates["requiredControlImmediateReward"] == 0.0
            and process["allowedActionsByStage"]["POST_TARGET"] == ["act_A", "act_B", "defer"]
        ),
        "comparators_and_noncompensatory_gates_are_fixed": bool(
            len(config["fixedComparators"]) == 8
            and gates["requiredSelectedExactRootAction"] == "ask_reference"
            and gates["requiredExactActionAfterRootAlpha"] == "ask_target"
            and gates["requiredExactActionAfterRootBeta"] == "ask_target"
            and gates["requiredExactActionAfterRootUnresolved"] == "defer"
            and gates["requiredClosedWorldActionAfterRootUnresolved"] == "ask_target"
            and gates["minimumDistinctReachableControlActions"] == 2
            and gates["maximumUnsettledControlTerminalCount"] == 0
            and gates["maximumHorizonEscapePathCount"] == 0
        ),
        "prelock_exact_external_language_model_API_training_and_execution_access_is_zero": bool(
            prelock["exactPolicyValueOrActionEvaluationCount"] == 0
            and all(
                prelock[key] == 0
                for key in (
                    "externalLanguageRecordReadCount",
                    "rawModelResponseReadCount",
                    "protectedAccessCount",
                    "modelLoadCount",
                    "modelGenerationCount",
                    "APICallCount",
                    "trainingRunCount",
                    "actualExecutionCount",
                )
            )
        ),
        "access_is_one_exact_oracle_and_every_forbidden_effect_is_zero": bool(
            access["requiredOracleEvaluationCount"] == 1
            and all(value == 0 for key, value in access.items() if key != "requiredOracleEvaluationCount")
        ),
        "pass_authorizes_only_a_separate_population_design": bool(
            config["decisionRule"]["passAuthorizesFreshControlledLanguagePopulationDesignOnly"]
            and not config["decisionRule"]["passAuthorizesImmediateLanguagePopulationOpeningOrModelRun"]
            and not config["decisionRule"]["passAuthorizesAPITrainingRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(
                path.is_file()
                for path in (
                    config_path,
                    plan_path,
                    protocol_path,
                    tests_path,
                    runner_path,
                    verifier_path,
                    auditor_path,
                    parent_path,
                    reference_path,
                    PROJECT_ROOT / config["roadmap"],
                )
            )
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "209-controlled-language-observation-POMDP-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V209_exact_oracle_evaluation" if passed else "reject_V209_design",
        "checks": checks,
        "prelock_exposure": prelock,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V208_outcome": parent_path,
        "reference_V205_outcome": reference_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "209-controlled-language-observation-POMDP-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_hypotheses_grammar_channel_stages_rewards_comparators_gates_or_decision": False,
            "run_exact_single_model_free_oracle_evaluation": True,
            "read_external_language_load_or_run_model_or_access_protected_data": False,
            "API_training_registration_authority_action_or_execution": False,
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
