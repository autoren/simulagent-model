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
    config_path = PROJECT_ROOT / "configs/v204-open-world-semantic-pomdp-oracle.json"
    plan_path = PROJECT_ROOT / "docs/v204-open-world-semantic-pomdp-oracle-plan.md"
    protocol_path = PROJECT_ROOT / "python/v204_open_world_semantic_pomdp_oracle.py"
    tests_path = PROJECT_ROOT / "python/test_v204_open_world_semantic_pomdp_oracle.py"
    runner_path = PROJECT_ROOT / "python/run_v204_open_world_semantic_pomdp_oracle.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v204_open_world_semantic_pomdp_oracle_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v204_open_world_semantic_pomdp_oracle.py"
    audit_path = PROJECT_ROOT / "outputs/v204-open-world-semantic-pomdp-oracle/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v204-open-world-semantic-pomdp-oracle-lock.json"
    output_root = PROJECT_ROOT / "outputs/v204-open-world-semantic-pomdp-oracle/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v204-open-world-semantic-pomdp-oracle-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V204 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV203OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    channel = config["channel"]
    process = config["decisionProcess"]
    gates = config["oracleGates"]
    access = config["accessGates"]
    prelock = config["preLockExposure"]
    checks = {
        "V203_is_valid_negative_and_authorizes_only_richer_model_free_POMDP_preregistration": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["scientific_feasibility_passed"]
            and parent["authorization"]["preregister_separate_richer_model_free_POMDP_only"]
            and not parent["authorization"]["immediate_population_language_model_or_protected_access"]
        ),
        "three_hypotheses_include_explicit_outside_semantics_and_full_retention": bool(
            config["hypotheses"]["semanticCodebooks"] == ["CANONICAL", "REVERSED", "OUTSIDE_UNKNOWN"]
            and len(config["hypotheses"]["codebookPrior"]) == gates["requiredLatentCount"] == 3
            and abs(sum(config["hypotheses"]["codebookPrior"]) - 1.0) <= 1e-12
            and not config["hypotheses"]["priorComesFromLLMRanks"]
            and config["hypotheses"]["fullHypothesisUniverseAlwaysRetained"]
        ),
        "common_support_action_dependent_sensing_and_delayed_consequence_are_fixed": bool(
            channel["calibrationAndInspectionHaveCommonSupportAcrossEveryCodebook"]
            and min(channel["knownCodebookLogicalA"] + channel["knownCodebookLogicalB"] + channel["outsideUnknownForEitherCondition"]) > 0.0
            and process["settlementIsRequiredToRevealControlConsequence"]
            and process["repairImmediateReward"] == gates["requiredRepairImmediateReward"] == 0.0
            and "calibrate" in process["actionNames"]
            and "inspect" in process["actionNames"]
            and "defer" in process["actionNames"]
            and "settle" in process["actionNames"]
        ),
        "comparators_and_noncompensatory_scientific_gates_are_fixed": bool(
            config["fixedComparators"] == [
                "FULL_OPEN_WORLD_EXACT",
                "CLOSED_WORLD_BAYES_ADAPTIVE_EXCLUDING_OUTSIDE",
                "FULL_BELIEF_FORCED_COMMIT_WITHOUT_DEFER",
                "MAP_CERTAINTY_EQUIVALENCE",
                "PERSISTENT_POSTERIOR_SAMPLING",
                "BEST_OPEN_LOOP",
                "MYOPIC",
                "IMMEDIATE_DEFER",
            ]
            and gates["requiredSelectedExactRootAction"] == "calibrate"
            and gates["requiredExactActionAfterRootGreen"] == "defer"
            and gates["minimumDistinctReachableRepairActions"] == 2
            and gates["minimumReachableDeferHistoryCount"] >= 1
        ),
        "prelock_exact_language_model_API_training_and_execution_access_is_zero": bool(
            prelock["exactPolicyValueOrActionEvaluationCount"] == 0
            and all(
                prelock[key] == 0
                for key in (
                    "languageRecordReadCount",
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
        "pass_authorizes_only_fresh_source_design": bool(
            not config["decisionRule"]["passAuthorizesImmediateExternalCandidateEvaluationOrModelRun"]
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
                    PROJECT_ROOT / config["roadmap"],
                )
            )
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "204-open-world-semantic-POMDP-oracle-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V204_exact_oracle_evaluation" if passed else "reject_V204_design",
        "checks": checks,
        "prelock_exposure": prelock,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V203_outcome": parent_path,
        "roadmap": PROJECT_ROOT / config["roadmap"],
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "204-open-world-semantic-POMDP-oracle-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_hypotheses_channel_rewards_horizon_comparators_gates_or_decision": False,
            "run_exact_single_model_free_oracle_evaluation": True,
            "read_language_load_or_run_model_or_access_protected_data": False,
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
