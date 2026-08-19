#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v188-binary-clarification-channel-frontier.json"
    plan_path = PROJECT_ROOT / "docs/v188-binary-clarification-channel-frontier-plan.md"
    protocol_path = PROJECT_ROOT / "python/v188_binary_clarification_channel_frontier.py"
    tests_path = PROJECT_ROOT / "python/test_v188_binary_clarification_channel_frontier.py"
    runner_path = PROJECT_ROOT / "python/run_v188_binary_clarification_channel_frontier.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v188_binary_clarification_channel_frontier_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v188_binary_clarification_channel_frontier.py"
    audit_path = PROJECT_ROOT / "outputs/v188-binary-clarification-channel-frontier/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v188-binary-clarification-channel-frontier-lock.json"
    output_root = PROJECT_ROOT / "outputs/v188-binary-clarification-channel-frontier/frontier"
    outcome_path = PROJECT_ROOT / "configs/v188-binary-clarification-channel-frontier-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V188 is already preregistered, evaluated, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV187r1OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    source_v187_lock_path = PROJECT_ROOT / parent["source_V187_lock"]
    source_v187_lock = json.loads(source_v187_lock_path.read_text())
    sources = {
        "question_codebook": PROJECT_ROOT / config["questionCodebook"],
        "contract_answer_vectors": PROJECT_ROOT / config["contractAnswerVectors"],
        "development_bindings": PROJECT_ROOT / config["developmentBindings"],
        "source_V187_result": PROJECT_ROOT / config["sourceV187Result"],
        "source_V187_problem_summary": PROJECT_ROOT / config["sourceV187ProblemSummary"],
    }
    population = config["frozenPopulation"]
    info = config["informationControls"]
    restricted = config["restrictedExactDepthTree"]
    frontier = config["costFrontier"]
    gates = config["frontierGates"]
    successor = config["successorRule"]
    checks = {
        "V187r1_is_valid_negative_and_authorizes_frontier_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and not parent["outcome"]["source_scientific_development_gates_passed"]
            and parent["authorization"]["preregister_text_free_channel_economics_frontier"]
            and not parent["authorization"]["preregister_correlated_error_or_model_successor"]
        ),
        "frozen_V186_V187_artifacts_are_exact": bool(
            file_sha256(sources["question_codebook"]) == source_v187_lock["question_codebook_sha256"]
            and file_sha256(sources["contract_answer_vectors"]) == source_v187_lock["contract_answer_vectors_sha256"]
            and file_sha256(sources["development_bindings"]) == source_v187_lock["development_bindings_sha256"]
            and file_sha256(sources["source_V187_result"]) == parent["source_V187_result_sha256"]
            and file_sha256(sources["source_V187_problem_summary"]) == parent["problem_summary_sha256"]
        ),
        "population_and_information_controls_are_frozen": bool(
            population["contractCount"] == 14
            and population["rawQuestionCount"] == 164
            and population["partitionDistinctQuestionCount"] == 25
            and population["priorIsUnchangedV187ObservedTargetFrequency"]
            and not population["readUtteranceOrDialogueLanguage"]
            and info["computeShannonEntropyBits"] and info["computeDeterministicHuffmanPrefixCode"]
        ),
        "restricted_tree_and_cost_grid_are_prospective": bool(
            not restricted["genericClarificationAvailable"]
            and not restricted["deferralAvailable"]
            and restricted["horizon"] == 13
            and restricted["requireSingletonLeaves"]
            and frontier["genericTrustedClarificationCost"] == 0.40
            and frontier["maximumTypedQuestionCount"] == 4
            and frontier["questionCostGridNumeratorsInclusive"] == [0, 80]
            and frontier["questionCostGridDenominator"] == 400
            and frontier["questionCostGridCount"] == 81
            and set(frontier["policies"]) == {"exact_adaptive", "best_fixed_open_loop"}
        ),
        "frontier_and_safety_gates_are_noncompensatory": bool(
            gates["requiredGridCellCount"] == 81
            and gates["requiredRestrictedExactLeafCount"] == 14
            and gates["requiredRestrictedTargetRetentionRate"] == 1.0
            and gates["requiredAllGridExactnessRate"] == 1.0
            and gates["maximumProtectedUtteranceLanguageReadCount"] == 0
        ),
        "successor_authority_and_prelock_exposure_are_closed": bool(
            successor["multiwayFeasibilityRequiresV187AtGenericBoundary"]
            and successor["multiwayFeasibilityRequiresSomeLowerBinaryCostWithPositiveValue"]
            and successor["multiwayFeasibilityRequiresPositiveTargetInformedOracleGap"]
            and successor["multiwayFeasibilityRequiresBinaryBreakEvenBelowV187QuestionCost"]
            and not successor["passAuthorizesImmediateMultiwayRun"]
            and not successor["passAuthorizesProtectedLanguageOrModelAccess"]
            and not successor["passAuthorizesRegistrationAuthorityActionOrExecution"]
            and all(value == 0 for value in config["preLockExposure"].values())
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, parent_path, source_v187_lock_path, plan_path, protocol_path,
                tests_path, runner_path, verifier_path, auditor_path, *sources.values(),
            )) and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "188-binary-clarification-channel-frontier-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V188_frontier_census" if passed else "reject_V188_design",
        "checks": checks,
        "prelock_exposure": config["preLockExposure"],
        "policy_score_count": 0,
        "protected_utterance_language_read_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_V187r1_outcome": parent_path,
        "source_V187_lock": source_v187_lock_path, **sources,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "188-binary-clarification-channel-frontier-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_population_questions_grid_gates_or_successor_rule": False,
            "run_frontier_census_once": True,
            "revise_V187_or_run_multiway_error_model_API_or_training": False,
            "read_protected_or_utterance_language": False,
            "register_mutate_call_service_act_or_execute": False,
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
