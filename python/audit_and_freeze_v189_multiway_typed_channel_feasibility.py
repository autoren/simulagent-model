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
    config_path = PROJECT_ROOT / "configs/v189-multiway-typed-channel-feasibility.json"
    plan_path = PROJECT_ROOT / "docs/v189-multiway-typed-channel-feasibility-plan.md"
    protocol_path = PROJECT_ROOT / "python/v189_multiway_typed_channel_feasibility.py"
    tests_path = PROJECT_ROOT / "python/test_v189_multiway_typed_channel_feasibility.py"
    runner_path = PROJECT_ROOT / "python/run_v189_multiway_typed_channel_feasibility.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v189_multiway_typed_channel_feasibility_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v189_multiway_typed_channel_feasibility.py"
    audit_path = PROJECT_ROOT / "outputs/v189-multiway-typed-channel-feasibility/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v189-multiway-typed-channel-feasibility-lock.json"
    output_root = PROJECT_ROOT / "outputs/v189-multiway-typed-channel-feasibility/feasibility"
    outcome_path = PROJECT_ROOT / "configs/v189-multiway-typed-channel-feasibility-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V189 is already preregistered, evaluated, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV188OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    v188_lock = json.loads((PROJECT_ROOT / parent["frontier_lock"]).read_text())
    v187_lock = json.loads((PROJECT_ROOT / v188_lock["source_V187_lock"]).read_text())
    v186_outcome = json.loads((PROJECT_ROOT / v187_lock["parent_V186_outcome"]).read_text())
    v186_lock = json.loads((PROJECT_ROOT / v186_outcome["codebook_lock"]).read_text())
    sources = {
        "contract_catalog": PROJECT_ROOT / config["contractCatalog"],
        "development_bindings": PROJECT_ROOT / config["developmentBindings"],
        "source_V187_result": PROJECT_ROOT / config["sourceV187Result"],
    }
    q = config["multiwayQuestions"]
    pricing = config["pricing"]
    gates = config["feasibilityGates"]
    interpretation = config["interpretationRule"]
    decision = config["decisionRule"]
    checks = {
        "V188_is_valid_and_authorizes_multiway_design_only": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_frontier_gates_passed"]
            and parent["authorization"]["preregister_multiway_typed_channel_feasibility"]
            and not parent["authorization"]["run_multiway_without_separate_lock"]
        ),
        "frozen_contract_prior_and_V187_boundary_are_exact": bool(
            file_sha256(sources["contract_catalog"]) == v186_lock["contract_catalog_sha256"]
            and file_sha256(sources["development_bindings"]) == v188_lock["development_bindings_sha256"]
            and file_sha256(sources["source_V187_result"]) == v188_lock["source_V187_result_sha256"]
        ),
        "multiway_questions_are_finite_semantic_and_outcome_independent": bool(
            q["deriveOnlyFromFrozenAllowedSemanticPayload"]
            and q["excludeServiceVersionSourceTruthPresentedCandidateLanguageAndOutcomes"]
            and q["allQuestionSet"] == ["domain", "intent_concept", "transactionality"]
            and q["coarseQuestionSet"] == ["domain", "transactionality"]
        ),
        "pricing_rules_and_scenarios_are_prospective": bool(
            pricing["binaryQuestionAnchorCost"] == 0.10
            and pricing["genericTrustedClarificationCost"] == 0.40
            and pricing["maximumMultiwayTurns"] == 2
            and pricing["turnOverheadGridNumeratorsInclusive"] == [0, 9]
            and pricing["turnOverheadGridDenominator"] == 100
            and pricing["doNotEquateOneCategoricalAnswerWithOneBinaryBit"]
            and pricing["costIsComputedBeforePolicyScoring"]
        ),
        "controls_interpretation_and_safety_gates_are_fixed": bool(
            set(config["policies"]) == {
                "exact_adaptive_all_multiway_questions", "best_fixed_open_loop_all_multiway_questions",
                "exact_adaptive_coarse_questions_only", "always_generic_trusted_clarification",
            }
            and interpretation["robustMultiwayValueRequiresStrictImprovementUnderPureBitSlotUpperPricing"]
            and interpretation["conditionalMultiwayValueMeansImprovementOnlyWithPositiveTurnOverheadOrEntropyLowerBound"]
            and gates["requiredPricingScenarioCount"] == 11
            and gates["requiredObservedFinalExactnessRate"] == 1.0
            and gates["maximumProtectedUtteranceLanguageReadCount"] == 0
        ),
        "successor_authority_and_prelock_exposure_are_closed": bool(
            not decision["passAuthorizesImmediateLanguageModelOrProtectedRun"]
            and not decision["passAuthorizesRegistrationAuthorityActionOrExecution"]
            and all(value == 0 for value in config["preLockExposure"].values())
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, parent_path, plan_path, protocol_path, tests_path,
                runner_path, verifier_path, auditor_path, *sources.values(),
            )) and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "189-multiway-typed-channel-feasibility-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V189_feasibility_census" if passed else "reject_V189_design",
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
        "config": config_path, "parent_V188_outcome": parent_path,
        "source_V188_lock": PROJECT_ROOT / parent["frontier_lock"],
        "source_V187_lock": PROJECT_ROOT / v188_lock["source_V187_lock"],
        "source_V186_outcome": PROJECT_ROOT / v187_lock["parent_V186_outcome"],
        "source_V186_lock": PROJECT_ROOT / v186_outcome["codebook_lock"],
        **sources, "plan": plan_path, "protocol": protocol_path, "tests": tests_path,
        "runner": runner_path, "verifier": verifier_path, "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "189-multiway-typed-channel-feasibility-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_questions_pricing_controls_gates_or_interpretation": False,
            "run_feasibility_census_once": True,
            "read_protected_or_utterance_language_run_model_API_or_training": False,
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
