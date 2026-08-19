#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from audit_and_freeze_v165_factored_ontology_identifiability_population import (
    valid_lock,
)


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config_path = PROJECT_ROOT / "configs/v166-model-free-factored-ontology-baselines.json"
    parent_path = PROJECT_ROOT / "configs/v165r1-outcome-verifier-repair-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v166-model-free-factored-ontology-baselines-plan.md"
    protocol_path = PROJECT_ROOT / "python/v166_model_free_factored_ontology_baselines.py"
    tests_path = PROJECT_ROOT / "python/test_v166_model_free_factored_ontology_baselines.py"
    runner_path = PROJECT_ROOT / "python/run_v166_model_free_factored_ontology_baselines.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v166_model_free_factored_ontology_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v166_model_free_factored_ontology_baselines.py"
    audit_path = PROJECT_ROOT / "outputs/v166-model-free-factored-ontology/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v166-model-free-factored-ontology-baselines-lock.json"
    output_root = PROJECT_ROOT / "outputs/v166-model-free-factored-ontology/baselines"
    outcome_path = PROJECT_ROOT / "configs/v166-model-free-factored-ontology-baselines-outcome-lock.json"
    if audit_path.exists() or lock_path.exists() or output_root.exists() or outcome_path.exists():
        raise RuntimeError("V166 is already preregistered, run, or frozen")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    gates = config["baselineGates"]
    authority = config["authorityBoundary"]
    exposure = config["preLockExposure"]
    input_keys = ("frozen_ontology", "public_records", "hidden_records", "population_summary")
    input_paths = {key: PROJECT_ROOT / parent[key] for key in input_keys}
    checks = {
        "V165r1_is_exact_and_authorizes_only_preregistered_model_free_baselines": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_population_passed"]
            and parent["authorization"]["preregister_model_free_V166_deterministic_baselines"]
            and not parent["authorization"]["score_baselines_without_separate_lock"]
            and not parent["authorization"]["load_or_run_local_or_API_model"]
            and not parent["authorization"]["register_provisional_primitive"]
        ),
        "frozen_inputs_exist_and_match_parent_fingerprints_without_semantic_open": all(
            input_paths[key].is_file()
            and file_sha256(input_paths[key]) == parent[f"{key}_sha256"]
            for key in input_keys
        ),
        "six_deterministic_baselines_and_complete_version_space_contract_are_frozen": bool(
            set(config["baselines"]) == set(gates["requiredBaselineNames"])
            and len(config["baselines"]) == 6
            and config["predictionContract"]["candidateIdsAreTruthTableIdentifiers"]
            and config["predictionContract"]["ambiguousCandidatesMayNotBeRankedOrPruned"]
            and config["predictionContract"]["contradictoryEvidenceMayNotBeAutoRepaired"]
            and config["predictionContract"]["provisionalCandidateMayNotBeRegistered"]
        ),
        "exactness_residual_and_ambiguity_gates_are_noncompensatory": bool(
            gates["requiredRecordCount"] == 144
            and gates["requiredExactCombinedVersionSpaceAccuracy"] == 1.0
            and gates["requiredExactCombinedEvidenceStatusAccuracy"] == 1.0
            and gates["requiredExactCombinedTargetRetention"] == 1.0
            and gates["requiredExactCombinedFalseProvisionalCreationRate"] == 0.0
            and gates["requiredExactCombinedFalseResolutionRate"] == 0.0
            and gates["requiredModelEligibleResidualCount"] == 0
            and gates["requiredIntentionallyAmbiguousRecordCount"] == 48
            and gates["requiredCandidatesPerAmbiguousRecord"] == 64
        ),
        "authority_and_model_external_boundaries_are_closed": bool(
            authority["authoritativeOntologyAndStateImmutable"]
            and authority["completeConsistentVersionSpaceRetained"]
            and not authority["candidateRankingOnAmbiguousEvidenceAllowed"]
            and not authority["contradictionAutoRepairAllowed"]
            and not authority["provisionalRegistrationAllowed"]
            and not authority["modelUseAllowed"]
            and not authority["actionOrExecutionAllowed"]
            and authority["realExecutionCount"] == 0
            and all(value == 0 for value in exposure.values())
            and all(gates[key] == 0 for key in (
                "maximumEvaluationRecordCount", "maximumManualJudgmentCount",
                "maximumModelLoadCount", "maximumModelGenerationCount",
                "maximumAPICallCount", "maximumTrainingRunCount",
                "maximumOntologyRegistrationCount", "maximumRealServiceCallCount",
                "maximumExternalSideEffectCount", "maximumActualExecutionCount",
            ))
        ),
        "decision_does_not_authorize_unlocked_follow_on_work": bool(
            not config["decisionRule"]["passAuthorizesImmediateEvidencePlannerOrSandboxRun"]
            and not config["decisionRule"]["passAuthorizesModelEvaluationOrEvaluationPopulation"]
            and not config["decisionRule"]["passAuthorizesRegistrationAuthorityActionOrExecution"]
        ),
        "required_locked_files_exist": all(path.is_file() for path in (
            config_path, parent_path, plan_path, protocol_path, tests_path,
            runner_path, verifier_path, auditor_path,
        )),
        "baseline_output_absent_before_lock": not output_root.exists(),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "166-model-free-factored-ontology-baselines-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_model_free_baseline_run" if passed else "reject_V166_design",
        "checks": checks,
        "prelock_exposure": exposure,
        "semantic_input_read_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_V165r1_outcome": parent_path,
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
        "schema_version": "166-model-free-factored-ontology-baselines-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_baselines_metrics_gates_residual_rule_or_decision": False,
            "read_frozen_public_hidden_ontology_and_summary_once": True,
            "run_deterministic_baselines_once": True,
            "create_or_open_evaluation_population": False,
            "make_manual_judgments": False,
            "load_or_run_local_or_API_model": False,
            "train_or_fit_learned_component": False,
            "register_provisional_primitive": False,
            "rank_or_prune_ambiguous_version_space": False,
            "grant_candidate_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_external_side_effect_or_execution": False,
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
