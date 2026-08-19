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
    config_path = PROJECT_ROOT / "configs/v211-deterministic-residual-baselines.json"
    plan_path = PROJECT_ROOT / "docs/v211-deterministic-residual-baselines-plan.md"
    protocol_path = PROJECT_ROOT / "python/v211_deterministic_residual_baselines.py"
    tests_path = PROJECT_ROOT / "python/test_v211_deterministic_residual_baselines.py"
    firewall_path = PROJECT_ROOT / "python/v211_firewall_worker.py"
    prediction_path = PROJECT_ROOT / "python/v211_prediction_worker.py"
    runner_path = PROJECT_ROOT / "python/run_v211_deterministic_residual_baselines.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v211_deterministic_residual_baselines_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v211_deterministic_residual_baselines.py"
    audit_path = PROJECT_ROOT / "outputs/v211-deterministic-residual-baselines/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v211-deterministic-residual-baselines-lock.json"
    outcome_path = PROJECT_ROOT / "configs/v211-deterministic-residual-baselines-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, outcome_path)):
        raise RuntimeError("V211 already preregistered or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV210OutcomeLock"]
    reference_path = PROJECT_ROOT / config["referenceV209r1OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    reference = json.loads(reference_path.read_text())
    pop = config["population"]
    baselines = config["baselines"]
    gates = config["gates"]
    exposure = config["preLockExposure"]
    artifacts_absent = all(not (PROJECT_ROOT / value).exists() for value in config["artifacts"].values())
    checks = {
        "V210_is_valid_positive_and_only_deterministic_baseline_is_authorized": bool(
            valid_lock(parent) and parent["outcome"]["passed"] and parent["outcome"]["population_projection_gates_passed"]
            and parent["authorization"]["preregister_separate_deterministic_development_baseline_design_only"]
            and not parent["authorization"]["open_protected_or_run_model"]
        ),
        "V209r1_is_valid_positive_fixed_decision_reference": bool(
            valid_lock(reference) and reference["outcome"]["passed"] and reference["outcome"]["scientific_oracle_passed"]
            and not reference["authorization"]["open_language_population_or_run_model"]
        ),
        "identifier_only_group_disjoint_split_is_frozen": bool(
            config["schemaVersion"] == "211-deterministic-residual-baselines-design"
            and pop["inputResidualCount"] == gates["requiredInputResidualCount"] == 180
            and pop["calibrationGroupCount"] == pop["evaluationGroupCount"] == 45
            and pop["calibrationRecordCount"] == gates["requiredCalibrationRecordCount"] == 90
            and pop["evaluationRecordCount"] == gates["requiredEvaluationRecordCount"] == 90
            and all(len(pop[key]) == 64 for key in ("calibrationGroupIdHash", "evaluationGroupIdHash", "calibrationRecordIdHash", "evaluationRecordIdHash"))
            and pop["splitUsesIdentifiersOnly"] and pop["groupDisjoint"] and pop["protectedArtifactsRemainClosed"]
        ),
        "required_independent_baselines_and_conservative_consensus_are_fixed": bool(
            set(baselines) >= {"RAW_LEXICAL", "COMPOSITIONAL_RESPONSE_SPAN", "ABSTENTION_FIRST_CONSENSUS", "ABSTAIN_ALWAYS"}
            and baselines["predictionInputFields"] == ["record_id", "context_id", "utterance"]
            and "group_id" in baselines["forbiddenPredictionFields"]
            and "semantic_observation_id" in baselines["forbiddenPredictionFields"]
            and not baselines["evaluationTruthReadDuringFitOrPrediction"]
            and not baselines["protectedReadDuringFitPredictionOrScoring"]
        ),
        "separate_prediction_worker_has_no_evaluation_truth_or_group_input": bool(
            "evaluation-truth" not in prediction_path.read_text()
            and "group_id" not in prediction_path.read_text()
            and "--evaluation-surface" in prediction_path.read_text()
            and gates["requiredPredictionWorkerEvaluationTruthPathCount"] == 0
            and gates["requiredPredictionWorkerGroupIdReadCount"] == 0
        ),
        "prediction_freeze_decision_impact_and_branch_rules_are_noncompensatory": bool(
            gates["requiredPredictionFreezeBeforeTruthJoin"]
            and gates["minimumAcceptedAccuracyForAnyAcceptingBaseline"] == 1.0
            and gates["maximumFalseAcceptanceCountForAnyBaseline"] == 0
            and gates["maximumConsensusNormalizedDecisionRegret"] == 0.0
            and gates["minimumAbstainAlwaysNormalizedDecisionRegret"] > 0.0
            and not config["decisionImpact"]["modelScoresOrRanksUsedAsLikelihoods"]
            and not config["decisionRule"]["zeroResidualAuthorizesModel"]
            and not config["decisionRule"]["passAuthorizesProtectedOpeningOrImmediateModelRun"]
        ),
        "prelock_exposure_is_identifier_only": bool(
            exposure["identifierOnlySplitEnumerationCount"] == 180
            and exposure["splitHashComputationCount"] == 4
            and all(value == 0 for key, value in exposure.items() if key not in {"identifierOnlySplitEnumerationCount", "splitHashComputationCount"})
        ),
        "required_files_exist_parent_artifacts_match_and_outputs_absent": bool(
            all(path.is_file() for path in (config_path, plan_path, protocol_path, tests_path, firewall_path, prediction_path, runner_path, verifier_path, auditor_path, parent_path, reference_path))
            and file_sha256(PROJECT_ROOT / pop["inputDevelopmentSurface"]) == parent["development_surface_sha256"]
            and file_sha256(PROJECT_ROOT / pop["inputDevelopmentTruth"]) == parent["development_truth_sha256"]
            and file_sha256(PROJECT_ROOT / pop["inputDevelopmentProjection"]) == parent["development_projection_sha256"]
            and artifacts_absent
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "211-deterministic-residual-baselines-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V211_development_only_evaluation" if passed else "reject_V211_design",
        "checks": checks,
        "prelock_exposure": exposure,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    dependencies = {
        "config": config_path, "parent_V210_outcome": parent_path, "reference_V209r1_outcome": reference_path,
        "plan": plan_path, "protocol": protocol_path, "tests": tests_path, "firewall_worker": firewall_path,
        "prediction_worker": prediction_path, "runner": runner_path, "verifier": verifier_path,
        "auditor": auditor_path, "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "211-deterministic-residual-baselines-lock",
        "experiment": config["experiment"], "config_payload": config,
        "authorization": {
            "modify_split_baselines_metrics_gates_or_decision": False,
            "run_one_development_only_deterministic_evaluation": True,
            "read_protected_or_run_model": False,
            "API_training_registration_authority_action_or_execution": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT)); lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    write_json(lock_path, lock)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
