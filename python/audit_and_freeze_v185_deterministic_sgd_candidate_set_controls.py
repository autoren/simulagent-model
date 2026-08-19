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
    config_path = PROJECT_ROOT / "configs/v185-deterministic-sgd-candidate-set-controls.json"
    plan_path = PROJECT_ROOT / "docs/v185-deterministic-sgd-candidate-set-controls-plan.md"
    protocol_path = PROJECT_ROOT / "python/v185_deterministic_sgd_candidate_set_controls.py"
    tests_path = PROJECT_ROOT / "python/test_v185_deterministic_sgd_candidate_set_controls.py"
    runner_path = PROJECT_ROOT / "python/run_v185_deterministic_sgd_candidate_set_controls.py"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v185_deterministic_sgd_candidate_set_controls_outcome.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v185_deterministic_sgd_candidate_set_controls.py"
    audit_path = PROJECT_ROOT / "outputs/v185-deterministic-sgd-candidate-set-controls/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v185-deterministic-sgd-candidate-set-controls-lock.json"
    output_root = PROJECT_ROOT / "outputs/v185-deterministic-sgd-candidate-set-controls/evaluation"
    outcome_path = PROJECT_ROOT / "configs/v185-deterministic-sgd-candidate-set-controls-outcome-lock.json"
    if any(path.exists() for path in (audit_path, lock_path, output_root, outcome_path)):
        raise RuntimeError("V185 is already preregistered, evaluated, or frozen")
    config = json.loads(config_path.read_text())
    parent_path = PROJECT_ROOT / config["parentV184OutcomeLock"]
    parent = json.loads(parent_path.read_text())
    paths = {
        "development_language": PROJECT_ROOT / config["developmentLanguage"],
        "protected_language": PROJECT_ROOT / config["protectedLanguage"],
        "declared_catalog_language": PROJECT_ROOT / config["declaredCatalogLanguage"],
        "hidden_identifiability": PROJECT_ROOT / config["hiddenIdentifiability"],
    }
    split = config["developmentSplit"]
    views = config["deterministicViews"]
    calibration = config["calibrationRule"]
    policy = config["trustedClarificationPolicy"]
    gates = config["evaluationGates"]
    exposure = config["preLockExposure"]
    decision = config["decisionRule"]
    checks = {
        "V184_is_valid_and_authorizes_only_deterministic_development_protocol": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_extraction_gates_passed"]
            and parent["authorization"]["preregister_deterministic_interface_and_development_controls"]
            and not parent["authorization"]["score_development_language_without_separate_lock"]
            and not parent["authorization"]["read_protected_language_during_development"]
        ),
        "frozen_language_catalog_and_hidden_artifacts_match_parent": bool(
            file_sha256(paths["development_language"]) == parent["development_language_sha256"]
            and file_sha256(paths["protected_language"]) == parent["protected_language_sha256"]
            and file_sha256(paths["declared_catalog_language"]) == parent["declared_catalog_language_sha256"]
            and all(path.is_file() for path in paths.values())
        ),
        "development_split_is_balanced_prospective_and_score_independent": bool(
            split["calibrationFractionWithinEachTruthKind"] == 0.5
            and split["requiredCalibrationCounts"] == split["requiredEvaluationCounts"]
            and sum(split["requiredCalibrationCounts"].values()) == 66
            and split["splitUsesOnlyFrozenRecordIdentifierTruthKindAndSalt"]
            and split["splitUsesNoLanguagePredictionScoreOrPolicyOutcome"]
        ),
        "two_views_are_fixed_set_valued_shadow_controls": bool(
            views["characterNgram"]["ngramWidths"] == [3, 4, 5]
            and views["characterNgram"]["minimumTopScoreGrid"]
            and views["tokenSchemaOverlap"]["minimumTopScoreGrid"]
            and views["belowThresholdOrTie"].startswith("return a multi-choice")
            and views["allCandidateSetsAreShadowOnly"]
            and views["authoritativeHypothesisUniverseAlwaysComplete"]
        ),
        "calibration_rule_requires_precision_selectivity_and_fixed_fallback": bool(
            calibration["jointGridSearchOverTwoDeterministicViews"]
            and calibration["candidateSpecificRouteOnlyOnSameSingletonAgreement"]
            and calibration["otherwiseGenericClarification"]
            and calibration["requiredCalibrationCandidateSpecificPrecision"] == 0.95
            and calibration["maximumCalibrationFalseSpecificRateOnNonKnown"] == 0.05
            and calibration["minimumCalibrationKnownSpecificCoverage"] == 0.20
            and calibration["ifNoConfigurationQualifies"].startswith("use always-generic")
            and calibration["noPostEvaluationThresholdChange"]
        ),
        "trusted_answers_are_sole_terminal_authority_and_costs_are_fixed": bool(
            policy["candidateSpecificQuestionCost"] == 0.25
            and policy["genericClarificationCost"] == 0.40
            and policy["wrongSpecificThenGenericCost"] == 0.65
            and policy["trustedCandidateConfirmationSucceedsOnlyForExactKnownTruth"]
            and policy["wrongOrNonKnownCandidateConfirmationProducesNoWitnessAndFallsBackGeneric"]
            and policy["trustedGenericClarificationReturnsExactSafeHiddenContractStatus"]
            and policy["retrievalNeverDeterminesTerminalState"]
        ),
        "evaluation_and_residual_gates_are_noncompensatory": bool(
            gates["minimumCandidateSpecificPrecision"] == 0.95
            and gates["maximumFalseSpecificRateOnNonKnown"] == 0.05
            and gates["minimumKnownSpecificCoverage"] == 0.20
            and gates["minimumImprovementOverAlwaysGeneric"] == 0.02
            and gates["requiredFinalExactnessAfterTrustedAnswers"] == 1.0
            and gates["requiredResidualMembershipUsesPredictionsOnly"]
            and gates["requiredResidualTruthKindCoverage"] == ["KNOWN", "PROVISIONAL", "UNSUPPORTED"]
        ),
        "prelock_protected_model_and_effect_access_is_zero": bool(
            all(value == 0 for value in exposure.values())
            and not decision["passAuthorizesImmediateModelRun"]
            and not decision["passAuthorizesProtectedAccess"]
            and not decision["passAuthorizesRegistrationAuthorityActionOrExecution"]
        ),
        "required_files_exist_and_outputs_are_absent": bool(
            all(path.is_file() for path in (
                config_path, parent_path, plan_path, protocol_path, tests_path,
                runner_path, verifier_path, auditor_path, *paths.values(),
            ))
            and not output_root.exists()
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "185-deterministic-SGD-candidate-set-controls-design-audit",
        "experiment": config["experiment"],
        "passed": passed,
        "decision": "freeze_and_authorize_one_V185_development_evaluation" if passed else "reject_V185_design",
        "checks": checks,
        "prelock_exposure": exposure,
        "development_language_read_count": 0,
        "protected_language_read_count": 0,
        "model_load_count": 0,
        "actual_execution_count": 0,
    }
    write_json(audit_path, audit)
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    dependencies = {
        "config": config_path,
        "parent_V184_outcome": parent_path,
        **paths,
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "185-deterministic-SGD-candidate-set-controls-lock",
        "experiment": config["experiment"],
        "config_payload": config,
        "authorization": {
            "modify_split_features_grids_calibration_costs_gates_or_decision": False,
            "evaluate_development_once": True,
            "read_protected_language": False,
            "run_model_API_training": False,
            "accept_prune_register_mutate_call_service_act_or_execute": False,
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
