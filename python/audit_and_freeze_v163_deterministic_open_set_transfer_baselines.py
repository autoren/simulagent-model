#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy
import sklearn

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def valid_lock(payload: dict[str, Any]) -> bool:
    return payload_hash(
        {key: value for key, value in payload.items() if key != "lock_payload_sha256"}
    ) == payload.get("lock_payload_sha256")


def main() -> None:
    config_path = (
        PROJECT_ROOT / "configs/v163-deterministic-open-set-transfer-baselines.json"
    )
    parent_path = (
        PROJECT_ROOT
        / "configs/v162-fresh-massive-transfer-language-extraction-outcome-lock.json"
    )
    historical_interface_path = (
        PROJECT_ROOT / "configs/v105-open-world-interface-outcome-lock.json"
    )
    plan_path = (
        PROJECT_ROOT / "docs/v163-deterministic-open-set-transfer-baselines-plan.md"
    )
    protocol_path = (
        PROJECT_ROOT / "python/v163_deterministic_open_set_transfer_baselines.py"
    )
    tests_path = (
        PROJECT_ROOT / "python/test_v163_deterministic_open_set_transfer_baselines.py"
    )
    runner_path = (
        PROJECT_ROOT / "python/run_v163_deterministic_open_set_transfer_baselines.py"
    )
    verifier_path = (
        PROJECT_ROOT
        / "python/verify_and_freeze_v163_deterministic_open_set_transfer_outcome.py"
    )
    auditor_path = (
        PROJECT_ROOT
        / "python/audit_and_freeze_v163_deterministic_open_set_transfer_baselines.py"
    )
    audit_path = (
        PROJECT_ROOT
        / "outputs/v163-deterministic-open-set-transfer/development-design-audit.json"
    )
    lock_path = (
        PROJECT_ROOT
        / "configs/v163-deterministic-open-set-transfer-baselines-lock.json"
    )
    output_root = (
        PROJECT_ROOT
        / "outputs/v163-deterministic-open-set-transfer/development-baselines"
    )
    if audit_path.exists() or lock_path.exists() or output_root.exists():
        raise RuntimeError("V163 baseline design is already frozen or materialized")

    config = json.loads(config_path.read_text())
    parent = json.loads(parent_path.read_text())
    historical = json.loads(historical_interface_path.read_text())
    catalog_path = PROJECT_ROOT / config["visibleCatalog"]
    hypotheses_path = PROJECT_ROOT / config["safeHypothesisUniverse"]
    catalog = json.loads(catalog_path.read_text())
    hypotheses = json.loads(hypotheses_path.read_text())
    split = config["developmentSplit"]
    baselines = config["deterministicBaselines"]
    pipeline_gates = config["baselinePipelineGates"]
    residual_gates = config["residualQualificationGates"]
    exposure = config["preLockExposure"]
    authority = config["authorityBoundary"]

    checks = {
        "V162_outcome_is_exact_and_authorizes_deterministic_design": bool(
            valid_lock(parent)
            and parent["outcome"]["passed"]
            and parent["outcome"]["scientific_extraction_passed"]
            and parent["authorization"][
                "preregister_deterministic_development_interface_controls_metrics_and_gates"
            ]
            and parent["authorization"][
                "automatically_read_development_language_after_interface_lock"
            ]
            and not parent["authorization"][
                "read_protected_transfer_during_development"
            ]
            and not parent["authorization"][
                "load_model_before_deterministic_baseline_outcome"
            ]
        ),
        "development_and_protected_identities_match_V162_without_protected_open": bool(
            config["developmentLanguage"] == parent["development_transfer_language"]
            and config["developmentLanguageSha256"]
            == parent["development_transfer_language_sha256"]
            and config["protectedLanguageIdentityOnly"]["path"]
            == parent["protected_transfer_language"]
            and config["protectedLanguageIdentityOnly"]["sha256"]
            == parent["protected_transfer_language_sha256"]
            and config["protectedLanguageIdentityOnly"]["fileOpenCount"] == 0
        ),
        "historical_training_only_interface_is_exact_and_complete": bool(
            valid_lock(historical)
            and historical["outcome"]["passed"]
            and file_sha256(catalog_path) == config["visibleCatalogSha256"]
            and file_sha256(hypotheses_path)
            == config["safeHypothesisUniverseSha256"]
            and config["visibleCatalog"] == historical["visible_catalog"]
            and config["safeHypothesisUniverse"]
            == historical["safe_hypothesis_universe"]
            and len(catalog["scenarios"]) == 3
            and len(catalog["intents"]) == 12
            and len(hypotheses["hypotheses"]) == 17
        ),
        "balanced_hash_only_development_split_and_controls_are_frozen": bool(
            split["calibrationCountPerClass"] == 24
            and split["evaluationCountPerClass"] == 24
            and len(split["classes"]) == 4
            and split["splitUsesOnlyFrozenRecordIdentifierAndClass"]
            and not split["manualLanguageInspection"]
            and config["controlledMissingObservation"]["count"] == 24
            and config["controlledMissingObservation"][
                "selectionUsesRecordIdentifierOnly"
            ]
            and config["controlledMissingObservation"]["utteranceExposureCount"]
            == 0
        ),
        "deterministic_controls_costs_metrics_and_consensus_are_frozen": bool(
            set(baselines)
            == {
                "complete_safe_enumeration",
                "ask_always",
                "identifier_grammar",
                "character_ngram_retrieval",
                "deterministic_consensus",
                "oracle",
            }
            and baselines["character_ngram_retrieval"]["knownThresholdGrid"]
            and baselines["character_ngram_retrieval"][
                "unsupportedThresholdGrid"
            ]
            and baselines["deterministic_consensus"]["otherwise"] == "ABSTAIN"
            and config["decisionCosts"]["known"]["wrong_known"] == 10.0
            and len(config["registeredMetrics"]) >= 12
        ),
        "pipeline_and_residual_gates_are_noncompensatory": bool(
            pipeline_gates["requiredCalibrationRecordCount"] == 96
            and pipeline_gates["requiredEvaluationRecordCount"] == 96
            and pipeline_gates["requiredControlledMissingObservationCount"] == 24
            and pipeline_gates["requiredOracleExactDecisionAccuracy"] == 1.0
            and pipeline_gates["requiredOracleMeanRegret"] == 0.0
            and pipeline_gates["maximumProtectedLanguageReadCount"] == 0
            and pipeline_gates["maximumModelLoadCount"] == 0
            and residual_gates["minimumResidualRecordCount"] == 24
            and residual_gates["maximumResidualRecordCount"] == 88
            and residual_gates["minimumNonResidualRecordCount"] == 8
            and residual_gates["minimumNonResidualExactDecisionAccuracy"] == 0.95
            and residual_gates["maximumNonResidualFalseKnownAcceptanceRate"]
            == 0.0
        ),
        "runtime_versions_match": bool(
            numpy.__version__ == config["deterministicRuntime"]["numpyVersion"]
            and sklearn.__version__
            == config["deterministicRuntime"]["scikitLearnVersion"]
        ),
        "authority_and_access_boundaries_remain_closed": bool(
            authority["allActionsAreCounterfactualShadowActions"]
            and authority["realExecutionCount"] == 0
            and authority["safeHypothesisUniverseMayNotBePruned"]
            and authority["authoritativeCapabilityStatePosteriorAndPolicyAreImmutable"]
            and authority["deterministicPredictionsDoNotAuthorizeAction"]
            and authority["futureModelMayNotSelectActionOrExecuteTool"]
            and all(value == 0 for value in exposure.values())
            and not config["decisionRule"]["baselinePassAuthorizesProtectedAccess"]
            and not config["decisionRule"]["baselinePassAuthorizesImmediateModelRun"]
            and not config["decisionRule"][
                "baselinePassAuthorizesAPITrainingOntologyPlanningActionOrExecution"
            ]
        ),
        "plan_and_locked_code_exist": all(
            path.is_file()
            for path in (
                plan_path,
                protocol_path,
                tests_path,
                runner_path,
                verifier_path,
                auditor_path,
            )
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "163-deterministic-open-set-transfer-design-audit",
        "experiment": "v163_deterministic_open_set_transfer_design_audit",
        "passed": passed,
        "decision": (
            "freeze_and_authorize_deterministic_development_baselines"
            if passed
            else "reject_V163_baseline_design"
        ),
        "checks": checks,
        "prelock_access": exposure,
        "protected_language_file_open_count": 0,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    dependencies = {
        "config": config_path,
        "parent_language_outcome": parent_path,
        "historical_interface_outcome": historical_interface_path,
        "visible_catalog": catalog_path,
        "safe_hypothesis_universe": hypotheses_path,
        "source_archive": PROJECT_ROOT / config["sourceArchive"],
        "plan": plan_path,
        "protocol": protocol_path,
        "tests": tests_path,
        "runner": runner_path,
        "verifier": verifier_path,
        "auditor": auditor_path,
        "design_audit": audit_path,
    }
    lock: dict[str, Any] = {
        "schema_version": "163-deterministic-open-set-transfer-baselines-lock",
        "experiment": "v163_deterministic_open_set_transfer_baselines_lock",
        "config_payload": config,
        "development_language": config["developmentLanguage"],
        "development_language_sha256": config["developmentLanguageSha256"],
        "source_archive": config["sourceArchive"],
        "source_archive_sha256": config["sourceArchiveSha256"],
        "protected_language_identity_only": config["protectedLanguageIdentityOnly"],
        "authorization": {
            "modify_split_baselines_costs_metrics_residual_rule_or_gates": False,
            "automatically_read_development_language_once": True,
            "automatically_read_declared_training_language_once": True,
            "read_protected_language": False,
            "manually_inspect_any_utterance": False,
            "run_deterministic_development_baselines_once": True,
            "load_or_run_local_or_API_model": False,
            "train_adapter_induce_ontology_plan_act_or_execute": False,
            "grant_capability_state_belief_action_or_execution_authority": False,
            "perform_real_service_call_or_external_side_effect": False,
        },
    }
    for key, path in dependencies.items():
        lock[key] = str(path.relative_to(PROJECT_ROOT))
        lock[f"{key}_sha256"] = file_sha256(path)
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "lock": str(lock_path.relative_to(PROJECT_ROOT)),
                "sha256": file_sha256(lock_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
