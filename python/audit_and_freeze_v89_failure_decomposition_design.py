#!/usr/bin/env python3
"""Audit and freeze the V89 model-free failure decomposition."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v89-model-free-failure-decomposition-design.json"
    parent_path = PROJECT_ROOT / "configs/v88r1-external-intent-candidate-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v89-model-free-failure-decomposition-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v89_failure_decomposition_design.py"
    audit_path = PROJECT_ROOT / "outputs/v89-model-free-failure-decomposition/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v89-model-free-failure-decomposition-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V89 design is already frozen")
    if (PROJECT_ROOT / "outputs/v89-model-free-failure-decomposition/evaluation").exists():
        raise RuntimeError("V89 evaluation exists before design lock")
    config = json.loads(design_path.read_text())
    parent = json.loads(parent_path.read_text())
    parent_payload = {key: value for key, value in parent.items() if key != "lock_payload_sha256"}
    stage = config["stageAuthorization"]
    gates = config["gates"]
    checks = {
        "verified_negative_V88r1_parent_exact_and_authorizes_only_model_free_decomposition": bool(
            payload_hash(parent_payload) == parent["lock_payload_sha256"]
            and not parent["outcome"]["passed"]
            and parent["authorization"]["preregister_model_free_failure_decomposition_only"]
            and not parent["authorization"]["run_API_fallback_or_capacity_comparator"]
            and not parent["authorization"]["train_adapter_or_learned_likelihood"]
        ),
        "complete_frozen_population_and_views_are_registered": bool(
            config["population"]["recordCount"] == 48
            and config["population"]["useEveryFrozenRawFixture"]
            and not config["population"]["recordReplacementAllowed"]
            and not config["population"]["sourceUtteranceOrPromptAccessAllowed"]
            and len(config["registeredViews"]) == 7
            and len(config["registeredDiagnostics"]) == 5
        ),
        "serialization_rule_is_an_optimistic_upper_bound_not_a_parser_tune": bool(
            config["perfectSerializationRule"].startswith("for every nonconforming row only")
            and "replace both predicted sets with gold" in config["perfectSerializationRule"]
        ),
        "decision_rule_cannot_authorize_model_API_training_deployment_or_execution": bool(
            not config["decisionRule"]["passAuthorizesModelOrAPIAccess"]
            and not config["decisionRule"]["passAuthorizesTrainingDeploymentOrExecution"]
        ),
        "exact_reconstruction_and_zero_access_gates": bool(
            gates["requiredRecordCount"] == gates["requiredRawFixtureCount"] == 48
            and gates["minimumStrictMetricReconstructionRate"] == 1.0
            and gates["minimumRawRowReconstructionRate"] == 1.0
            and all(gates[key] == 0 for key in (
                "maximumModelLoadCount", "maximumModelGenerationCount", "maximumLLMAPICallCount",
                "maximumAdapterTrainingRunCount", "maximumSourceLanguageAccessCount",
                "maximumManualUtteranceInspectionCount", "maximumRealServiceCallCount",
                "maximumExternalSideEffectCount"
            ))
        ),
        "design_stage_has_no_evaluation_language_model_training_or_execution_authority": bool(
            stage["auditAndFreezeDesign"]
            and not stage["evaluateFrozenIdentifierOnlyArtifacts"]
            and not stage["readSourceUtterancesOrPrompts"]
            and not stage["accessLocalOrAPIModel"]
            and not stage["trainAdapter"]
            and not stage["performRealServiceCall"]
            and not stage["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {"schema_version": "89-model-free-failure-decomposition-design-audit", "experiment": "v89_failure_decomposition_design_audit", "passed": passed, "decision": "freeze_design_and_authorize_one_identifier_only_decomposition" if passed else "reject_V89_design", "checks": checks, "access": {"source_language_access_count": 0, "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "manual_utterance_inspection_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0}}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    lock = {"schema_version": "89-model-free-failure-decomposition-design-lock", "experiment": "v89_failure_decomposition_design_lock", "design": str(design_path.relative_to(PROJECT_ROOT)), "design_sha256": file_sha256(design_path), "config_payload": config, "parent_V88r1_outcome_lock": str(parent_path.relative_to(PROJECT_ROOT)), "parent_V88r1_outcome_lock_sha256": file_sha256(parent_path), "plan": str(plan_path.relative_to(PROJECT_ROOT)), "plan_sha256": file_sha256(plan_path), "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)), "design_auditor_sha256": file_sha256(auditor_path), "design_audit": str(audit_path.relative_to(PROJECT_ROOT)), "design_audit_sha256": file_sha256(audit_path), "authorization": {"modify_population_views_or_decision_rule": False, "evaluate_identifier_only_artifacts_once": True, "read_source_language_or_prompts": False, "access_local_or_API_model": False, "train_adapter": False, "perform_real_service_call_or_external_side_effect": False}}
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
