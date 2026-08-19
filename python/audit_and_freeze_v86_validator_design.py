#!/usr/bin/env python3
"""Audit and freeze the V86 partial-option validator correction."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v86-partial-option-validator-design.json"
    parent_path = PROJECT_ROOT / "configs/v85-local-adversarial-generator-outcome-lock.json"
    schema_path = PROJECT_ROOT / "configs/v84-schema-grounded-shadow-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v86-partial-option-validator-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v86_validator_design.py"
    audit_path = PROJECT_ROOT / "outputs/v86-partial-option-validator/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v86-partial-option-validator-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V86 design is already frozen")
    if (PROJECT_ROOT / "outputs/v86-partial-option-validator/evaluation").exists():
        raise RuntimeError("V86 evaluation exists before design lock")
    config = json.loads(design_path.read_text())
    parent = json.loads(parent_path.read_text())
    parent_payload = {key: value for key, value in parent.items() if key != "lock_payload_sha256"}
    schema = json.loads(schema_path.read_text())
    schema_payload = {key: value for key, value in schema.items() if key != "lock_payload_sha256"}
    diagnostic = parent["outcome"]["post_outcome_stricter_diagnostic"]
    e = config["enumeration"]
    checks = {
        "verified_negative_V85_parent_exact_and_authorizes_correction": bool(
            payload_hash(parent_payload) == parent["lock_payload_sha256"]
            and not parent["outcome"]["passed"]
            and diagnostic["registered_validator_false_positive_count"] == 1
            and diagnostic["provenance_prevented_false_positive_deployment"]
            and parent["authorization"]["preregister_model_free_partial_option_validator_correction"]
            and not parent["authorization"]["access_local_or_API_model_or_train_adapter"]
        ),
        "positive_V84_schema_source_exact": bool(
            payload_hash(schema_payload) == schema["lock_payload_sha256"]
            and schema["outcome"]["passed"]
        ),
        "complete_exact_enumeration": bool(
            e["schemaCount"] == 4 and e["schemaRenderedCaseCount"] == 108
            and e["V79BridgeNodeCount"] == 6 and e["V79BridgeRenderedCaseCount"] == 54
            and e["baseUnsafeMutationCount"] == 16
            and e["partialOptionInjectionMutationCount"] == 4 * 2 * 2
            and e["V85RegisteredFalsePositiveRegressionCount"] == 1
        ),
        "only_rule_changes_and_model_free_sources_remain_fixed": bool(
            config["registeredCorrection"]["newRule"].startswith("forbid every individual option surface")
            and not config["registeredCorrection"]["changesSchemaBeliefActionOrPolicy"]
            and not config["registeredCorrection"]["changesAuthorizedRendererSources"]
            and not config["registeredCorrection"]["changesCanonicalOrFiniteGrammarText"]
            and config["authorizedRendererSources"] == ["canonical", "finite_grammar"]
        ),
        "noncompensatory_identity_safety_and_zero_access_gates": bool(
            all(config["gates"][key] == 1.0 for key in (
                "minimumSchemaSurfaceValidityRate", "minimumTypedRequestPreservationRate",
                "minimumBaseUnsafeMutationRejectionRate", "minimumPartialOptionInjectionRejectionRate",
                "minimumV85FalsePositiveRegressionRejectionRate", "minimumDisabledUntrustedDeploymentRate",
                "minimumV79BridgeActionPreservationRate", "minimumV79BridgeStructuralPreservationRate"
            ))
            and config["gates"]["maximumV79PolicyValueAbsoluteError"] <= 1e-12
            and all(config["gates"][key] == 0 for key in (
                "maximumModelLoadCount", "maximumModelGenerationCount", "maximumAPICallCount",
                "maximumAdapterTrainingRunCount", "maximumHumanRecordAccessCount",
                "maximumOriginalUserLanguageAccessCount", "maximumRealToolCallCount",
                "maximumExternalSideEffectCount"
            ))
        ),
        "design_stage_has_no_model_human_tool_or_side_effect_authority": bool(
            config["stageAuthorization"]["auditAndFreezeDesign"]
            and not config["stageAuthorization"]["accessLocalOrAPIModel"]
            and not config["stageAuthorization"]["trainAdapter"]
            and not config["stageAuthorization"]["collectHumanOrOriginalUserLanguage"]
            and not config["stageAuthorization"]["performRealToolCall"]
            and not config["stageAuthorization"]["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "86-partial-option-validator-design-audit",
        "experiment": "v86_partial_option_validator_design_audit",
        "passed": passed,
        "decision": "freeze_design_and_authorize_model_free_validator_implementation" if passed else "reject_V86_design",
        "checks": checks,
        "access": {
            "model_load_count": 0, "model_generation_count": 0, "API_call_count": 0,
            "adapter_training_run_count": 0, "human_record_access_count": 0,
            "original_user_language_access_count": 0, "real_tool_call_count": 0,
            "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    lock = {
        "schema_version": "86-partial-option-validator-design-lock",
        "experiment": "v86_partial_option_validator_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)), "design_sha256": file_sha256(design_path),
        "config_payload": config,
        "parent_V85_outcome_lock": str(parent_path.relative_to(PROJECT_ROOT)), "parent_V85_outcome_lock_sha256": file_sha256(parent_path),
        "schema_V84_outcome_lock": str(schema_path.relative_to(PROJECT_ROOT)), "schema_V84_outcome_lock_sha256": file_sha256(schema_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)), "plan_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)), "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)), "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_design_rule_population_or_gates": False,
            "implement_and_test_model_free_validator": True,
            "evaluate_model_free_census_once": True,
            "access_local_or_API_model": False,
            "train_adapter": False,
            "collect_human_or_original_user_language": False,
            "perform_real_tool_call_or_external_side_effect": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
