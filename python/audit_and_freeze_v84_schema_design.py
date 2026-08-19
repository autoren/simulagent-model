#!/usr/bin/env python3
"""Audit and freeze the V84 schema-grounded shadow design."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v84-schema-grounded-shadow-design.json"
    parent_path = PROJECT_ROOT / "configs/v83-strict-clarification-interface-outcome-lock.json"
    plan_path = PROJECT_ROOT / "docs/v84-schema-grounded-shadow-plan.md"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v84_schema_design.py"
    audit_path = PROJECT_ROOT / "outputs/v84-schema-grounded-shadow/design-audit.json"
    lock_path = PROJECT_ROOT / "configs/v84-schema-grounded-shadow-design-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V84 schema design is already frozen")
    if (PROJECT_ROOT / "outputs/v84-schema-grounded-shadow/evaluation").exists():
        raise RuntimeError("V84 evaluation exists before design lock")
    design = json.loads(design_path.read_text())
    parent = json.loads(parent_path.read_text())
    parent_payload = {
        key: value for key, value in parent.items() if key != "lock_payload_sha256"
    }
    schemas = design["schemas"]
    schema_ids = [schema["schemaId"] for schema in schemas]
    slot_ids = [
        slot["slotId"] for schema in schemas for slot in schema["slots"]
    ]
    option_ids = [
        option["optionId"]
        for schema in schemas
        for slot in schema["slots"]
        for option in slot["options"]
    ]
    rendered_count = (
        len(schemas)
        * design["enumeration"]["clarificationTargetsPerSchema"]
        * (1 + len(design["finiteGrammarStyles"]))
    )
    forbidden_surface_fragments = (
        "i have", "i've", "i will", "i'll", "already", "completed",
        "executed", "scheduled", "sent", "booked", "done",
    )
    schema_text_fields = [
        value
        for schema in schemas
        for slot in schema["slots"]
        for value in (
            slot["questionPrefix"],
            *(option["surface"] for option in slot["options"]),
        )
    ]
    checks = {
        "positive_V83_parent_exact_and_authorizes_preregistration": bool(
            payload_hash(parent_payload) == parent["lock_payload_sha256"]
            and parent["outcome"]["passed"]
            and parent["authorization"]["preregister_fresh_schema_grounded_shadow_benchmark"]
            and not parent["authorization"]["deploy_local_API_adapter_or_untrusted_surface_renderer"]
        ),
        "complete_unique_schema_population": bool(
            len(schemas) == design["enumeration"]["schemaCount"]
            and len(set(schema_ids)) == len(schema_ids)
            and all(len(schema["slots"]) == 2 for schema in schemas)
            and all(
                len(slot["options"]) == 2
                for schema in schemas for slot in schema["slots"]
            )
            and len(slot_ids) == 8
            and len(option_ids) == 16
            and all(
                len({slot["slotId"] for slot in schema["slots"]}) == 2
                for schema in schemas
            )
            and all(
                len({option["optionId"] for option in slot["options"]}) == 2
                and len({option["surface"] for option in slot["options"]}) == 2
                for schema in schemas for slot in schema["slots"]
            )
        ),
        "schema_text_population_is_safe_and_bounded": bool(
            all(
                value.isascii()
                and 1 <= len(value) <= 64
                and "?" not in value
                and "\n" not in value
                and "\r" not in value
                and "_" not in value
                and not any(
                    fragment in value.lower()
                    for fragment in forbidden_surface_fragments
                )
                for value in schema_text_fields
            )
            and all(
                1 <= len(slot["questionPrefix"]) <= 48
                for schema in schemas for slot in schema["slots"]
            )
        ),
        "rendered_and_bridge_census_counts_are_exact": bool(
            rendered_count == design["enumeration"]["requiredSchemaRenderedCaseCount"]
            and design["enumeration"]["requiredV79BridgeNodeCount"] == 6
            and design["enumeration"]["requiredV79BridgeRenderedCaseCount"]
            == 6 * (1 + len(design["finiteGrammarStyles"]))
        ),
        "negative_control_counts_are_frozen": bool(
            len(design["invalidSchemaMutationNames"])
            == design["enumeration"]["requiredInvalidSchemaMutationCount"]
            and design["enumeration"]["requiredInvalidRequestCount"] == 13
            and design["enumeration"]["requiredUnsafeSurfaceMutationCount"] == 16
        ),
        "only_model_free_sources_are_authorized": bool(
            design["authorizedRendererSources"] == ["canonical", "finite_grammar"]
            and set(design["disabledRendererSources"])
            == {"local_model", "API_model", "adapter_model", "untrusted_passthrough"}
        ),
        "noncompensatory_safety_identity_and_zero_access_gates": bool(
            all(
                design["gates"][key] == 1.0
                for key in (
                    "minimumValidSchemaAcceptanceRate",
                    "minimumInvalidSchemaRejectionRate",
                    "minimumInvalidRequestFailClosedRate",
                    "minimumStrictSchemaSurfaceValidityRate",
                    "minimumTypedRequestPreservationRate",
                    "minimumUnsafeSurfaceMutationRejectionRate",
                    "minimumDisabledUntrustedDeploymentRate",
                    "minimumFreshSchemaCoverageRate",
                    "minimumV79BridgeActionPreservationRate",
                    "minimumV79BridgePolicyNodeStructuralPreservationRate"
                )
            )
            and design["gates"]["maximumV79PolicyValueAbsoluteError"] <= 1e-12
            and all(
                design["gates"][key] == 0
                for key in (
                    "maximumModelLoadCount", "maximumModelGenerationCount",
                    "maximumAPICallCount", "maximumAdapterTrainingRunCount",
                    "maximumHumanRecordAccessCount", "maximumOriginalUserLanguageAccessCount",
                    "maximumRealToolCallCount", "maximumExternalSideEffectCount"
                )
            )
        ),
        "design_stage_has_no_model_human_tool_or_side_effect_authority": bool(
            design["stageAuthorization"]["auditAndFreezeDesign"]
            and not design["stageAuthorization"]["accessLocalOrAPIModel"]
            and not design["stageAuthorization"]["trainAdapter"]
            and not design["stageAuthorization"]["collectHumanOrOriginalUserLanguage"]
            and not design["stageAuthorization"]["performRealToolCall"]
            and not design["stageAuthorization"]["performExternalSideEffect"]
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "84-schema-grounded-shadow-design-audit",
        "experiment": "v84_schema_grounded_shadow_design_audit",
        "passed": passed,
        "decision": (
            "freeze_design_and_authorize_model_free_schema_implementation"
            if passed else "reject_V84_design"
        ),
        "checks": checks,
        "access": {
            "model_load_count": 0, "model_generation_count": 0,
            "API_call_count": 0, "adapter_training_run_count": 0,
            "human_record_access_count": 0, "original_user_language_access_count": 0,
            "real_tool_call_count": 0, "external_side_effect_count": 0
        }
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "84-schema-grounded-shadow-design-lock",
        "experiment": "v84_schema_grounded_shadow_design_lock",
        "design": str(design_path.relative_to(PROJECT_ROOT)),
        "design_sha256": file_sha256(design_path),
        "config_payload": design,
        "parent_V83_outcome_lock": str(parent_path.relative_to(PROJECT_ROOT)),
        "parent_V83_outcome_lock_sha256": file_sha256(parent_path),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": file_sha256(plan_path),
        "design_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "design_auditor_sha256": file_sha256(auditor_path),
        "design_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "design_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_design_schema_population_controls_or_gates": False,
            "implement_and_test_model_free_schema_interface": True,
            "evaluate_model_free_shadow_census_once": True,
            "access_local_or_API_model": False,
            "train_adapter": False,
            "collect_human_or_original_user_language": False,
            "perform_real_tool_call_or_external_side_effect": False
        }
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
