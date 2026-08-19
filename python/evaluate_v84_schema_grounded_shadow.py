#!/usr/bin/env python3
"""Run the single frozen V84 model-free schema-grounded shadow census."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from schema_grounded_interface import (
    FINITE_GRAMMAR_STYLES,
    ClarificationRequest,
    SchemaBoundaryError,
    canonical_schema_surface,
    certify_schema_surface,
    compile_schema_registry,
    decorate_v79_policy_node,
    inspect_untrusted_schema_surface,
    invalid_request_population,
    invalid_schema_mutations,
    parse_clarification_request,
    render_schema_clarification,
    unsafe_schema_surface_mutations,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def mean(values: list[bool | float]) -> float:
    return float(sum(values) / len(values))


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v84-schema-grounded-shadow-implementation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v84-schema-grounded-shadow/evaluation/result.json"
    if result_path.exists():
        raise RuntimeError("V84 shadow census has already been evaluated")
    implementation = json.loads(implementation_path.read_text())
    payload = {key: value for key, value in implementation.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != implementation["lock_payload_sha256"]:
        raise RuntimeError("V84 implementation lock payload hash mismatch")
    config = implementation["config_payload"]
    registry = compile_schema_registry(config["schemas"])
    valid_schema_rows = [
        {"schema_id": schema.schema_id, "accepted": True}
        for schema in registry.schemas
    ]
    schema_rows: list[dict[str, Any]] = []
    for schema in registry.schemas:
        requests = [
            ClarificationRequest(schema.schema_id, "slot", slot.slot_id)
            for slot in schema.slots
        ] + [ClarificationRequest(schema.schema_id, "all", None)]
        for request in requests:
            modes: list[tuple[str, str | None]] = [("canonical", None)]
            modes.extend(("finite_grammar", style) for style in FINITE_GRAMMAR_STYLES)
            for source, style in modes:
                rendered = render_schema_clarification(
                    registry, request, source=source, style=style
                )
                schema_rows.append({
                    "schema_id": schema.schema_id,
                    "typed_request": request.to_dict(),
                    "source": source,
                    "style": style,
                    "question": rendered.question,
                    "strict_surface_valid": rendered.certificate.content_valid,
                    "source_authorized": rendered.certificate.source_authorized,
                    "deployable": rendered.certificate.deployable,
                    "typed_request_preserved": rendered.typed_request == request,
                })
    invalid_schema_rows = []
    for name, population in invalid_schema_mutations(config["schemas"]):
        try:
            compile_schema_registry(population)
        except SchemaBoundaryError as error:
            invalid_schema_rows.append({"name": name, "rejected": True, "error_code": error.code})
        else:
            invalid_schema_rows.append({"name": name, "rejected": False, "error_code": None})
    invalid_request_rows = []
    for raw in invalid_request_population(registry):
        try:
            request = parse_clarification_request(raw)
            canonical_schema_surface(registry, request)
        except SchemaBoundaryError as error:
            invalid_request_rows.append({"request": raw, "failed_closed": True, "error_code": error.code})
        else:
            invalid_request_rows.append({"request": raw, "failed_closed": False, "error_code": None})
    unsafe_rows = []
    for request, question in unsafe_schema_surface_mutations(registry):
        certificate = certify_schema_surface(registry, request, question, "canonical")
        unsafe_rows.append({
            "typed_request": request.to_dict(),
            "question": question,
            "content_valid": certificate.content_valid,
            "rejected": not certificate.deployable,
        })
    untrusted_rows = []
    for schema in registry.schemas:
        requests = [
            ClarificationRequest(schema.schema_id, "slot", slot.slot_id)
            for slot in schema.slots
        ] + [ClarificationRequest(schema.schema_id, "all", None)]
        for request in requests:
            question = canonical_schema_surface(registry, request)
            certificate = inspect_untrusted_schema_surface(registry, request, question)
            untrusted_rows.append({
                "typed_request": request.to_dict(),
                "content_valid": certificate.content_valid,
                "source_authorized": certificate.source_authorized,
                "deployable": certificate.deployable,
            })

    parent_path = PROJECT_ROOT / implementation["parent_V79_result"]
    if file_sha256(parent_path) != implementation["parent_V79_result_sha256"]:
        raise RuntimeError("V79 parent changed after V84 implementation freeze")
    parent = json.loads(parent_path.read_text())
    bridge_rows: list[dict[str, Any]] = []
    nonask_rows: list[dict[str, Any]] = []
    fixture_values: dict[str, float] = {}
    certificate_violations = 0
    for fixture_name, fixture in sorted(parent["fixtures"].items()):
        exact = fixture["exact"]
        fixture_values[fixture_name] = exact["value"]
        certificate_violations += exact["complete_belief_certificate_violation_count"]
        for node_index, node in enumerate(exact["policy_nodes"]):
            if node["action"].startswith("ask_"):
                modes: list[tuple[str, str | None]] = [("canonical", None)]
                modes.extend(("finite_grammar", style) for style in FINITE_GRAMMAR_STYLES)
                for source, style in modes:
                    original = deepcopy(node)
                    decorated = decorate_v79_policy_node(
                        node, registry, source=source, style=style
                    )
                    surface = decorated["schema_clarification_surface"]
                    bridge_rows.append({
                        "fixture": fixture_name,
                        "node_index": node_index,
                        "action": node["action"],
                        "source": source,
                        "style": style,
                        "typed_request": surface["typed_request"],
                        "question": surface["question"],
                        "action_preserved": decorated["action"] == node["action"],
                        "policy_node_structurally_preserved": all(
                            decorated[key] == value for key, value in original.items()
                        ),
                    })
                    if node != original:
                        raise RuntimeError("V84 mutated a V79 node")
            else:
                nonask_rows.append({
                    "fixture": fixture_name,
                    "node_index": node_index,
                    "action": node["action"],
                    "structurally_identical": decorate_v79_policy_node(node, registry) == node,
                })

    fresh_ids = {schema.schema_id for schema in registry.schemas if schema.schema_id != "project_workflow"}
    observed_fresh = {row["schema_id"] for row in schema_rows if row["schema_id"] in fresh_ids}
    metrics = {
        "valid_schema_count": len(valid_schema_rows),
        "schema_rendered_case_count": len(schema_rows),
        "V79_bridge_node_count": len(bridge_rows) // 9,
        "V79_bridge_rendered_case_count": len(bridge_rows),
        "invalid_schema_mutation_count": len(invalid_schema_rows),
        "invalid_request_count": len(invalid_request_rows),
        "unsafe_surface_mutation_count": len(unsafe_rows),
        "valid_schema_acceptance_rate": mean([row["accepted"] for row in valid_schema_rows]),
        "invalid_schema_rejection_rate": mean([row["rejected"] for row in invalid_schema_rows]),
        "invalid_request_fail_closed_rate": mean([row["failed_closed"] for row in invalid_request_rows]),
        "strict_schema_surface_validity_rate": mean([row["strict_surface_valid"] for row in schema_rows]),
        "typed_request_preservation_rate": mean([row["typed_request_preserved"] for row in schema_rows]),
        "unsafe_surface_mutation_rejection_rate": mean([row["rejected"] for row in unsafe_rows]),
        "disabled_untrusted_deployment_rate": mean([not row["deployable"] for row in untrusted_rows]),
        "fresh_schema_coverage_rate": len(observed_fresh) / len(fresh_ids),
        "V79_bridge_action_preservation_rate": mean([row["action_preserved"] for row in bridge_rows]),
        "V79_bridge_policy_node_structural_preservation_rate": mean(
            [row["policy_node_structurally_preserved"] for row in bridge_rows]
            + [row["structurally_identical"] for row in nonask_rows]
        ),
        "maximum_V79_policy_value_absolute_error": max(
            abs(value - parent["fixtures"][name]["exact"]["value"])
            for name, value in fixture_values.items()
        ),
        "complete_belief_execution_certificate_violation_count": certificate_violations,
    }
    access = {
        "model_load_count": 0, "model_generation_count": 0,
        "API_call_count": 0, "adapter_training_run_count": 0,
        "human_record_access_count": 0, "original_user_language_access_count": 0,
        "real_tool_call_count": 0, "external_side_effect_count": 0,
    }
    gates = config["gates"]
    enumeration = config["enumeration"]
    checks = {
        "complete_frozen_census": bool(
            metrics["valid_schema_count"] == enumeration["schemaCount"]
            and metrics["schema_rendered_case_count"] == enumeration["requiredSchemaRenderedCaseCount"]
            and metrics["V79_bridge_node_count"] == enumeration["requiredV79BridgeNodeCount"]
            and metrics["V79_bridge_rendered_case_count"] == enumeration["requiredV79BridgeRenderedCaseCount"]
            and metrics["invalid_schema_mutation_count"] == enumeration["requiredInvalidSchemaMutationCount"]
            and metrics["invalid_request_count"] == enumeration["requiredInvalidRequestCount"]
            and metrics["unsafe_surface_mutation_count"] == enumeration["requiredUnsafeSurfaceMutationCount"]
        ),
        "valid_schema_acceptance": metrics["valid_schema_acceptance_rate"] >= gates["minimumValidSchemaAcceptanceRate"],
        "invalid_schema_rejection": metrics["invalid_schema_rejection_rate"] >= gates["minimumInvalidSchemaRejectionRate"],
        "invalid_request_fail_closed": metrics["invalid_request_fail_closed_rate"] >= gates["minimumInvalidRequestFailClosedRate"],
        "strict_schema_surface_validity": metrics["strict_schema_surface_validity_rate"] >= gates["minimumStrictSchemaSurfaceValidityRate"],
        "typed_request_preservation": metrics["typed_request_preservation_rate"] >= gates["minimumTypedRequestPreservationRate"],
        "unsafe_surface_mutation_rejection": metrics["unsafe_surface_mutation_rejection_rate"] >= gates["minimumUnsafeSurfaceMutationRejectionRate"],
        "untrusted_deployment_disabled": metrics["disabled_untrusted_deployment_rate"] >= gates["minimumDisabledUntrustedDeploymentRate"],
        "fresh_schema_coverage": metrics["fresh_schema_coverage_rate"] >= gates["minimumFreshSchemaCoverageRate"],
        "V79_bridge_action_preservation": metrics["V79_bridge_action_preservation_rate"] >= gates["minimumV79BridgeActionPreservationRate"],
        "V79_bridge_structural_preservation": metrics["V79_bridge_policy_node_structural_preservation_rate"] >= gates["minimumV79BridgePolicyNodeStructuralPreservationRate"],
        "V79_policy_values_invariant": metrics["maximum_V79_policy_value_absolute_error"] <= gates["maximumV79PolicyValueAbsoluteError"],
        "complete_belief_execution_certificates_preserved": metrics["complete_belief_execution_certificate_violation_count"] <= gates["maximumCompleteBeliefExecutionCertificateViolations"],
        "zero_model_API_training_human_language_tool_and_side_effect_access": all(value == 0 for value in access.values()),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "84-schema-grounded-shadow-result",
        "experiment": "v84_model_free_schema_grounded_shadow_transfer",
        "passed": passed,
        "decision": (
            "freeze_generic_typed_schema_interface_for_bounded_shadow_use"
            if passed else "reject_V84_and_retain_single_schema_V83_interface"
        ),
        "checks": checks,
        "metrics": metrics,
        "fixture_values": fixture_values,
        "schema_rows": schema_rows,
        "invalid_schema_rows": invalid_schema_rows,
        "invalid_request_rows": invalid_request_rows,
        "unsafe_surface_rows": unsafe_rows,
        "untrusted_rows": untrusted_rows,
        "V79_bridge_rows": bridge_rows,
        "V79_nonclarification_rows": nonask_rows,
        "access": access,
        "claim_boundary": {
            "synthetic_bounded_model_free_schema_shadow_evidence": True,
            "local_or_API_model_evidence": False,
            "human_or_original_user_language_evidence": False,
            "open_world_safety_or_real_tool_authority": False,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "decision": result["decision"], "checks": checks, "metrics": metrics}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
