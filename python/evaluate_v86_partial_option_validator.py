#!/usr/bin/env python3
"""Run the single frozen model-free V86 hardening census."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from schema_grounded_interface import (
    FINITE_GRAMMAR_STYLES,
    ClarificationRequest,
    compile_schema_registry,
    unsafe_schema_surface_mutations,
)
from schema_grounded_interface_v86 import (
    decorate_v79_policy_node_v86,
    hardened_certify_schema_surface,
    inspect_untrusted_hardened_surface,
    partial_option_injection_mutations,
    render_hardened_schema_clarification,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def mean(values: list[bool | float]) -> float:
    return float(sum(values) / len(values))


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v86-partial-option-validator-implementation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v86-partial-option-validator/evaluation/result.json"
    if result_path.exists():
        raise RuntimeError("V86 census has already been evaluated")
    implementation = json.loads(implementation_path.read_text())
    payload = {key: value for key, value in implementation.items() if key != "lock_payload_sha256"}
    if payload_hash(payload) != implementation["lock_payload_sha256"]:
        raise RuntimeError("V86 implementation lock mismatch")
    config = implementation["config_payload"]
    registry = compile_schema_registry(implementation["schemas"])
    surface_rows = []
    for schema in registry.schemas:
        requests = [ClarificationRequest(schema.schema_id, "slot", slot.slot_id) for slot in schema.slots]
        requests.append(ClarificationRequest(schema.schema_id, "all", None))
        for request in requests:
            modes: list[tuple[str, str | None]] = [("canonical", None)]
            modes.extend(("finite_grammar", style) for style in FINITE_GRAMMAR_STYLES)
            for source, style in modes:
                rendered = render_hardened_schema_clarification(registry, request, source=source, style=style)
                surface_rows.append({
                    "typed_request": request.to_dict(), "source": source, "style": style,
                    "question": rendered.question, "content_valid": rendered.certificate.content_valid,
                    "deployable": rendered.certificate.deployable,
                    "typed_request_preserved": rendered.typed_request == request,
                })
    base_rows = []
    for request, question in unsafe_schema_surface_mutations(registry):
        certificate = hardened_certify_schema_surface(registry, request, question, "canonical")
        base_rows.append({"typed_request": request.to_dict(), "question": question, "rejected": not certificate.deployable})
    partial_rows = []
    for request, question, injected in partial_option_injection_mutations(registry):
        certificate = hardened_certify_schema_surface(registry, request, question, "canonical")
        partial_rows.append({
            "typed_request": request.to_dict(), "question": question, "injected_option_surface": injected,
            "individual_unrequested_option_surface_count": certificate.individual_unrequested_option_surface_count,
            "rejected": not certificate.deployable,
        })
    v85_path = PROJECT_ROOT / implementation["V85_false_positive_artifact"]
    if file_sha256(v85_path) != implementation["V85_false_positive_artifact_sha256"]:
        raise RuntimeError("V85 regression artifact changed")
    v85 = json.loads(v85_path.read_text())
    request = ClarificationRequest(v85["typed_target"]["schema_id"], v85["typed_target"]["kind"], v85["typed_target"]["slot_id"])
    v85_certificate = hardened_certify_schema_surface(registry, request, v85["question"], "local_model_adversarial")
    v85_rows = [{
        "id": v85["id"], "question": v85["question"],
        "individual_unrequested_option_surface_count": v85_certificate.individual_unrequested_option_surface_count,
        "content_valid": v85_certificate.content_valid, "deployable": v85_certificate.deployable,
        "rejected": not v85_certificate.deployable and not v85_certificate.content_valid,
    }]
    untrusted_rows = []
    for schema in registry.schemas:
        requests = [ClarificationRequest(schema.schema_id, "slot", slot.slot_id) for slot in schema.slots]
        requests.append(ClarificationRequest(schema.schema_id, "all", None))
        for request in requests:
            rendered = render_hardened_schema_clarification(registry, request)
            certificate = inspect_untrusted_hardened_surface(registry, request, rendered.question)
            untrusted_rows.append({"typed_request": request.to_dict(), "content_valid": certificate.content_valid, "deployable": certificate.deployable})

    parent_path = PROJECT_ROOT / implementation["parent_V79_result"]
    if file_sha256(parent_path) != implementation["parent_V79_result_sha256"]:
        raise RuntimeError("V79 result changed")
    parent = json.loads(parent_path.read_text())
    bridge_rows = []; nonask_rows = []; fixture_values = {}; violations = 0
    for fixture_name, fixture in sorted(parent["fixtures"].items()):
        exact = fixture["exact"]; fixture_values[fixture_name] = exact["value"]
        violations += exact["complete_belief_certificate_violation_count"]
        for node_index, node in enumerate(exact["policy_nodes"]):
            if node["action"].startswith("ask_"):
                modes: list[tuple[str, str | None]] = [("canonical", None)]
                modes.extend(("finite_grammar", style) for style in FINITE_GRAMMAR_STYLES)
                for source, style in modes:
                    original = deepcopy(node)
                    decorated = decorate_v79_policy_node_v86(node, registry, source=source, style=style)
                    bridge_rows.append({
                        "fixture": fixture_name, "node_index": node_index, "action": node["action"],
                        "source": source, "style": style,
                        "action_preserved": decorated["action"] == node["action"],
                        "structurally_preserved": all(decorated[key] == value for key, value in original.items()),
                    })
                    if node != original: raise RuntimeError("V86 mutated V79 node")
            else:
                nonask_rows.append({"fixture": fixture_name, "node_index": node_index, "structurally_identical": decorate_v79_policy_node_v86(node, registry) == node})
    metrics = {
        "schema_rendered_case_count": len(surface_rows),
        "V79_bridge_node_count": len(bridge_rows) // 9,
        "V79_bridge_rendered_case_count": len(bridge_rows),
        "base_unsafe_mutation_count": len(base_rows),
        "partial_option_injection_mutation_count": len(partial_rows),
        "V85_false_positive_regression_count": len(v85_rows),
        "schema_surface_validity_rate": mean([row["content_valid"] and row["deployable"] for row in surface_rows]),
        "typed_request_preservation_rate": mean([row["typed_request_preserved"] for row in surface_rows]),
        "base_unsafe_mutation_rejection_rate": mean([row["rejected"] for row in base_rows]),
        "partial_option_injection_rejection_rate": mean([row["rejected"] for row in partial_rows]),
        "V85_false_positive_regression_rejection_rate": mean([row["rejected"] for row in v85_rows]),
        "disabled_untrusted_deployment_rate": mean([not row["deployable"] for row in untrusted_rows]),
        "V79_bridge_action_preservation_rate": mean([row["action_preserved"] for row in bridge_rows]),
        "V79_bridge_structural_preservation_rate": mean([row["structurally_preserved"] for row in bridge_rows] + [row["structurally_identical"] for row in nonask_rows]),
        "maximum_V79_policy_value_absolute_error": max(abs(value - parent["fixtures"][name]["exact"]["value"]) for name, value in fixture_values.items()),
        "complete_belief_execution_certificate_violation_count": violations,
    }
    e = config["enumeration"]; g = config["gates"]
    access = {"model_load_count":0,"model_generation_count":0,"API_call_count":0,"adapter_training_run_count":0,"human_record_access_count":0,"original_user_language_access_count":0,"real_tool_call_count":0,"external_side_effect_count":0}
    checks = {
        "complete_census": bool(metrics["schema_rendered_case_count"] == e["schemaRenderedCaseCount"] and metrics["V79_bridge_node_count"] == e["V79BridgeNodeCount"] and metrics["V79_bridge_rendered_case_count"] == e["V79BridgeRenderedCaseCount"] and metrics["base_unsafe_mutation_count"] == e["baseUnsafeMutationCount"] and metrics["partial_option_injection_mutation_count"] == e["partialOptionInjectionMutationCount"] and metrics["V85_false_positive_regression_count"] == e["V85RegisteredFalsePositiveRegressionCount"]),
        "safe_schema_surfaces_valid": metrics["schema_surface_validity_rate"] >= g["minimumSchemaSurfaceValidityRate"],
        "typed_requests_preserved": metrics["typed_request_preservation_rate"] >= g["minimumTypedRequestPreservationRate"],
        "base_unsafe_mutations_rejected": metrics["base_unsafe_mutation_rejection_rate"] >= g["minimumBaseUnsafeMutationRejectionRate"],
        "partial_option_injections_rejected": metrics["partial_option_injection_rejection_rate"] >= g["minimumPartialOptionInjectionRejectionRate"],
        "V85_false_positive_regression_rejected": metrics["V85_false_positive_regression_rejection_rate"] >= g["minimumV85FalsePositiveRegressionRejectionRate"],
        "untrusted_deployment_disabled": metrics["disabled_untrusted_deployment_rate"] >= g["minimumDisabledUntrustedDeploymentRate"],
        "V79_action_and_structure_preserved": metrics["V79_bridge_action_preservation_rate"] >= g["minimumV79BridgeActionPreservationRate"] and metrics["V79_bridge_structural_preservation_rate"] >= g["minimumV79BridgeStructuralPreservationRate"],
        "V79_values_and_execution_certificates_preserved": metrics["maximum_V79_policy_value_absolute_error"] <= g["maximumV79PolicyValueAbsoluteError"] and metrics["complete_belief_execution_certificate_violation_count"] <= g["maximumCompleteBeliefExecutionCertificateViolations"],
        "zero_model_API_training_human_language_tool_and_side_effect_access": all(value == 0 for value in access.values()),
    }
    passed = all(checks.values())
    result = {
        "schema_version":"86-partial-option-validator-result","experiment":"v86_model_free_unrequested_partial_option_hardening",
        "passed":passed,"decision":"freeze_hardened_partial_option_validator" if passed else "reject_V86_and_retain_V84_with_provenance_containment",
        "checks":checks,"metrics":metrics,"surface_rows":surface_rows,"base_unsafe_rows":base_rows,
        "partial_option_rows":partial_rows,"V85_regression_rows":v85_rows,"untrusted_rows":untrusted_rows,
        "V79_bridge_rows":bridge_rows,"V79_nonclarification_rows":nonask_rows,"fixture_values":fixture_values,"access":access,
        "claim_boundary":{"model_free_bounded_shadow_validator_correction":True,"model_or_human_language_evidence":False,"real_tool_or_open_world_safety_authority":False},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True); result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"passed":passed,"decision":result["decision"],"checks":checks,"metrics":metrics},indent=2,sort_keys=True))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
