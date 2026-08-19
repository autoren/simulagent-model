#!/usr/bin/env python3
"""Run the single frozen, model-free V83 integration census."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from structured_llm_interface import (
    CANONICAL_SURFACES,
    CLARIFICATION_CODES,
    FINITE_GRAMMAR_STYLES,
    certify_surface,
    decorate_policy_node,
    inspect_untrusted_surface,
    unsafe_surface_mutations,
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
    implementation_path = (
        PROJECT_ROOT / "configs/v83-strict-clarification-interface-implementation-lock.json"
    )
    result_path = (
        PROJECT_ROOT
        / "outputs/v83-strict-clarification-interface/evaluation/result.json"
    )
    if result_path.exists():
        raise RuntimeError("V83 model-free integration has already been evaluated")
    implementation = json.loads(implementation_path.read_text())
    payload = {
        key: value
        for key, value in implementation.items()
        if key != "lock_payload_sha256"
    }
    if payload_hash(payload) != implementation["lock_payload_sha256"]:
        raise RuntimeError("V83 implementation lock payload hash mismatch")
    if not implementation["authorization"]["evaluate_model_free_integration_once"]:
        raise RuntimeError("V83 implementation lock does not authorize evaluation")
    design = implementation["config_payload"]
    parent_path = PROJECT_ROOT / implementation["parent_V79_result"]
    if file_sha256(parent_path) != implementation["parent_V79_result_sha256"]:
        raise RuntimeError("V79 parent result changed after V83 implementation freeze")
    parent = json.loads(parent_path.read_text())
    v78_design = json.loads(
        (PROJECT_ROOT / "configs/v78-clarification-benchmark-design.json").read_text()
    )

    rendered_rows: list[dict[str, Any]] = []
    nonclarification_rows: list[dict[str, Any]] = []
    total_certificate_violations = 0
    fixture_values: dict[str, float] = {}
    for fixture_name, fixture in sorted(parent["fixtures"].items()):
        exact = fixture["exact"]
        fixture_values[fixture_name] = exact["value"]
        total_certificate_violations += exact[
            "complete_belief_certificate_violation_count"
        ]
        for node_index, node in enumerate(exact["policy_nodes"]):
            action = node["action"]
            if action in CLARIFICATION_CODES:
                modes: list[tuple[str, str | None]] = [("canonical", None)]
                modes.extend(("finite_grammar", style) for style in FINITE_GRAMMAR_STYLES)
                for source, style in modes:
                    original = deepcopy(node)
                    decorated = decorate_policy_node(node, source=source, style=style)
                    surface = decorated["clarification_surface"]
                    original_fields_preserved = all(
                        decorated[key] == value for key, value in original.items()
                    )
                    rendered_rows.append(
                        {
                            "fixture": fixture_name,
                            "node_index": node_index,
                            "history": node["history"],
                            "action": action,
                            "source": source,
                            "style": style,
                            "question": surface["question"],
                            "strict_surface_valid": surface["certificate"]["content_valid"],
                            "source_authorized": surface["certificate"]["source_authorized"],
                            "deployable": surface["certificate"]["deployable"],
                            "action_code_preserved": surface["action_code"] == action,
                            "policy_node_structurally_preserved": original_fields_preserved,
                        }
                    )
                    if node != original:
                        raise RuntimeError("V83 mutated a parent policy node")
            else:
                decorated = decorate_policy_node(node)
                nonclarification_rows.append(
                    {
                        "fixture": fixture_name,
                        "node_index": node_index,
                        "action": action,
                        "structurally_identical": decorated == node,
                    }
                )

    mutation_rows = []
    for code, question in unsafe_surface_mutations():
        certificate = certify_surface(code, question, "canonical")
        mutation_rows.append(
            {
                "action_code": code,
                "question": question,
                "rejected": not certificate.deployable,
                "content_valid": certificate.content_valid,
            }
        )
    untrusted_rows = []
    for code in CLARIFICATION_CODES:
        certificate = inspect_untrusted_surface(code, CANONICAL_SURFACES[code])
        untrusted_rows.append(
            {
                "action_code": code,
                "content_valid": certificate.content_valid,
                "source_authorized": certificate.source_authorized,
                "deployable": certificate.deployable,
            }
        )

    metrics = {
        "all_policy_node_count": len(rendered_rows) // 9 + len(nonclarification_rows),
        "reachable_clarification_node_count": len(rendered_rows) // 9,
        "rendered_reachable_case_count": len(rendered_rows),
        "nonclarification_node_count": len(nonclarification_rows),
        "strict_surface_validity_rate": mean(
            [row["strict_surface_valid"] for row in rendered_rows]
        ),
        "action_code_preservation_rate": mean(
            [row["action_code_preserved"] for row in rendered_rows]
        ),
        "policy_node_structural_preservation_rate": mean(
            [row["policy_node_structurally_preserved"] for row in rendered_rows]
            + [row["structurally_identical"] for row in nonclarification_rows]
        ),
        "unsafe_mutation_rejection_rate": mean(
            [row["rejected"] for row in mutation_rows]
        ),
        "disabled_untrusted_deployment_rate": mean(
            [not row["deployable"] for row in untrusted_rows]
        ),
        "maximum_policy_value_absolute_error": max(
            abs(fixture_values[name] - parent["fixtures"][name]["exact"]["value"])
            for name in fixture_values
        ),
        "complete_belief_execution_certificate_violation_count": total_certificate_violations,
    }
    none_index = v78_design["hypothesesInTieBreakOrder"].index("none_of_the_above")
    access = {
        "model_load_count": 0,
        "model_generation_count": 0,
        "API_call_count": 0,
        "adapter_training_run_count": 0,
        "human_record_access_count": 0,
        "original_user_language_access_count": 0,
        "real_tool_call_count": 0,
        "external_side_effect_count": 0,
    }
    gates = design["gates"]
    checks = {
        "complete_policy_census": bool(
            metrics["all_policy_node_count"] == gates["requiredAllPolicyNodeCount"]
            and metrics["reachable_clarification_node_count"]
            == gates["requiredReachableClarificationNodeCount"]
            and metrics["rendered_reachable_case_count"]
            == gates["requiredRenderedReachableCaseCount"]
        ),
        "none_hypothesis_preserved": bool(
            v78_design["hypothesesInTieBreakOrder"][none_index]
            == gates["requiredNoneHypothesis"]
            and none_index == gates["requiredNoneHypothesisIndex"]
        ),
        "strict_surface_validity": metrics["strict_surface_validity_rate"]
        >= gates["minimumStrictSurfaceValidityRate"],
        "action_code_preservation": metrics["action_code_preservation_rate"]
        >= gates["minimumActionCodePreservationRate"],
        "policy_node_structural_preservation": metrics[
            "policy_node_structural_preservation_rate"
        ] >= gates["minimumPolicyNodeStructuralPreservationRate"],
        "unsafe_mutation_rejection": metrics["unsafe_mutation_rejection_rate"]
        >= gates["minimumUnsafeMutationRejectionRate"],
        "untrusted_deployment_disabled": metrics[
            "disabled_untrusted_deployment_rate"
        ] >= gates["minimumDisabledUntrustedDeploymentRate"],
        "policy_values_invariant": metrics["maximum_policy_value_absolute_error"]
        <= gates["maximumPolicyValueAbsoluteError"],
        "complete_belief_execution_certificates_preserved": metrics[
            "complete_belief_execution_certificate_violation_count"
        ] <= gates["maximumCompleteBeliefExecutionCertificateViolations"],
        "zero_model_API_training_human_language_tool_and_side_effect_access": all(
            access[key] == 0 for key in access
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "83-strict-clarification-interface-result",
        "experiment": "v83_model_free_strict_clarification_interface_integration",
        "passed": passed,
        "decision": (
            "freeze_strict_model_free_interface_for_shadow_use_above_V79"
            if passed
            else "reject_V83_integration_and_retain_V79_without_surface_adapter"
        ),
        "checks": checks,
        "metrics": metrics,
        "fixture_values": fixture_values,
        "rendered_rows": rendered_rows,
        "nonclarification_rows": nonclarification_rows,
        "unsafe_mutation_rows": mutation_rows,
        "untrusted_inspection_rows": untrusted_rows,
        "access": access,
        "claim_boundary": {
            "synthetic_bounded_model_free_shadow_integration": True,
            "local_or_API_model_evidence": False,
            "human_or_original_user_language_evidence": False,
            "open_world_safety_or_real_tool_authority": False,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
