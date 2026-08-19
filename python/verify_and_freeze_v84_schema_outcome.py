#!/usr/bin/env python3
"""Independently reconstruct and freeze the V84 schema-grounded outcome."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


STYLES = (
    "concise", "neutral", "polite", "formal",
    "direct", "friendly", "explicit", "minimal",
)
WRAPPERS = {
    "concise": "{core}?",
    "neutral": "Please clarify: {core}?",
    "polite": "Could you please clarify: {core}?",
    "formal": "Please specify: {core}?",
    "direct": "Direct clarification: {core}?",
    "friendly": "Could you help me clarify: {core}?",
    "explicit": "For clarity: {core}?",
    "minimal": "{core}?",
}
ACTION_MAP = {
    "ask_operation": ("slot", "operation"),
    "ask_recipient": ("slot", "recipient"),
    "ask_full_details": ("all", None),
}


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def request(schema_id: str, kind: str, slot_id: str | None) -> dict[str, Any]:
    return {"schema_id": schema_id, "kind": kind, "slot_id": slot_id}


def choice(slot: dict[str, Any]) -> str:
    return f"{slot['options'][0]['surface']} or {slot['options'][1]['surface']}"


def canonical(schema: dict[str, Any], typed_request: dict[str, Any]) -> str:
    slots = (
        schema["slots"]
        if typed_request["kind"] == "all"
        else [
            slot for slot in schema["slots"]
            if slot["slotId"] == typed_request["slot_id"]
        ]
    )
    clauses = [f"{slot['questionPrefix']} {choice(slot)}" for slot in slots]
    if len(clauses) == 1:
        return clauses[0] + "?"
    continuation = clauses[1][0].lower() + clauses[1][1:]
    return f"{clauses[0]}, and {continuation}?"


def render(schema: dict[str, Any], typed_request: dict[str, Any], source: str, style: str | None) -> str:
    base = canonical(schema, typed_request)
    if source == "canonical":
        return base
    core = base[:-1]
    core = core[0].lower() + core[1:]
    return WRAPPERS[style].format(core=core)


def expected_schema_rows(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schema in schemas:
        requests = [
            request(schema["schemaId"], "slot", slot["slotId"])
            for slot in schema["slots"]
        ] + [request(schema["schemaId"], "all", None)]
        for typed_request in requests:
            modes: list[tuple[str, str | None]] = [("canonical", None)]
            modes.extend(("finite_grammar", style) for style in STYLES)
            for source, style in modes:
                rows.append({
                    "schema_id": schema["schemaId"],
                    "typed_request": typed_request,
                    "source": source,
                    "style": style,
                    "question": render(schema, typed_request, source, style),
                    "strict_surface_valid": True,
                    "source_authorized": True,
                    "deployable": True,
                    "typed_request_preserved": True,
                })
    return rows


def expected_invalid_requests(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [{
        "request": request("unknown_schema", "slot", schemas[0]["slots"][0]["slotId"]),
        "failed_closed": True,
        "error_code": "unknown_schema",
    }]
    rows.extend({
        "request": request(schema["schemaId"], "slot", "unknown_slot"),
        "failed_closed": True,
        "error_code": "unknown_slot",
    } for schema in schemas)
    rows.extend({
        "request": request(schema["schemaId"], "slot", None),
        "failed_closed": True,
        "error_code": "missing_slot_id",
    } for schema in schemas)
    rows.extend({
        "request": request(schema["schemaId"], "all", schema["slots"][0]["slotId"]),
        "failed_closed": True,
        "error_code": "unexpected_slot_id",
    } for schema in schemas)
    return rows


def expected_unsafe_rows(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schema in schemas:
        first, second = schema["slots"]
        first_request = request(schema["schemaId"], "slot", first["slotId"])
        second_request = request(schema["schemaId"], "slot", second["slotId"])
        questions = (
            (first_request, f"{first['questionPrefix']} {first['options'][0]['surface']} and {first['options'][1]['surface']}?"),
            (second_request, f"{second['questionPrefix']} {second['options'][0]['surface']}?"),
            (second_request, f"{second['questionPrefix']} {choice(second)}, and {choice(first)}?"),
            (first_request, f"I will {choice(first)}?"),
        )
        rows.extend({
            "typed_request": typed_request,
            "question": question,
            "content_valid": False,
            "rejected": True,
        } for typed_request, question in questions)
    return rows


def main() -> None:
    implementation_path = PROJECT_ROOT / "configs/v84-schema-grounded-shadow-implementation-lock.json"
    result_path = PROJECT_ROOT / "outputs/v84-schema-grounded-shadow/evaluation/result.json"
    verifier_path = PROJECT_ROOT / "python/verify_and_freeze_v84_schema_outcome.py"
    results_doc_path = PROJECT_ROOT / "docs/v84-schema-grounded-shadow-results.md"
    audit_path = PROJECT_ROOT / "outputs/v84-schema-grounded-shadow/outcome-audit.json"
    lock_path = PROJECT_ROOT / "configs/v84-schema-grounded-shadow-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V84 outcome is already frozen")
    implementation = json.loads(implementation_path.read_text())
    implementation_payload = {
        key: value for key, value in implementation.items() if key != "lock_payload_sha256"
    }
    config = implementation["config_payload"]
    schemas = config["schemas"]
    result = json.loads(result_path.read_text())
    parent_path = PROJECT_ROOT / implementation["parent_V79_result"]
    parent = json.loads(parent_path.read_text())

    schema_rows = expected_schema_rows(schemas)
    invalid_schema_codes = {
        "duplicate_schema_id": "duplicate_schema_id",
        "duplicate_slot_id": "duplicate_slot_id",
        "missing_slot_id": "invalid_slot_shape",
        "one_option": "invalid_option_count",
        "three_options": "invalid_option_count",
        "duplicate_option_id": "duplicate_option_id",
        "duplicate_option_surface": "duplicate_option_surface",
        "unsafe_surface_newline": "unsafe_schema_text",
        "unsafe_surface_underscore": "unsafe_schema_text",
        "unsafe_surface_question_mark": "unsafe_schema_text",
        "unsafe_surface_execution_claim": "unsafe_schema_text",
        "missing_question_prefix": "invalid_slot_shape",
    }
    invalid_schema_rows = [
        {"name": name, "rejected": True, "error_code": invalid_schema_codes[name]}
        for name in config["invalidSchemaMutationNames"]
    ]
    invalid_request_rows = expected_invalid_requests(schemas)
    unsafe_rows = expected_unsafe_rows(schemas)
    untrusted_rows = []
    for schema in schemas:
        requests = [
            request(schema["schemaId"], "slot", slot["slotId"])
            for slot in schema["slots"]
        ] + [request(schema["schemaId"], "all", None)]
        untrusted_rows.extend({
            "typed_request": typed_request,
            "content_valid": True,
            "source_authorized": False,
            "deployable": False,
        } for typed_request in requests)

    project_schema = next(schema for schema in schemas if schema["schemaId"] == "project_workflow")
    bridge_rows: list[dict[str, Any]] = []
    nonask_rows: list[dict[str, Any]] = []
    fixture_values: dict[str, float] = {}
    certificate_violations = 0
    ask_nodes = 0
    for fixture_name, fixture in sorted(parent["fixtures"].items()):
        exact = fixture["exact"]
        fixture_values[fixture_name] = exact["value"]
        certificate_violations += exact["complete_belief_certificate_violation_count"]
        for node_index, node in enumerate(exact["policy_nodes"]):
            if node["action"] in ACTION_MAP:
                ask_nodes += 1
                kind, slot_id = ACTION_MAP[node["action"]]
                typed_request = request("project_workflow", kind, slot_id)
                modes: list[tuple[str, str | None]] = [("canonical", None)]
                modes.extend(("finite_grammar", style) for style in STYLES)
                for source, style in modes:
                    bridge_rows.append({
                        "fixture": fixture_name,
                        "node_index": node_index,
                        "action": node["action"],
                        "source": source,
                        "style": style,
                        "typed_request": typed_request,
                        "question": render(project_schema, typed_request, source, style),
                        "action_preserved": True,
                        "policy_node_structurally_preserved": True,
                    })
            else:
                nonask_rows.append({
                    "fixture": fixture_name,
                    "node_index": node_index,
                    "action": node["action"],
                    "structurally_identical": True,
                })
    expected_metrics = {
        "valid_schema_count": 4,
        "schema_rendered_case_count": len(schema_rows),
        "V79_bridge_node_count": ask_nodes,
        "V79_bridge_rendered_case_count": len(bridge_rows),
        "invalid_schema_mutation_count": len(invalid_schema_rows),
        "invalid_request_count": len(invalid_request_rows),
        "unsafe_surface_mutation_count": len(unsafe_rows),
        "valid_schema_acceptance_rate": 1.0,
        "invalid_schema_rejection_rate": 1.0,
        "invalid_request_fail_closed_rate": 1.0,
        "strict_schema_surface_validity_rate": 1.0,
        "typed_request_preservation_rate": 1.0,
        "unsafe_surface_mutation_rejection_rate": 1.0,
        "disabled_untrusted_deployment_rate": 1.0,
        "fresh_schema_coverage_rate": 1.0,
        "V79_bridge_action_preservation_rate": 1.0,
        "V79_bridge_policy_node_structural_preservation_rate": 1.0,
        "maximum_V79_policy_value_absolute_error": 0.0,
        "complete_belief_execution_certificate_violation_count": certificate_violations,
    }
    checks = {
        "implementation_lock_and_frozen_sources_exact": bool(
            payload_hash(implementation_payload) == implementation["lock_payload_sha256"]
            and file_sha256(PROJECT_ROOT / implementation["module"])
            == implementation["module_sha256"]
            and file_sha256(PROJECT_ROOT / implementation["evaluator"])
            == implementation["evaluator_sha256"]
        ),
        "parent_V79_result_exact": file_sha256(parent_path)
        == implementation["parent_V79_result_sha256"],
        "schema_rows_independently_reconstructed": close(result["schema_rows"], schema_rows),
        "invalid_schema_rows_independently_reconstructed": close(result["invalid_schema_rows"], invalid_schema_rows),
        "invalid_request_rows_independently_reconstructed": close(result["invalid_request_rows"], invalid_request_rows),
        "unsafe_surface_rows_independently_reconstructed": close(result["unsafe_surface_rows"], unsafe_rows),
        "untrusted_rows_independently_reconstructed": close(result["untrusted_rows"], untrusted_rows),
        "V79_bridge_rows_independently_reconstructed": close(result["V79_bridge_rows"], bridge_rows),
        "V79_nonclarification_rows_independently_reconstructed": close(result["V79_nonclarification_rows"], nonask_rows),
        "metrics_and_values_independently_reconstructed": bool(
            close(result["metrics"], expected_metrics)
            and close(result["fixture_values"], fixture_values)
        ),
        "all_gates_pass_and_access_is_zero": bool(
            result["passed"]
            and all(result["checks"].values())
            and all(value == 0 for value in result["access"].values())
        ),
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "84-schema-grounded-shadow-outcome-audit",
        "experiment": "v84_schema_grounded_shadow_outcome_audit",
        "passed": passed,
        "decision": "freeze_positive_generic_schema_shadow_outcome" if passed else "reject_V84_outcome",
        "checks": checks,
        "independent_metrics": expected_metrics,
        "claim_boundary": result["claim_boundary"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    lock = {
        "schema_version": "84-schema-grounded-shadow-outcome-lock",
        "experiment": "v84_schema_grounded_shadow_outcome_lock",
        "implementation_lock": str(implementation_path.relative_to(PROJECT_ROOT)),
        "implementation_lock_sha256": file_sha256(implementation_path),
        "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
        "verifier": str(verifier_path.relative_to(PROJECT_ROOT)),
        "verifier_sha256": file_sha256(verifier_path),
        "audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "audit_sha256": file_sha256(audit_path),
        "results_document": str(results_doc_path.relative_to(PROJECT_ROOT)),
        "results_document_sha256": file_sha256(results_doc_path),
        "outcome": {"passed": True, "decision": result["decision"], "metrics": result["metrics"]},
        "authorization": {
            "modify_or_rerun_V84": False,
            "use_generic_typed_schema_interface_in_bounded_shadow_mode": True,
            "deploy_model_or_untrusted_generated_surface": False,
            "grant_schema_or_surface_belief_action_or_execution_authority": False,
            "perform_real_tool_call_or_external_side_effect": False,
            "claim_human_language_open_world_or_safety_evidence": False,
            "preregister_offline_non_deployable_local_adversarial_generator_test": True,
            "access_local_model_before_successor_lock": False,
            "access_API_model_or_train_adapter": False,
        },
    }
    lock["lock_payload_sha256"] = payload_hash(lock)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__":
    main()
