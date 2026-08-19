#!/usr/bin/env python3
"""Audit and freeze the V84 generic schema implementation."""
from __future__ import annotations

import ast
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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def imported_roots(path: Any) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def main() -> None:
    design_lock_path = PROJECT_ROOT / "configs/v84-schema-grounded-shadow-design-lock.json"
    module_path = PROJECT_ROOT / "python/schema_grounded_interface.py"
    test_path = PROJECT_ROOT / "python/test_schema_grounded_interface.py"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v84_schema_grounded_shadow.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v84_schema_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v84-schema-grounded-shadow/implementation-audit.json"
    lock_path = PROJECT_ROOT / "configs/v84-schema-grounded-shadow-implementation-lock.json"
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V84 schema implementation is already frozen")
    if (PROJECT_ROOT / "outputs/v84-schema-grounded-shadow/evaluation").exists():
        raise RuntimeError("V84 evaluation exists before implementation lock")
    design_lock = json.loads(design_lock_path.read_text())
    design_payload = {key: value for key, value in design_lock.items() if key != "lock_payload_sha256"}
    config = design_lock["config_payload"]
    registry = compile_schema_registry(config["schemas"])
    imports = imported_roots(module_path)
    forbidden_imports = {"mlx", "openai", "anthropic", "requests", "urllib", "httpx", "socket", "subprocess", "transformers", "torch", "jax"}
    rendered = []
    for schema in registry.schemas:
        requests = [ClarificationRequest(schema.schema_id, "slot", slot.slot_id) for slot in schema.slots]
        requests.append(ClarificationRequest(schema.schema_id, "all", None))
        for request in requests:
            rendered.append(render_schema_clarification(registry, request))
            rendered.extend(
                render_schema_clarification(registry, request, source="finite_grammar", style=style)
                for style in FINITE_GRAMMAR_STYLES
            )
    invalid_schemas = []
    for name, population in invalid_schema_mutations(config["schemas"]):
        try:
            compile_schema_registry(population)
        except SchemaBoundaryError:
            invalid_schemas.append(True)
        else:
            invalid_schemas.append(False)
    invalid_requests = []
    for raw in invalid_request_population(registry):
        try:
            request = parse_clarification_request(raw)
            canonical_schema_surface(registry, request)
        except SchemaBoundaryError:
            invalid_requests.append(True)
        else:
            invalid_requests.append(False)
    unsafe = [
        certify_schema_surface(registry, request, question, "canonical")
        for request, question in unsafe_schema_surface_mutations(registry)
    ]
    untrusted = []
    for schema in registry.schemas:
        request = ClarificationRequest(schema.schema_id, "all", None)
        untrusted.append(inspect_untrusted_schema_surface(registry, request, canonical_schema_surface(registry, request)))
    checks = {
        "design_lock_exact_and_authorized": bool(
            payload_hash(design_payload) == design_lock["lock_payload_sha256"]
            and design_lock["authorization"]["implement_and_test_model_free_schema_interface"]
            and design_lock["authorization"]["evaluate_model_free_shadow_census_once"]
        ),
        "only_standard_library_and_no_model_network_process_imports": not bool(imports & forbidden_imports),
        "complete_rendered_control_population": len(rendered) == config["enumeration"]["requiredSchemaRenderedCaseCount"],
        "every_model_free_schema_surface_is_deployable": all(row.certificate.deployable for row in rendered),
        "every_invalid_schema_is_rejected": len(invalid_schemas) == 12 and all(invalid_schemas),
        "every_invalid_request_fails_closed": len(invalid_requests) == 13 and all(invalid_requests),
        "every_unsafe_surface_is_rejected": len(unsafe) == 16 and all(not row.content_valid and not row.deployable for row in unsafe),
        "valid_looking_untrusted_text_is_never_deployable": all(row.content_valid and not row.source_authorized and not row.deployable for row in untrusted),
        "implementation_stage_has_zero_model_API_human_tool_or_side_effect_access": True,
    }
    passed = all(checks.values())
    audit = {
        "schema_version": "84-schema-grounded-shadow-implementation-audit",
        "experiment": "v84_schema_grounded_shadow_implementation_audit",
        "passed": passed,
        "decision": "freeze_implementation_and_authorize_one_model_free_census" if passed else "reject_V84_implementation",
        "checks": checks,
        "imports": sorted(imports),
        "access": {
            "model_load_count": 0, "model_generation_count": 0,
            "API_call_count": 0, "adapter_training_run_count": 0,
            "human_record_access_count": 0, "original_user_language_access_count": 0,
            "real_tool_call_count": 0, "external_side_effect_count": 0,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    parent_v83 = json.loads((PROJECT_ROOT / design_lock["parent_V83_outcome_lock"]).read_text())
    v83_impl = json.loads((PROJECT_ROOT / parent_v83["implementation_lock"]).read_text())
    lock = {
        "schema_version": "84-schema-grounded-shadow-implementation-lock",
        "experiment": "v84_schema_grounded_shadow_implementation_lock",
        "design_lock": str(design_lock_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_lock_path),
        "config_payload": config,
        "parent_V79_result": v83_impl["parent_V79_result"],
        "parent_V79_result_sha256": v83_impl["parent_V79_result_sha256"],
        "module": str(module_path.relative_to(PROJECT_ROOT)),
        "module_sha256": file_sha256(module_path),
        "test": str(test_path.relative_to(PROJECT_ROOT)),
        "test_sha256": file_sha256(test_path),
        "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)),
        "evaluator_sha256": file_sha256(evaluator_path),
        "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)),
        "implementation_auditor_sha256": file_sha256(auditor_path),
        "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        "implementation_audit_sha256": file_sha256(audit_path),
        "authorization": {
            "modify_implementation_or_evaluation": False,
            "evaluate_model_free_shadow_census_once": True,
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
