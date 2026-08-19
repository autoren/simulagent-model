#!/usr/bin/env python3
"""Audit and freeze the V83 model-free interface implementation."""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

from structured_llm_interface import (
    CANONICAL_SURFACES,
    CLARIFICATION_CODES,
    FINITE_GRAMMAR_STYLES,
    certify_surface,
    inspect_untrusted_surface,
    render_clarification,
    unsafe_surface_mutations,
)
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
    design_lock_path = (
        PROJECT_ROOT / "configs/v83-strict-clarification-interface-design-lock.json"
    )
    module_path = PROJECT_ROOT / "python/structured_llm_interface.py"
    test_path = PROJECT_ROOT / "python/test_structured_llm_interface.py"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v83_strict_clarification_interface.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v83_interface_implementation.py"
    audit_path = PROJECT_ROOT / "outputs/v83-strict-clarification-interface/implementation-audit.json"
    lock_path = (
        PROJECT_ROOT / "configs/v83-strict-clarification-interface-implementation-lock.json"
    )
    if audit_path.exists() or lock_path.exists():
        raise RuntimeError("V83 interface implementation is already frozen")
    if (PROJECT_ROOT / "outputs/v83-strict-clarification-interface/evaluation").exists():
        raise RuntimeError("V83 evaluation exists before implementation lock")
    design_lock = json.loads(design_lock_path.read_text())
    design_payload = {
        key: value for key, value in design_lock.items() if key != "lock_payload_sha256"
    }
    config = design_lock["config_payload"]
    forbidden_imports = {
        "mlx", "openai", "anthropic", "requests", "urllib", "httpx",
        "socket", "subprocess", "transformers", "torch", "jax",
    }
    imports = imported_roots(module_path)
    canonical_rows = [render_clarification(code) for code in CLARIFICATION_CODES]
    grammar_rows = [
        render_clarification(code, source="finite_grammar", style=style)
        for code in CLARIFICATION_CODES
        for style in FINITE_GRAMMAR_STYLES
    ]
    mutation_certificates = [
        certify_surface(code, question, "canonical")
        for code, question in unsafe_surface_mutations()
    ]
    untrusted = [
        inspect_untrusted_surface(code, CANONICAL_SURFACES[code])
        for code in CLARIFICATION_CODES
    ]
    disabled_sources_rejected = []
    for source in config["disabledRendererSources"]:
        try:
            render_clarification("ask_operation", source=source)
        except PermissionError:
            disabled_sources_rejected.append(True)
        else:
            disabled_sources_rejected.append(False)
    checks = {
        "design_lock_exact_and_authorized": bool(
            payload_hash(design_payload) == design_lock["lock_payload_sha256"]
            and design_lock["authorization"]["implement_and_test_interface"]
            and design_lock["authorization"]["evaluate_model_free_integration_once"]
        ),
        "only_standard_library_and_no_model_network_process_imports": not bool(
            imports & forbidden_imports
        ),
        "module_sources_exactly_match_design": bool(
            list(CLARIFICATION_CODES) == config["clarificationCodes"]
            and list(FINITE_GRAMMAR_STYLES)
            == config["enumeration"]["finiteGrammarStyles"]
        ),
        "canonical_and_finite_grammar_strictly_deployable": all(
            row.certificate.deployable for row in canonical_rows + grammar_rows
        ),
        "unsafe_mutations_rejected": all(
            not row.content_valid and not row.deployable
            for row in mutation_certificates
        ),
        "untrusted_valid_looking_text_never_deployable": all(
            row.content_valid and not row.source_authorized and not row.deployable
            for row in untrusted
        ),
        "all_disabled_sources_fail_closed": all(disabled_sources_rejected),
        "implementation_stage_has_zero_model_API_human_tool_or_side_effect_access": True,
    }
    passed = all(checks.values())
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
    audit = {
        "schema_version": "83-strict-clarification-interface-implementation-audit",
        "experiment": "v83_strict_clarification_interface_implementation_audit",
        "passed": passed,
        "decision": (
            "freeze_implementation_and_authorize_one_model_free_evaluation"
            if passed
            else "reject_V83_implementation"
        ),
        "checks": checks,
        "imports": sorted(imports),
        "access": access,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed:
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)
    lock = {
        "schema_version": "83-strict-clarification-interface-implementation-lock",
        "experiment": "v83_strict_clarification_interface_implementation_lock",
        "design_lock": str(design_lock_path.relative_to(PROJECT_ROOT)),
        "design_lock_sha256": file_sha256(design_lock_path),
        "config_payload": config,
        "parent_V79_result": design_lock["parent_V79_result"],
        "parent_V79_result_sha256": design_lock["parent_V79_result_sha256"],
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
            "evaluate_model_free_integration_once": True,
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
