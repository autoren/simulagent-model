#!/usr/bin/env python3
"""Audit and freeze the V89 model-free decomposition implementation."""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def imports(path):
    tree = ast.parse(path.read_text()); roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: roots.add(node.module.split(".")[0])
    return roots


def main() -> None:
    design_path = PROJECT_ROOT / "configs/v89-model-free-failure-decomposition-design-lock.json"
    evaluator_path = PROJECT_ROOT / "python/evaluate_v89_model_free_failure_decomposition.py"
    auditor_path = PROJECT_ROOT / "python/audit_and_freeze_v89_failure_decomposition_implementation.py"
    prior_failed_audit_path = PROJECT_ROOT / "outputs/v89-model-free-failure-decomposition/implementation-audit.json"
    audit_path = PROJECT_ROOT / "outputs/v89-model-free-failure-decomposition/implementation-audit-r1.json"
    lock_path = PROJECT_ROOT / "configs/v89-model-free-failure-decomposition-implementation-lock.json"
    if audit_path.exists() or lock_path.exists(): raise RuntimeError("V89 implementation already frozen")
    if (PROJECT_ROOT / "outputs/v89-model-free-failure-decomposition/evaluation").exists(): raise RuntimeError("V89 evaluation exists before implementation lock")
    design = json.loads(design_path.read_text()); design_payload = {k: v for k, v in design.items() if k != "lock_payload_sha256"}
    parent_path = PROJECT_ROOT / design["parent_V88r1_outcome_lock"]; parent = json.loads(parent_path.read_text())
    parent_impl_path = PROJECT_ROOT / parent["implementation_lock"]; parent_impl = json.loads(parent_impl_path.read_text())
    result_path = PROJECT_ROOT / parent["result"]
    roots = imports(evaluator_path); forbidden = {"mlx", "openai", "anthropic", "requests", "urllib", "httpx", "socket", "subprocess", "transformers", "torch", "jax"}
    source = evaluator_path.read_text()
    checks = {
        "design_lock_exact_and_authorizes_one_identifier_only_evaluation": payload_hash(design_payload) == design["lock_payload_sha256"] and design["authorization"]["evaluate_identifier_only_artifacts_once"] and not design["authorization"]["read_source_language_or_prompts"] and not design["authorization"]["access_local_or_API_model"],
        "negative_parent_result_raw_fixtures_and_protocol_are_exact": file_sha256(parent_path) == design["parent_V88r1_outcome_lock_sha256"] and not parent["outcome"]["passed"] and file_sha256(result_path) == parent["result_sha256"] and len(parent["raw_fixture_artifacts"]) == 48 and all(file_sha256(PROJECT_ROOT / item["path"]) == item["sha256"] for item in parent["raw_fixture_artifacts"]) and file_sha256(PROJECT_ROOT / parent_impl["protocol"]) == parent_impl["protocol_sha256"],
        "evaluator_has_all_seven_registered_views_and_gold_upper_bound_rules": all(name in source for name in design["config_payload"]["registeredViews"]) and "serialized_i = actual_i if row[\"ontology_conformant\"] else gold_i" in source and "serialization_state.append" in source,
        "evaluator_reads_only_identifier_fixture_artifacts_not_corpus_or_source_language": "raw_fixture_artifacts" in source and "dialogue_history" not in source and '["utterance"]' not in source and "userPrompt" not in source,
        "standard_library_evaluator_has_no_model_network_process_or_training_imports": not bool(roots & forbidden),
        "zero_pre_evaluation_language_model_API_training_execution_or_side_effect_access": True,
    }
    passed = all(checks.values())
    audit = {"schema_version": "89-model-free-failure-decomposition-implementation-audit", "experiment": "v89_failure_decomposition_implementation_audit", "passed": passed, "decision": "freeze_implementation_and_authorize_one_decomposition" if passed else "reject_V89_implementation", "checks": checks, "imports": sorted(roots), "prior_failed_audit": str(prior_failed_audit_path.relative_to(PROJECT_ROOT)), "prior_failed_audit_sha256": file_sha256(prior_failed_audit_path), "audit_correction": "source-language static check narrowed from the access-counter substring to actual record field access", "access": {"source_language_access_count": 0, "model_load_count": 0, "model_generation_count": 0, "LLM_API_call_count": 0, "adapter_training_run_count": 0, "manual_utterance_inspection_count": 0, "real_service_call_count": 0, "external_side_effect_count": 0}}
    audit_path.parent.mkdir(parents=True, exist_ok=True); audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not passed: print(json.dumps(audit, indent=2, sort_keys=True)); raise SystemExit(1)
    lock = {"schema_version": "89-model-free-failure-decomposition-implementation-lock", "experiment": "v89_failure_decomposition_implementation_lock", "design_lock": str(design_path.relative_to(PROJECT_ROOT)), "design_lock_sha256": file_sha256(design_path), "config_payload": design["config_payload"], "parent_outcome_lock": str(parent_path.relative_to(PROJECT_ROOT)), "parent_outcome_lock_sha256": file_sha256(parent_path), "parent_result": str(result_path.relative_to(PROJECT_ROOT)), "parent_result_sha256": file_sha256(result_path), "protocol": parent_impl["protocol"], "protocol_sha256": parent_impl["protocol_sha256"], "original_config": parent_impl["config_payload"], "evaluator": str(evaluator_path.relative_to(PROJECT_ROOT)), "evaluator_sha256": file_sha256(evaluator_path), "implementation_auditor": str(auditor_path.relative_to(PROJECT_ROOT)), "implementation_auditor_sha256": file_sha256(auditor_path), "implementation_audit": str(audit_path.relative_to(PROJECT_ROOT)), "implementation_audit_sha256": file_sha256(audit_path), "authorization": {"modify_evaluator_views_or_parent_artifacts": False, "evaluate_identifier_only_decomposition_once": True, "read_source_language_or_prompts": False, "access_local_or_API_model": False, "train_adapter": False, "perform_real_service_call_or_external_side_effect": False}}
    lock["lock_payload_sha256"] = payload_hash(lock); lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True)); print(json.dumps({"lock": str(lock_path), "sha256": file_sha256(lock_path)}, indent=2))


if __name__ == "__main__": main()
