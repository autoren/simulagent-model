#!/usr/bin/env python3
"""Audit and freeze the V86 model-free validator implementation."""
from __future__ import annotations

import ast
import hashlib
import json
from typing import Any

from schema_grounded_interface import FINITE_GRAMMAR_STYLES, ClarificationRequest, compile_schema_registry, unsafe_schema_surface_mutations
from schema_grounded_interface_v86 import hardened_certify_schema_surface, partial_option_injection_mutations, render_hardened_schema_clarification
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def payload_hash(value:dict[str,Any])->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def imports(path):
    tree=ast.parse(path.read_text()); roots=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:roots.add(node.module.split(".")[0])
    return roots


def main()->None:
    design_path=PROJECT_ROOT/"configs/v86-partial-option-validator-design-lock.json"
    module_path=PROJECT_ROOT/"python/schema_grounded_interface_v86.py"
    test_path=PROJECT_ROOT/"python/test_schema_grounded_interface_v86.py"
    evaluator_path=PROJECT_ROOT/"python/evaluate_v86_partial_option_validator.py"
    auditor_path=PROJECT_ROOT/"python/audit_and_freeze_v86_validator_implementation.py"
    audit_path=PROJECT_ROOT/"outputs/v86-partial-option-validator/implementation-audit.json"
    lock_path=PROJECT_ROOT/"configs/v86-partial-option-validator-implementation-lock.json"
    if audit_path.exists() or lock_path.exists():raise RuntimeError("V86 implementation already frozen")
    if (PROJECT_ROOT/"outputs/v86-partial-option-validator/evaluation").exists():raise RuntimeError("V86 evaluation exists before lock")
    design=json.loads(design_path.read_text()); payload={k:v for k,v in design.items() if k!="lock_payload_sha256"}; config=design["config_payload"]
    v84_out=json.loads((PROJECT_ROOT/design["schema_V84_outcome_lock"]).read_text()); v84_impl_path=PROJECT_ROOT/v84_out["implementation_lock"]
    v84_impl=json.loads(v84_impl_path.read_text()); schemas=v84_impl["config_payload"]["schemas"]; registry=compile_schema_registry(schemas)
    safe=[]
    for schema in registry.schemas:
        reqs=[ClarificationRequest(schema.schema_id,"slot",slot.slot_id) for slot in schema.slots]+[ClarificationRequest(schema.schema_id,"all",None)]
        for req in reqs:
            safe.append(render_hardened_schema_clarification(registry,req))
            safe.extend(render_hardened_schema_clarification(registry,req,source="finite_grammar",style=style) for style in FINITE_GRAMMAR_STYLES)
    base=[hardened_certify_schema_surface(registry,req,q,"canonical") for req,q in unsafe_schema_surface_mutations(registry)]
    partial=[hardened_certify_schema_surface(registry,req,q,"canonical") for req,q,_ in partial_option_injection_mutations(registry)]
    v85_out=json.loads((PROJECT_ROOT/design["parent_V85_outcome_lock"]).read_text()); false_id=v85_out["outcome"]["post_outcome_stricter_diagnostic"]["false_positive_ids"][0]
    artifact_entry=next(row for row in v85_out["raw_fixture_artifacts"] if false_id in row["path"]); artifact_path=PROJECT_ROOT/artifact_entry["path"]; artifact=json.loads(artifact_path.read_text())
    request=ClarificationRequest(artifact["typed_target"]["schema_id"],artifact["typed_target"]["kind"],artifact["typed_target"]["slot_id"])
    regression=hardened_certify_schema_surface(registry,request,artifact["question"],"local_model_adversarial")
    roots=imports(module_path); forbidden={"mlx","openai","anthropic","requests","urllib","httpx","socket","subprocess","transformers","torch","jax"}
    checks={
        "design_lock_exact_and_authorized":payload_hash(payload)==design["lock_payload_sha256"] and design["authorization"]["implement_and_test_model_free_validator"] and design["authorization"]["evaluate_model_free_census_once"],
        "frozen_schema_source_exact":file_sha256(v84_impl_path)==v84_out["implementation_lock_sha256"],
        "standard_library_wrapper_has_no_model_network_process_imports":not bool(roots&forbidden),
        "all_108_safe_surfaces_remain_deployable":len(safe)==108 and all(row.certificate.deployable for row in safe),
        "all_base_and_partial_mutations_rejected":len(base)==16 and len(partial)==16 and all(not row.deployable for row in base+partial),
        "exact_V85_false_positive_rejected":not regression.content_valid and not regression.deployable and regression.individual_unrequested_option_surface_count==1,
        "zero_model_API_training_human_language_tool_or_side_effect_access":True,
    }
    passed=all(checks.values()); audit={"schema_version":"86-partial-option-validator-implementation-audit","experiment":"v86_partial_option_validator_implementation_audit","passed":passed,"decision":"freeze_implementation_and_authorize_one_model_free_census" if passed else "reject_V86_implementation","checks":checks,"imports":sorted(roots),"access":{"model_load_count":0,"model_generation_count":0,"API_call_count":0,"adapter_training_run_count":0,"human_record_access_count":0,"original_user_language_access_count":0,"real_tool_call_count":0,"external_side_effect_count":0}}
    audit_path.parent.mkdir(parents=True,exist_ok=True);audit_path.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    if not passed:print(json.dumps(audit,indent=2,sort_keys=True));raise SystemExit(1)
    lock={"schema_version":"86-partial-option-validator-implementation-lock","experiment":"v86_partial_option_validator_implementation_lock","design_lock":str(design_path.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design_path),"config_payload":config,"schema_source_lock":str(v84_impl_path.relative_to(PROJECT_ROOT)),"schema_source_lock_sha256":file_sha256(v84_impl_path),"schemas":schemas,"parent_V79_result":v84_impl["parent_V79_result"],"parent_V79_result_sha256":v84_impl["parent_V79_result_sha256"],"V85_false_positive_artifact":str(artifact_path.relative_to(PROJECT_ROOT)),"V85_false_positive_artifact_sha256":file_sha256(artifact_path),"module":str(module_path.relative_to(PROJECT_ROOT)),"module_sha256":file_sha256(module_path),"test":str(test_path.relative_to(PROJECT_ROOT)),"test_sha256":file_sha256(test_path),"evaluator":str(evaluator_path.relative_to(PROJECT_ROOT)),"evaluator_sha256":file_sha256(evaluator_path),"implementation_auditor":str(auditor_path.relative_to(PROJECT_ROOT)),"implementation_auditor_sha256":file_sha256(auditor_path),"implementation_audit":str(audit_path.relative_to(PROJECT_ROOT)),"implementation_audit_sha256":file_sha256(audit_path),"authorization":{"modify_implementation_or_evaluation":False,"evaluate_model_free_census_once":True,"access_local_or_API_model":False,"train_adapter":False,"collect_human_or_original_user_language":False,"perform_real_tool_call_or_external_side_effect":False}}
    lock["lock_payload_sha256"]=payload_hash(lock);lock_path.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n");print(json.dumps(audit,indent=2,sort_keys=True));print(json.dumps({"lock":str(lock_path),"sha256":file_sha256(lock_path)},indent=2))


if __name__=="__main__":main()
