#!/usr/bin/env python3
"""Independently reconstruct and freeze the V86 hardened validator outcome."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


STYLES=("concise","neutral","polite","formal","direct","friendly","explicit","minimal")
WRAPPERS={"concise":"{core}?","neutral":"Please clarify: {core}?","polite":"Could you please clarify: {core}?","formal":"Please specify: {core}?","direct":"Direct clarification: {core}?","friendly":"Could you help me clarify: {core}?","explicit":"For clarity: {core}?","minimal":"{core}?"}
ACTION_MAP={"ask_operation":("slot","operation"),"ask_recipient":("slot","recipient"),"ask_full_details":("all",None)}


def payload_hash(value:dict[str,Any])->str:return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def close(a:Any,b:Any,t:float=1e-12)->bool:
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(close(a[k],b[k],t) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(close(x,y,t) for x,y in zip(a,b))
    if isinstance(a,float) or isinstance(b,float):return abs(float(a)-float(b))<=t
    return a==b
def req(schema:str,kind:str,slot:str|None)->dict[str,Any]:return {"schema_id":schema,"kind":kind,"slot_id":slot}
def choice(slot:dict[str,Any])->str:return f"{slot['options'][0]['surface']} or {slot['options'][1]['surface']}"
def canonical(schema:dict[str,Any],request:dict[str,Any])->str:
    slots=schema["slots"] if request["kind"]=="all" else [s for s in schema["slots"] if s["slotId"]==request["slot_id"]]
    clauses=[f"{s['questionPrefix']} {choice(s)}" for s in slots]
    if len(clauses)==1:return clauses[0]+"?"
    return f"{clauses[0]}, and {clauses[1][0].lower()+clauses[1][1:]}?"
def render(schema,request,source,style):
    base=canonical(schema,request)
    if source=="canonical":return base
    core=base[:-1];core=core[0].lower()+core[1:];return WRAPPERS[style].format(core=core)


def main()->None:
    impl_path=PROJECT_ROOT/"configs/v86-partial-option-validator-implementation-lock.json"
    result_path=PROJECT_ROOT/"outputs/v86-partial-option-validator/evaluation/result.json"
    verifier_path=PROJECT_ROOT/"python/verify_and_freeze_v86_validator_outcome.py"
    doc_path=PROJECT_ROOT/"docs/v86-partial-option-validator-results.md"
    audit_path=PROJECT_ROOT/"outputs/v86-partial-option-validator/outcome-audit.json"
    lock_path=PROJECT_ROOT/"configs/v86-partial-option-validator-outcome-lock.json"
    if audit_path.exists() or lock_path.exists():raise RuntimeError("V86 outcome already frozen")
    impl=json.loads(impl_path.read_text());impl_payload={k:v for k,v in impl.items() if k!="lock_payload_sha256"};schemas=impl["schemas"];result=json.loads(result_path.read_text())
    surface=[];untrusted=[]
    for schema in schemas:
        requests=[req(schema["schemaId"],"slot",s["slotId"]) for s in schema["slots"]]+[req(schema["schemaId"],"all",None)]
        for request in requests:
            for source,style in [("canonical",None)]+[("finite_grammar",s) for s in STYLES]:
                surface.append({"typed_request":request,"source":source,"style":style,"question":render(schema,request,source,style),"content_valid":True,"deployable":True,"typed_request_preserved":True})
            untrusted.append({"typed_request":request,"content_valid":True,"deployable":False})
    base=[];partial=[]
    for schema in schemas:
        first,second=schema["slots"]
        base_questions=(
            (req(schema["schemaId"],"slot",first["slotId"]),f"{first['questionPrefix']} {first['options'][0]['surface']} and {first['options'][1]['surface']}?"),
            (req(schema["schemaId"],"slot",second["slotId"]),f"{second['questionPrefix']} {second['options'][0]['surface']}?"),
            (req(schema["schemaId"],"slot",second["slotId"]),f"{second['questionPrefix']} {choice(second)}, and {choice(first)}?"),
            (req(schema["schemaId"],"slot",first["slotId"]),f"I will {choice(first)}?"),
        )
        base.extend({"typed_request":r,"question":q,"rejected":True} for r,q in base_questions)
        for requested in schema["slots"]:
            request=req(schema["schemaId"],"slot",requested["slotId"]);start=canonical(schema,request)[:-1]
            other=next(s for s in schema["slots"] if s is not requested)
            for option in other["options"]:
                partial.append({"typed_request":request,"question":f"{start}, and {option['surface']}?","injected_option_surface":option["surface"],"individual_unrequested_option_surface_count":1,"rejected":True})
    artifact=json.loads((PROJECT_ROOT/impl["V85_false_positive_artifact"]).read_text())
    v85=[{"id":artifact["id"],"question":artifact["question"],"individual_unrequested_option_surface_count":1,"content_valid":False,"deployable":False,"rejected":True}]
    parent_path=PROJECT_ROOT/impl["parent_V79_result"];parent=json.loads(parent_path.read_text());bridge=[];nonask=[];values={};violations=0;ask_count=0
    for fname,fixture in sorted(parent["fixtures"].items()):
        exact=fixture["exact"];values[fname]=exact["value"];violations+=exact["complete_belief_certificate_violation_count"]
        for idx,node in enumerate(exact["policy_nodes"]):
            if node["action"] in ACTION_MAP:
                ask_count+=1
                for source,style in [("canonical",None)]+[("finite_grammar",s) for s in STYLES]:bridge.append({"fixture":fname,"node_index":idx,"action":node["action"],"source":source,"style":style,"action_preserved":True,"structurally_preserved":True})
            else:nonask.append({"fixture":fname,"node_index":idx,"structurally_identical":True})
    metrics={"schema_rendered_case_count":len(surface),"V79_bridge_node_count":ask_count,"V79_bridge_rendered_case_count":len(bridge),"base_unsafe_mutation_count":len(base),"partial_option_injection_mutation_count":len(partial),"V85_false_positive_regression_count":1,"schema_surface_validity_rate":1.0,"typed_request_preservation_rate":1.0,"base_unsafe_mutation_rejection_rate":1.0,"partial_option_injection_rejection_rate":1.0,"V85_false_positive_regression_rejection_rate":1.0,"disabled_untrusted_deployment_rate":1.0,"V79_bridge_action_preservation_rate":1.0,"V79_bridge_structural_preservation_rate":1.0,"maximum_V79_policy_value_absolute_error":0.0,"complete_belief_execution_certificate_violation_count":violations}
    checks={
        "implementation_lock_and_sources_exact":payload_hash(impl_payload)==impl["lock_payload_sha256"] and file_sha256(PROJECT_ROOT/impl["module"])==impl["module_sha256"] and file_sha256(PROJECT_ROOT/impl["evaluator"])==impl["evaluator_sha256"],
        "parent_V79_and_V85_artifact_exact":file_sha256(parent_path)==impl["parent_V79_result_sha256"] and file_sha256(PROJECT_ROOT/impl["V85_false_positive_artifact"])==impl["V85_false_positive_artifact_sha256"],
        "safe_surface_rows_reconstructed":close(result["surface_rows"],surface),"base_unsafe_rows_reconstructed":close(result["base_unsafe_rows"],base),"partial_option_rows_reconstructed":close(result["partial_option_rows"],partial),"V85_regression_reconstructed":close(result["V85_regression_rows"],v85),"untrusted_rows_reconstructed":close(result["untrusted_rows"],untrusted),"V79_rows_reconstructed":close(result["V79_bridge_rows"],bridge) and close(result["V79_nonclarification_rows"],nonask),"metrics_and_values_reconstructed":close(result["metrics"],metrics) and close(result["fixture_values"],values),"all_registered_gates_pass_and_access_zero":result["passed"] and all(result["checks"].values()) and all(v==0 for v in result["access"].values())}
    passed=all(checks.values());audit={"schema_version":"86-partial-option-validator-outcome-audit","experiment":"v86_partial_option_validator_outcome_audit","passed":passed,"decision":"freeze_positive_hardened_validator_outcome" if passed else "reject_V86_outcome","checks":checks,"independent_metrics":metrics,"claim_boundary":result["claim_boundary"]}
    audit_path.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    if not passed:print(json.dumps(audit,indent=2,sort_keys=True));raise SystemExit(1)
    lock={"schema_version":"86-partial-option-validator-outcome-lock","experiment":"v86_partial_option_validator_outcome_lock","implementation_lock":str(impl_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(impl_path),"result":str(result_path.relative_to(PROJECT_ROOT)),"result_sha256":file_sha256(result_path),"verifier":str(verifier_path.relative_to(PROJECT_ROOT)),"verifier_sha256":file_sha256(verifier_path),"audit":str(audit_path.relative_to(PROJECT_ROOT)),"audit_sha256":file_sha256(audit_path),"results_document":str(doc_path.relative_to(PROJECT_ROOT)),"results_document_sha256":file_sha256(doc_path),"outcome":{"passed":True,"decision":result["decision"],"metrics":result["metrics"]},"authorization":{"modify_or_rerun_V86":False,"use_hardened_validator_in_future_bounded_shadow_integrations":True,"deploy_model_or_untrusted_surface":False,"access_local_or_API_model_or_train_adapter":False,"grant_schema_surface_belief_action_or_execution_authority":False,"perform_real_tool_call_or_external_side_effect":False,"preregister_fresh_independently_authored_language_shadow_evaluation":True}}
    lock["lock_payload_sha256"]=payload_hash(lock);lock_path.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n");print(json.dumps(audit,indent=2,sort_keys=True));print(json.dumps({"lock":str(lock_path),"sha256":file_sha256(lock_path)},indent=2))


if __name__=="__main__":main()
