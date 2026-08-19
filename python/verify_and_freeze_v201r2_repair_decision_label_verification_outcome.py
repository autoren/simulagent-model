#!/usr/bin/env python3
from __future__ import annotations
import json
from typing import Any
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v201r2_repair_decision_label_verification import evaluate_repair
from v22r2_grounding import PROJECT_ROOT


def write_json(path,value:Any)->None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def main()->None:
    lp=PROJECT_ROOT/"configs/v201r2-repair-decision-label-verification-lock.json"; l=json.loads(lp.read_text()); out=PROJECT_ROOT/"outputs/v201r2-repair-decision-label-verification/repair"; ap=PROJECT_ROOT/"outputs/v201r2-repair-decision-label-verification/outcome-audit.json"; op=PROJECT_ROOT/"configs/v201r2-repair-decision-label-verification-outcome-lock.json"; dp=PROJECT_ROOT/"docs/v201r2-repair-decision-label-verification-results.md"
    if ap.exists() or op.exists(): raise RuntimeError("V201r2 outcome exists")
    exact=valid_lock(l) and all(file_sha256(PROJECT_ROOT/l[k])==l[f"{k}_sha256"] for k in l if not k.endswith("_sha256") and f"{k}_sha256" in l); c=l["config_payload"]; r=evaluate_repair(json.loads((PROJECT_ROOT/l["source_failed_outcome_audit"]).read_text()),json.loads((PROJECT_ROOT/l["source_repair_result"]).read_text()),json.loads((PROJECT_ROOT/l["source_V201_result"]).read_text()),c); decision=c["decisionRule"]["ifExactDecisionOverwriteAndEverySubstantiveRepairCheckPasses" if r["passed"] else "otherwise"]; result_path=out/"result.json"; stored=json.loads(result_path.read_text()); result_exact=stored["passed"]==r["passed"] and stored["decision"]==decision and stored["repair"]==r and stored["source_artifact_mutation_count"]==stored["model_policy_or_scoring_rerun_count"]==stored["raw_model_response_read_count"]==stored["API_call_count"]==stored["actual_execution_count"]==0; checks={"lock_dependencies_exact":exact,"repair_exact":result_exact,"repair_passes":r["passed"],"source_hashes_exact":all(file_sha256(PROJECT_ROOT/l[k])==l[f"{k}_sha256"] for k in ("source_repair_result","source_V201_result","source_V201_summary","source_V201_scored_records","source_V201_access","source_V201_census")),"results_document_exists":dp.is_file(),"zero_mutation_model_raw_API_execution":stored["source_artifact_mutation_count"]==stored["model_policy_or_scoring_rerun_count"]==stored["raw_model_response_read_count"]==stored["API_call_count"]==stored["actual_execution_count"]==0}; passed=all(checks.values()); audit={"schema_version":"201r2-outcome-audit","experiment":l["experiment"],"passed":passed,"checks":checks,"decision":"freeze_verified_V201r2" if passed else "freeze_failed_V201r2"}; write_json(ap,audit)
    if not passed: print(json.dumps(audit,indent=2,sort_keys=True)); raise SystemExit(1)
    deps={"repair_lock":lp,"audit":ap,"result":result_path,"source_V201r1_result":PROJECT_ROOT/l["source_repair_result"],"source_V201_result":PROJECT_ROOT/l["source_V201_result"],"source_V201_summary":PROJECT_ROOT/l["source_V201_summary"],"source_V201_scored_records":PROJECT_ROOT/l["source_V201_scored_records"],"source_V201_access":PROJECT_ROOT/l["source_V201_access"],"source_V201_census":PROJECT_ROOT/l["source_V201_census"],"source_V201_results_document":PROJECT_ROOT/l["source_V201_results_document"],"source_V201r1_results_document":PROJECT_ROOT/l["source_V201r1_results_document"],"results_document":dp,"verifier":PROJECT_ROOT/l["verifier"]}; summary=json.loads((PROJECT_ROOT/l["source_V201_summary"]).read_text()); outcome:dict[str,Any]={"schema_version":"201r2-outcome-lock","experiment":l["experiment"],"outcome":{"passed":True,"V201_scientific_qualification_gates_passed":False,"decision":"freeze_V201_negative_or_presentation_sensitive_without_retry_reprompt_model_selection_or_API","V201_summary":summary},"authorization":{"update_roadmap_and_preregister_separate_model_free_decision_sufficiency_design":True,"run_paired_protected_robustness":False,"run_API_additional_model_synthetic_language_registration_authority_action_or_execution":False}}
    for k,p in deps.items(): outcome[k]=str(p.relative_to(PROJECT_ROOT)); outcome[f"{k}_sha256"]=file_sha256(p)
    outcome["lock_payload_sha256"]=payload_hash(outcome); write_json(op,outcome); print(json.dumps(audit,indent=2,sort_keys=True)); print(json.dumps({"outcome_lock":str(op.relative_to(PROJECT_ROOT)),"sha256":file_sha256(op)},indent=2))


if __name__=="__main__": main()
