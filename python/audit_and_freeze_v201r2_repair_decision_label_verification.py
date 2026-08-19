#!/usr/bin/env python3
from __future__ import annotations
import json
from typing import Any
from audit_and_freeze_v165_factored_ontology_identifiability_population import valid_lock
from audit_and_freeze_v167_exact_evidence_gathering_planner import payload_hash
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def write_json(path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    cp=PROJECT_ROOT/"configs/v201r2-repair-decision-label-verification.json"; pp=PROJECT_ROOT/"docs/v201r2-repair-decision-label-verification-plan.md"; pr=PROJECT_ROOT/"python/v201r2_repair_decision_label_verification.py"; tp=PROJECT_ROOT/"python/test_v201r2_repair_decision_label_verification.py"; rp=PROJECT_ROOT/"python/run_v201r2_repair_decision_label_verification.py"; vp=PROJECT_ROOT/"python/verify_and_freeze_v201r2_repair_decision_label_verification_outcome.py"; ap=PROJECT_ROOT/"python/audit_and_freeze_v201r2_repair_decision_label_verification.py"; dap=PROJECT_ROOT/"outputs/v201r2-repair-decision-label-verification/design-audit.json"; lp=PROJECT_ROOT/"configs/v201r2-repair-decision-label-verification-lock.json"; out=PROJECT_ROOT/"outputs/v201r2-repair-decision-label-verification/repair"; op=PROJECT_ROOT/"configs/v201r2-repair-decision-label-verification-outcome-lock.json"
    if any(p.exists() for p in (dap,lp,out,op)): raise RuntimeError("V201r2 already started")
    c=json.loads(cp.read_text()); names=(("source_V201r1_lock","sourceV201r1Lock"),("source_failed_outcome_audit","sourceFailedOutcomeAudit"),("source_repair_result","sourceRepairResult"),("source_V201_result","sourceV201Result"),("source_V201_summary","sourceV201Summary"),("source_V201_scored_records","sourceV201ScoredRecords"),("source_V201_access","sourceV201Access"),("source_V201_census","sourceV201Census"),("source_V201_results_document","sourceV201ResultsDocument"),("source_V201r1_results_document","sourceV201r1ResultsDocument")); inputs={k:PROJECT_ROOT/c[n] for k,n in names}; sl=json.loads(inputs["source_V201r1_lock"].read_text()); fa=json.loads(inputs["source_failed_outcome_audit"].read_text())
    checks={"V201r1_lock_and_dependencies_are_exact":valid_lock(sl) and all(file_sha256(PROJECT_ROOT/sl[k])==sl[f"{k}_sha256"] for k in sl if not k.endswith("_sha256") and f"{k}_sha256" in sl),"V201r1_failed_only_expected_label_check":not fa["passed"] and [k for k,v in fa["checks"].items() if not v]==["repair_reconstructs_exactly"],"scope_is_serialization_only":not c["repairContract"]["sourceArtifactsMayBeModified"] and not c["repairContract"]["modelPolicyOrScoringMayBeRerun"],"prelock_access_zero":all(v==0 for v in c["preLockExposure"].values()),"authority_closed":not c["decisionRule"]["passAuthorizesProtectedAPITrainingRegistrationAuthorityActionOrExecution"],"files_exist_outputs_absent":all(p.is_file() for p in (cp,pp,pr,tp,rp,vp,ap,*inputs.values())) and not out.exists()}; passed=all(checks.values()); audit={"schema_version":"201r2-design-audit","experiment":c["experiment"],"passed":passed,"checks":checks,"decision":"freeze_V201r2_design" if passed else "reject_V201r2_design"}; write_json(dap,audit)
    if not passed: print(json.dumps(audit,indent=2,sort_keys=True)); raise SystemExit(1)
    deps={"config":cp,**inputs,"plan":pp,"protocol":pr,"tests":tp,"runner":rp,"verifier":vp,"auditor":ap,"design_audit":dap}; lock:dict[str,Any]={"schema_version":"201r2-repair-decision-label-verification-lock","experiment":c["experiment"],"config_payload":c,"authorization":{"modify_sources_or_rerun":False,"run_exact_single_serialization_verification":True,"protected_API_training_registration_authority_action_or_execution":False}}
    for k,p in deps.items(): lock[k]=str(p.relative_to(PROJECT_ROOT)); lock[f"{k}_sha256"]=file_sha256(p)
    lock["lock_payload_sha256"]=payload_hash(lock); write_json(lp,lock); print(json.dumps(audit,indent=2,sort_keys=True)); print(json.dumps({"lock":str(lp.relative_to(PROJECT_ROOT)),"sha256":file_sha256(lp)},indent=2))


if __name__=="__main__": main()
