#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--result",default="outputs/v46-oracle-stochastic-transitions/development/result.json"); p.add_argument("--audit",default="outputs/v46-oracle-stochastic-transitions/post-result-audit.json"); p.add_argument("--output",default="configs/v46-outcome-lock.json"); a=p.parse_args(); result_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.result,a.audit,a.output))
 if output.exists(): raise RuntimeError("V46 outcome already frozen")
 result=json.loads(result_path.read_text()); audit=json.loads(audit_path.read_text())
 if not audit["passed"] or audit["result_sha256"]!=file_sha256(result_path): raise RuntimeError("V46 post-result audit failed")
 passed=result["qualification"]["passed"]; lock={"schema_version":46,"experiment":"v46_outcome_lock","scientific_decision":result["decision"],"qualification_passed":passed,"metrics":result["metrics"],"gate_checks":result["qualification"]["checks"],"result":str(result_path.relative_to(PROJECT_ROOT)),"result_sha256":file_sha256(result_path),"post_result_audit":str(audit_path.relative_to(PROJECT_ROOT)),"post_result_audit_sha256":file_sha256(audit_path),"authorization":{"preregister_sampled_transition_estimation":passed,"construct_sampled_transition_population":False,"sample_outcomes":False,"language_grounding":False,"active_intervention_selection":False,"open_ontology":False,"final_evaluation":False,"model_access":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
