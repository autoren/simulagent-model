#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--result",default="outputs/v38-ontology-anchored-focus-parser/evaluation/result.json"); p.add_argument("--audit",default="outputs/v38-ontology-anchored-focus-parser/post-result-audit.json"); p.add_argument("--output",default="configs/v38-outcome-lock.json"); a=p.parse_args(); result_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.result,a.audit,a.output));
 if output.exists(): raise RuntimeError("V38 outcome already frozen")
 result=json.loads(result_path.read_text()); audit=json.loads(audit_path.read_text());
 if not audit["passed"] or audit["result_sha256"]!=file_sha256(result_path): raise RuntimeError("V38 audit did not pass")
 passed=result["qualification"]["passed"]; lock={"schema_version":38,"experiment":"v38_outcome_lock","scientific_decision":result["decision"],"qualification_passed":passed,"focus_selection":result["focus_selection"],"selected_validation":result["selected_validation"],"gate_checks":result["qualification"]["checks"],"result":str(result_path.relative_to(PROJECT_ROOT)),"result_sha256":file_sha256(result_path),"post_result_audit":str(audit_path.relative_to(PROJECT_ROOT)),"post_result_audit_sha256":file_sha256(audit_path),"authorization":{"preregister_fresh_semantic_confirmation":passed,"construct_confirmation":False,"end_to_end_relational_suite":False,"v32_evaluation":False,"v28":False,"adapter_training":False,"change_backbone":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
