#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--result",default="outputs/v41-relational-mechanic-confirmation/evaluation/result.json"); p.add_argument("--audit",default="outputs/v41-relational-mechanic-confirmation/post-result-audit.json"); p.add_argument("--output",default="configs/v41-outcome-lock.json"); a=p.parse_args(); result_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.result,a.audit,a.output))
 if output.exists(): raise RuntimeError("V41 outcome already frozen")
 result=json.loads(result_path.read_text()); audit=json.loads(audit_path.read_text());
 if not audit["passed"] or audit["result_sha256"]!=file_sha256(result_path): raise RuntimeError("V41 post-result audit did not pass")
 passed=result["qualification"]["passed"]; lock={"schema_version":41,"experiment":"v41_outcome_lock","scientific_decision":result["decision"],"qualification_passed":passed,"metrics":result["metrics"],"gate_checks":result["qualification"]["checks"],"result":str(result_path.relative_to(PROJECT_ROOT)),"result_sha256":file_sha256(result_path),"post_result_audit":str(audit_path.relative_to(PROJECT_ROOT)),"post_result_audit_sha256":file_sha256(audit_path),"authorization":{"begin_architecture_breaking_benchmark":passed,"construct_architecture_breaking_benchmark":False,"expand_declared_scope_claim":False,"v22r2_evaluation":False,"v28":False,"adapter_training":False,"change_compiler":False,"change_semantic_kernel":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
