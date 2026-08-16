#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v39-declared-language-compiler.json"); p.add_argument("--audit",default="outputs/v39-declared-language-compiler/design-audit.json"); p.add_argument("--output",default="configs/v39-declared-language-compiler-lock.json"); a=p.parse_args(); config_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.config,a.audit,a.output));
 if output.exists(): raise RuntimeError("V39 design already frozen")
 audit=json.loads(audit_path.read_text());
 if not audit["passed"]: raise RuntimeError("V39 design audit failed")
 c=json.loads(config_path.read_text()); lock={"schema_version":39,"experiment":"v39_declared_language_compiler_design_lock","config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"config_payload":c,"preregistration":"docs/v39-declared-language-compiler-plan.md","preregistration_sha256":file_sha256(PROJECT_ROOT/"docs/v39-declared-language-compiler-plan.md"),"design_audit":str(audit_path.relative_to(PROJECT_ROOT)),"design_audit_sha256":file_sha256(audit_path),"authorization":{"write_implementation":True,"construct_evaluation":False,"score_evaluation":False,"preregister_confirmation":False,"v32_evaluation":False,"v28":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
