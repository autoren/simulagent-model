#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v47-sampled-transition-estimation.json"); p.add_argument("--audit",default="outputs/v47-sampled-transition-estimation/design-audit.json"); p.add_argument("--output",default="configs/v47-design-lock.json"); a=p.parse_args(); config_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.config,a.audit,a.output))
 if output.exists(): raise RuntimeError("V47 design already frozen")
 audit=json.loads(audit_path.read_text())
 if not audit["passed"] or audit["config_sha256"]!=file_sha256(config_path): raise RuntimeError("V47 design audit failed")
 c=json.loads(config_path.read_text()); source=PROJECT_ROOT/c["sourceV46OutcomeLock"]; plan=PROJECT_ROOT/"docs/v47-sampled-transition-estimation-plan.md"; lock={"schema_version":47,"experiment":"v47_design_lock","config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"config_payload":c,"preregistration":str(plan.relative_to(PROJECT_ROOT)),"preregistration_sha256":file_sha256(plan),"design_audit":str(audit_path.relative_to(PROJECT_ROOT)),"design_audit_sha256":file_sha256(audit_path),"source_v46_outcome_lock":str(source.relative_to(PROJECT_ROOT)),"source_v46_outcome_lock_sha256":file_sha256(source),"authorization":{"write_estimator_implementation":True,"construct_sampled_population":False,"run_sampled_development":False,"model_access":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
