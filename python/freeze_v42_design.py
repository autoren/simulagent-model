#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v42-sequential-state-foundation.json"); p.add_argument("--audit",default="outputs/v42-sequential-state-foundation/design-audit.json"); p.add_argument("--output",default="configs/v42-design-lock.json"); a=p.parse_args(); config_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.config,a.audit,a.output))
 if output.exists(): raise RuntimeError("V42 design already frozen")
 audit=json.loads(audit_path.read_text());
 if not audit["passed"] or audit["config_sha256"]!=file_sha256(config_path): raise RuntimeError("V42 design audit did not pass")
 c=json.loads(config_path.read_text()); source_path=PROJECT_ROOT/c["sourceV41OutcomeLock"]; lock={"schema_version":42,"experiment":"v42_design_lock","config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"config_payload":c,"preregistration":"docs/v42-sequential-state-foundation-plan.md","preregistration_sha256":file_sha256(PROJECT_ROOT/"docs/v42-sequential-state-foundation-plan.md"),"design_audit":str(audit_path.relative_to(PROJECT_ROOT)),"design_audit_sha256":file_sha256(audit_path),"source_v41_outcome_lock":str(source_path.relative_to(PROJECT_ROOT)),"source_v41_outcome_lock_sha256":file_sha256(source_path),"authorization":{"write_oracle_implementation":True,"construct_development_population":False,"run_oracle_development":False,"language_grounding":False,"model_access":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
