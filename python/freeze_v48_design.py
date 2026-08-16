#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v48-stochastic-language-composition.json"); p.add_argument("--audit",default="outputs/v48-stochastic-language-composition/design-audit.json"); p.add_argument("--output",default="configs/v48-design-lock.json"); a=p.parse_args(); config,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.config,a.audit,a.output)); audit=json.loads(audit_path.read_text())
 if output.exists() or not audit["passed"] or audit["config_sha256"]!=file_sha256(config): raise RuntimeError("Cannot freeze V48 design")
 c=json.loads(config.read_text()); source=PROJECT_ROOT/c["sourceV47OutcomeLock"]; plan=PROJECT_ROOT/"docs/v48-stochastic-language-composition-plan.md"; lock={"schema_version":48,"experiment":"v48_design_lock","config":str(config.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config),"config_payload":c,"preregistration":str(plan.relative_to(PROJECT_ROOT)),"preregistration_sha256":file_sha256(plan),"design_audit":str(audit_path.relative_to(PROJECT_ROOT)),"design_audit_sha256":file_sha256(audit_path),"source_v47_outcome_lock":str(source.relative_to(PROJECT_ROOT)),"source_v47_outcome_lock_sha256":file_sha256(source),"authorization":{"write_composition_implementation":True,"construct_development_population":False,"run_development":False,"model_access":False}}; lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
