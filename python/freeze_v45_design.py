#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v45-delayed-language-grounding.json"); p.add_argument("--audit",default="outputs/v45-delayed-language-grounding/design-audit.json"); p.add_argument("--output",default="configs/v45-design-lock.json"); a=p.parse_args(); config_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.config,a.audit,a.output))
 if output.exists(): raise RuntimeError("V45 design already frozen")
 audit=json.loads(audit_path.read_text());
 if not audit["passed"] or audit["config_sha256"]!=file_sha256(config_path): raise RuntimeError("V45 design audit failed")
 c=json.loads(config_path.read_text()); outcome=PROJECT_ROOT/c["sourceV44OutcomeLock"]; seal=PROJECT_ROOT/c["sourceV44CorpusSeal"]; plan=PROJECT_ROOT/"docs/v45-delayed-language-grounding-plan.md"; lock={"schema_version":45,"experiment":"v45_design_lock","config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"config_payload":c,"preregistration":str(plan.relative_to(PROJECT_ROOT)),"preregistration_sha256":file_sha256(plan),"design_audit":str(audit_path.relative_to(PROJECT_ROOT)),"design_audit_sha256":file_sha256(audit_path),"source_v44_outcome_lock":str(outcome.relative_to(PROJECT_ROOT)),"source_v44_outcome_lock_sha256":file_sha256(outcome),"source_v44_corpus_seal":str(seal.relative_to(PROJECT_ROOT)),"source_v44_corpus_seal_sha256":file_sha256(seal),"authorization":{"write_and_audit_implementation":True,"construct_paired_language_corpus":False,"run_paired_development":False,"preregister_stochastic_foundation":False,"model_access":False}}; lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
