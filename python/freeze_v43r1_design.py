#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v43r1-graph-measurement-repair.json"); p.add_argument("--audit",default="outputs/v43r1-graph-measurement-repair/design-audit.json"); p.add_argument("--output",default="configs/v43r1-design-lock.json"); a=p.parse_args(); config_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.config,a.audit,a.output))
 if output.exists(): raise RuntimeError("V43r1 design already frozen")
 audit=json.loads(audit_path.read_text());
 if not audit["passed"] or audit["config_sha256"]!=file_sha256(config_path): raise RuntimeError("V43r1 design audit failed")
 c=json.loads(config_path.read_text()); plan=PROJECT_ROOT/"docs/v43r1-graph-measurement-repair-plan.md"; lock={"schema_version":"43r1","experiment":"v43r1_design_lock","config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"config_payload":c,"preregistration":str(plan.relative_to(PROJECT_ROOT)),"preregistration_sha256":file_sha256(plan),"design_audit":str(audit_path.relative_to(PROJECT_ROOT)),"design_audit_sha256":file_sha256(audit_path),"source_v43_outcome_lock":c["sourceV43OutcomeLock"],"source_v43_outcome_lock_sha256":file_sha256(PROJECT_ROOT/c["sourceV43OutcomeLock"]),"source_v43_corpus_seal":c["sourceV43CorpusSeal"],"source_v43_corpus_seal_sha256":file_sha256(PROJECT_ROOT/c["sourceV43CorpusSeal"]),"source_post_hoc_diagnostic":c["sourcePostHocDiagnostic"],"source_post_hoc_diagnostic_sha256":file_sha256(PROJECT_ROOT/c["sourcePostHocDiagnostic"]),"authorization":{"write_and_audit_repair_implementation":True,"run_repair_rescore":False,"preregister_delayed_effects":False,"model_access":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
