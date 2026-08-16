#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
FILES=("python/v43r1_measurement.py","python/test_v43r1_measurement.py","python/evaluate_v43r1_repair.py","python/audit_v43r1_implementation.py","python/freeze_v43r1_implementation.py","python/audit_and_summarize_v43r1.py","python/freeze_v43r1_outcome.py","python/evaluate_v43_language.py","python/v43_language.py","python/v42_stateful.py","python/v39_compiler.py","python/v38_focus_parser.py","python/v22_relational.py","python/v22r2_grounding.py","python/v10_protocol.py")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--design-lock",default="configs/v43r1-design-lock.json"); p.add_argument("--audit",default="outputs/v43r1-graph-measurement-repair/implementation-audit.json"); p.add_argument("--output",default="configs/v43r1-implementation-lock.json"); a=p.parse_args(); design_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.design_lock,a.audit,a.output))
 if output.exists(): raise RuntimeError("V43r1 implementation already frozen")
 d=json.loads(design_path.read_text()); audit=json.loads(audit_path.read_text());
 if not audit["passed"] or audit["design_lock_sha256"]!=file_sha256(design_path): raise RuntimeError("V43r1 implementation audit failed")
 for path in FILES:
  if not (PROJECT_ROOT/path).is_file(): raise RuntimeError(f"V43r1 implementation incomplete: {path}")
 v43_seal=json.loads((PROJECT_ROOT/d["source_v43_corpus_seal"]).read_text()); v43_impl=json.loads((PROJECT_ROOT/v43_seal["implementation_lock"]).read_text()); lock={"schema_version":"43r1","experiment":"v43r1_implementation_lock","design_lock":str(design_path.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design_path),"implementation_audit":str(audit_path.relative_to(PROJECT_ROOT)),"implementation_audit_sha256":file_sha256(audit_path),"config_payload":d["config_payload"],"source_v43_outcome_lock":d["source_v43_outcome_lock"],"source_v43_outcome_lock_sha256":file_sha256(PROJECT_ROOT/d["source_v43_outcome_lock"]),"source_v43_corpus_seal":d["source_v43_corpus_seal"],"source_v43_corpus_seal_sha256":file_sha256(PROJECT_ROOT/d["source_v43_corpus_seal"]),"source_post_hoc_diagnostic":d["source_post_hoc_diagnostic"],"source_post_hoc_diagnostic_sha256":file_sha256(PROJECT_ROOT/d["source_post_hoc_diagnostic"]),"v43_registered_gates":v43_impl["config_payload"]["gates"],"implementation":{path:file_sha256(PROJECT_ROOT/path) for path in FILES},"authorization":{"repair_rescores":1,"preregister_delayed_effects":False,"model_access":False,"new_corpus":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
