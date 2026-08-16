#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
FILES=("python/v46_stochastic.py","python/generate_v46_stochastic.py","python/evaluate_v46_stochastic.py","python/test_v46_stochastic.py","python/audit_v46_implementation.py","python/freeze_v46_implementation.py","python/audit_v46_corpus.py","python/seal_v46_corpus.py","python/audit_and_summarize_v46.py","python/freeze_v46_outcome.py","python/v42_stateful.py","python/v22_relational.py","python/v22r2_grounding.py","python/v10_protocol.py")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--design-lock",default="configs/v46-design-lock.json"); p.add_argument("--audit",default="outputs/v46-oracle-stochastic-transitions/implementation-audit.json"); p.add_argument("--output",default="configs/v46-implementation-lock.json"); a=p.parse_args(); design_path,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.design_lock,a.audit,a.output))
 if output.exists(): raise RuntimeError("V46 implementation already frozen")
 d=json.loads(design_path.read_text()); audit=json.loads(audit_path.read_text())
 if not audit["passed"] or audit["design_lock_sha256"]!=file_sha256(design_path): raise RuntimeError("V46 implementation audit failed")
 for path in FILES:
  if not (PROJECT_ROOT/path).is_file(): raise RuntimeError(f"V46 implementation incomplete: {path}")
 count_keys=("mechanics","support_sequences","query_sequences","probability_sensitive_queries","uniform_sensitive_queries","timing_sensitive_queries")
 lock={"schema_version":46,"experiment":"v46_implementation_lock","design_lock":str(design_path.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design_path),"implementation_audit":str(audit_path.relative_to(PROJECT_ROOT)),"implementation_audit_sha256":file_sha256(audit_path),"config_payload":d["config_payload"],"source_v45_outcome_lock":d["source_v45_outcome_lock"],"source_v45_outcome_lock_sha256":file_sha256(PROJECT_ROOT/d["source_v45_outcome_lock"]),"expected_corpus_sha256":audit["dry_run"]["expected_corpus_sha256"],"expected_counts":{k:audit["dry_run"][k] for k in count_keys},"implementation":{path:file_sha256(PROJECT_ROOT/path) for path in FILES},"authorization":{"construct_development_population":True,"run_oracle_development":False,"sample_outcomes":False,"language_grounding":False,"model_access":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
