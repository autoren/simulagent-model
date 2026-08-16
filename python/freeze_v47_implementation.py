#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
FILES=("python/v47_sampling.py","python/generate_v47_sampled.py","python/evaluate_v47_sampled.py","python/test_v47_sampling.py","python/audit_v47_implementation.py","python/freeze_v47_implementation.py","python/audit_v47_corpus.py","python/seal_v47_corpus.py","python/audit_and_summarize_v47.py","python/freeze_v47_outcome.py","python/v46_stochastic.py","python/v42_stateful.py","python/v22_relational.py","python/v22r2_grounding.py","python/v10_protocol.py")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--design-lock",default="configs/v47-design-lock.json"); p.add_argument("--audit",default="outputs/v47-sampled-transition-estimation/implementation-audit.json"); p.add_argument("--output",default="configs/v47-implementation-lock.json"); a=p.parse_args(); design,audit_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.design_lock,a.audit,a.output))
 if output.exists(): raise RuntimeError("V47 implementation frozen")
 d=json.loads(design.read_text()); audit=json.loads(audit_path.read_text())
 if not audit["passed"] or audit["design_lock_sha256"]!=file_sha256(design): raise RuntimeError("V47 audit failed")
 for path in FILES:
  if not (PROJECT_ROOT/path).is_file(): raise RuntimeError(f"Missing {path}")
 keys=("mechanics","support_interventions","support_trials","queries","heldout_trials"); lock={"schema_version":47,"experiment":"v47_implementation_lock","design_lock":str(design.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design),"implementation_audit":str(audit_path.relative_to(PROJECT_ROOT)),"implementation_audit_sha256":file_sha256(audit_path),"config_payload":d["config_payload"],"source_v46_outcome_lock":d["source_v46_outcome_lock"],"source_v46_outcome_lock_sha256":file_sha256(PROJECT_ROOT/d["source_v46_outcome_lock"]),"expected_corpus_sha256":audit["dry_run"]["expected_corpus_sha256"],"expected_counts":{k:audit["dry_run"][k] for k in keys},"implementation":{path:file_sha256(PROJECT_ROOT/path) for path in FILES},"authorization":{"construct_sampled_population":True,"run_sampled_development":False,"model_access":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
