#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v47-implementation-lock.json"); p.add_argument("--audit",default="outputs/v47-sampled-transition-estimation/corpus-audit.json"); p.add_argument("--manifest",default="data/v47-sampled-transition-estimation/manifest.json"); p.add_argument("--output",default="configs/v47-corpus-seal.json"); a=p.parse_args(); lock,audit_path,manifest,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.implementation_lock,a.audit,a.manifest,a.output)); audit=json.loads(audit_path.read_text())
 if output.exists() or not audit["passed"] or audit["implementation_lock_sha256"]!=file_sha256(lock): raise RuntimeError("Cannot seal V47")
 seal={"schema_version":47,"experiment":"v47_corpus_seal","implementation_lock":str(lock.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock),"corpus_audit":str(audit_path.relative_to(PROJECT_ROOT)),"corpus_audit_sha256":file_sha256(audit_path),"manifest":str(manifest.relative_to(PROJECT_ROOT)),"manifest_sha256":file_sha256(manifest),"corpora":audit["artifacts"],"authorization":{"sampled_development_runs":1,"model_forward_passes":0,"adapter_training_runs":0,"final_evaluations":0}}; seal["lock_payload_sha256"]=hashlib.sha256(json.dumps(seal,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(seal,indent=2,sort_keys=True)+"\n"); print(json.dumps(seal,indent=2,sort_keys=True))
if __name__=="__main__": main()
