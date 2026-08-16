#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v44-implementation-lock.json"); p.add_argument("--audit",default="outputs/v44-deterministic-delayed-effects/corpus-audit.json"); p.add_argument("--manifest",default="data/v44-deterministic-delayed-effects/manifest.json"); p.add_argument("--output",default="configs/v44-corpus-seal.json"); a=p.parse_args(); lock_path,audit_path,manifest_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.implementation_lock,a.audit,a.manifest,a.output))
 if output.exists(): raise RuntimeError("V44 corpus already sealed")
 audit=json.loads(audit_path.read_text());
 if not audit["passed"] or audit["implementation_lock_sha256"]!=file_sha256(lock_path): raise RuntimeError("V44 corpus audit failed")
 seal={"schema_version":44,"experiment":"v44_corpus_seal","implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"corpus_audit":str(audit_path.relative_to(PROJECT_ROOT)),"corpus_audit_sha256":file_sha256(audit_path),"manifest":str(manifest_path.relative_to(PROJECT_ROOT)),"manifest_sha256":file_sha256(manifest_path),"corpora":audit["artifacts"],"authorization":{"oracle_development_runs":1,"model_forward_passes":0,"adapter_training_runs":0,"final_evaluations":0}}; seal["lock_payload_sha256"]=hashlib.sha256(json.dumps(seal,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(seal,indent=2,sort_keys=True)+"\n"); print(json.dumps(seal,indent=2,sort_keys=True))
if __name__=="__main__": main()
