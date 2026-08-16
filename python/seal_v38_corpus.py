#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v38-implementation-lock.json"); p.add_argument("--audit",default="outputs/v38-ontology-anchored-focus-parser/corpus-audit.json"); p.add_argument("--manifest",default="data/v38-ontology-anchored-focus-parser/manifest.json"); p.add_argument("--output",default="configs/v38-corpus-seal.json"); a=p.parse_args(); lock_path,audit_path,manifest_path,output=tuple((PROJECT_ROOT/x).resolve() for x in (a.implementation_lock,a.audit,a.manifest,a.output))
 if output.exists(): raise RuntimeError("V38 corpus already sealed")
 audit=json.loads(audit_path.read_text());
 if not audit["passed"] or audit["implementation_lock_sha256"]!=file_sha256(lock_path): raise RuntimeError("V38 corpus audit did not pass")
 seal={"schema_version":38,"experiment":"v38_corpus_seal","implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"corpus_audit":str(audit_path.relative_to(PROJECT_ROOT)),"corpus_audit_sha256":file_sha256(audit_path),"manifest":str(manifest_path.relative_to(PROJECT_ROOT)),"manifest_sha256":file_sha256(manifest_path),"corpora":audit["artifacts"],"authorization":{"feature_extraction_attempts":1,"backbone_forward_passes":1280,"validation_evaluations":1,"v32_evaluation":False,"v28":False}}
 seal["lock_payload_sha256"]=hashlib.sha256(json.dumps(seal,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(seal,indent=2,sort_keys=True)+"\n"); print(json.dumps(seal,indent=2,sort_keys=True))
if __name__=="__main__": main()
