#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--metadata",default="outputs/v38-ontology-anchored-focus-parser/features/metadata.json"); p.add_argument("--output",default="configs/v38-features-lock.json"); a=p.parse_args(); meta_path=(PROJECT_ROOT/a.metadata).resolve(); output=(PROJECT_ROOT/a.output).resolve()
 if output.exists(): raise RuntimeError("V38 features already frozen")
 m=json.loads(meta_path.read_text()); artifact=PROJECT_ROOT/m["feature_artifact"]
 if m["backbone_forward_passes"]!=1280 or m["truncated_prompts"]!=0 or file_sha256(artifact)!=m["feature_artifact_sha256"]: raise RuntimeError("V38 feature budget/hash failed")
 lock={"schema_version":38,"experiment":"v38_features_lock","corpus_seal":m["corpus_seal"],"corpus_seal_sha256":m["corpus_seal_sha256"],"feature_metadata":str(meta_path.relative_to(PROJECT_ROOT)),"feature_metadata_sha256":file_sha256(meta_path),"feature_artifact":m["feature_artifact"],"feature_artifact_sha256":m["feature_artifact_sha256"],"authorization":{"validation_evaluations":1,"fit_selection":"ontology_focus_fit_only","v32_evaluation":False,"v28":False}}
 lock["lock_payload_sha256"]=hashlib.sha256(json.dumps(lock,sort_keys=True,separators=(",",":")).encode()).hexdigest(); output.write_text(json.dumps(lock,indent=2,sort_keys=True)+"\n"); print(json.dumps(lock,indent=2,sort_keys=True))
if __name__=="__main__": main()
