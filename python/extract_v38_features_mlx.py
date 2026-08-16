#!/usr/bin/env python3
"""Extract exactly 800 focus-candidate and 480 operation views for V38."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
import mlx.core as mx
import numpy as np
from mlx_lm import load
from evaluate_v30_signed_fact_language_mlx import dequantized_label_rows
from extract_v10_features_mlx import chat_prompt
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v34_operation import operation_prompt
from v38_focus_parser import candidate_prompt,deterministic_focus_index,extract_literal_candidates
def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--seal",default="configs/v38-corpus-seal.json"); p.add_argument("--output-dir",default="outputs/v38-ontology-anchored-focus-parser/features"); p.add_argument("--progress-every",type=int,default=10); a=p.parse_args(); seal_path=(PROJECT_ROOT/a.seal).resolve(); output=(PROJECT_ROOT/a.output_dir).resolve(); attempt=output.parent/"feature-attempt.json"
 if output.exists() or attempt.exists(): raise RuntimeError("V38 extraction already attempted")
 seal=json.loads(seal_path.read_text()); impl_path=PROJECT_ROOT/seal["implementation_lock"]; impl=json.loads(impl_path.read_text())
 for path,expected in impl["implementation"].items():
  if file_sha256(PROJECT_ROOT/path)!=expected: raise RuntimeError(f"V38 implementation changed: {path}")
 rows=[]
 for name in ("ontology_focus_fit","ontology_focus_validation"):
  meta=seal["corpora"][name]; path=PROJECT_ROOT/meta["path"]
  if file_sha256(path)!=meta["sha256"]: raise RuntimeError(f"V38 {name} changed")
  rows.extend(read(path))
 rows=sorted(rows,key=lambda r:r["id"]); candidate_total=sum(r["oracle_metadata"]["grounded_literal_candidates"] for r in rows)
 if candidate_total+len(rows)!=1280: raise RuntimeError("V38 forward budget mismatch")
 attempt.parent.mkdir(parents=True,exist_ok=True); attempt.write_text(json.dumps({"schema_version":38,"status":"started_before_model_load","planned_backbone_forward_passes":1280,"corpus_seal_sha256":file_sha256(seal_path)},indent=2,sort_keys=True)+"\n")
 spec=impl["v34_config_payload"]["model"]; model,tokenizer,model_config=load(spec["model"],revision=spec["revision"],return_config=True); model.eval(); tc=model_config["text_config"]
 if tc["num_hidden_layers"]!=spec["totalLayers"] or tc["hidden_size"]!=spec["hiddenSize"]: raise RuntimeError("V38 model differs from lock")
 labels=["Yes","No"]; encoded=[tokenizer.encode(x,add_special_tokens=False) for x in labels]
 if any(len(x)!=1 for x in encoded): raise RuntimeError("V38 Yes/No not single tokens")
 label_rows=dequantized_label_rows(model,[x[0] for x in encoded]); mx.eval(label_rows); yes,no=0,1; max_candidates=2
 values={"candidate_hidden":[],"candidate_margin":[],"candidate_mask":[],"deterministic_index":[],"operation_hidden":[]}; lengths=[]; hashes=[]; forwards=0; started=time.perf_counter()
 def run(content,system):
  nonlocal forwards
  prompt=chat_prompt(content,system,tokenizer); tokens=tokenizer.encode(prompt)
  if len(tokens)>spec["maxSequenceLength"]: raise RuntimeError("V38 prompt truncation")
  hidden=model.language_model.model(mx.array([tokens]))[0,-1].astype(mx.float32); mx.eval(hidden); forwards+=1; lengths.append(len(tokens)); hashes.append(hashlib.sha256(content.encode()).hexdigest()); return hidden
 for index,row in enumerate(rows,1):
  candidates=extract_literal_candidates(row); hiddens=[]; margins=[]
  for candidate in candidates:
   hidden=run(candidate_prompt(row,candidate),"Identify which ontology-grounded literal span is the proposition at issue. Ignore opposites, comparisons, and asides. Answer only Yes or No."); logits=hidden@label_rows.T; mx.eval(logits); hiddens.append(np.asarray(hidden,dtype=np.float32)); margins.append(float(logits[yes].item()-logits[no].item()))
  mask=[True]*len(candidates)
  while len(hiddens)<max_candidates: hiddens.append(np.zeros(spec["hiddenSize"],dtype=np.float32)); margins.append(float("-inf")); mask.append(False)
  operation=run(operation_prompt(row,impl["v34_config_payload"]),spec["systemPrompt"])
  values["candidate_hidden"].append(np.stack(hiddens)); values["candidate_margin"].append(np.asarray(margins,dtype=np.float32)); values["candidate_mask"].append(mask); values["deterministic_index"].append(deterministic_focus_index(row,candidates)); values["operation_hidden"].append(np.asarray(operation,dtype=np.float32))
  if a.progress_every and (index%a.progress_every==0 or index==len(rows)): print(f"v38 focus features: {index}/{len(rows)} records ({forwards}/1280 forwards)",file=sys.stderr,flush=True)
  mx.clear_cache()
 if forwards!=1280: raise RuntimeError("V38 did not execute exact forward budget")
 output.mkdir(parents=True,exist_ok=False); artifact=output/"features.npz"; np.savez_compressed(artifact,record_ids=np.asarray([r["id"] for r in rows]),splits=np.asarray([r["split"] for r in rows]),candidate_hidden=np.stack(values["candidate_hidden"]),candidate_margin=np.stack(values["candidate_margin"]),candidate_mask=np.asarray(values["candidate_mask"],dtype=bool),deterministic_index=np.asarray(values["deterministic_index"],dtype=np.int64),operation_hidden=np.stack(values["operation_hidden"]))
 metadata={"schema_version":38,"experiment":"v38_focus_features","corpus_seal":str(seal_path.relative_to(PROJECT_ROOT)),"corpus_seal_sha256":file_sha256(seal_path),"records":len(rows),"candidate_views":candidate_total,"operation_views":len(rows),"backbone_forward_passes":forwards,"minimum_prompt_tokens":min(lengths),"maximum_prompt_tokens":max(lengths),"truncated_prompts":0,"prompt_payload_sha256":hashlib.sha256("".join(hashes).encode()).hexdigest(),"feature_artifact":str(artifact.relative_to(PROJECT_ROOT)),"feature_artifact_sha256":file_sha256(artifact),"runtime_seconds":time.perf_counter()-started,"data_access":{"target_fields_used_in_prompts":0,"fit_runs":0,"selection_runs":0,"validation_evaluations":0,"v32_evaluation_records_read":0,"v28_runs":0}}
 meta_path=output/"metadata.json"; meta_path.write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n"); state=json.loads(attempt.read_text()); state.update({"status":"completed","metadata_sha256":file_sha256(meta_path)}); attempt.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n"); print(json.dumps(metadata,indent=2,sort_keys=True))
if __name__=="__main__": main()
