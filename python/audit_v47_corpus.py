#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v47_sampled import corpus_hash
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v47-implementation-lock.json"); p.add_argument("--output",default="outputs/v47-sampled-transition-estimation/corpus-audit.json"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); lock=json.loads(lock_path.read_text()); base=PROJECT_ROOT/"data/v47-sampled-transition-estimation"; rows=[]; artifacts={}; errors=[]
 for split in ("development_fit","development_evaluation"):
  path=base/f"{split}.jsonl"; selected=read(path); rows+=selected; artifacts[split]={"path":str(path.relative_to(PROJECT_ROOT)),"records":len(selected),"sha256":file_sha256(path)}
  if len(selected)!=24: errors.append(f"V47 {split} count")
 counts={"mechanics":len(rows),"support_interventions":sum(x["oracle_metadata"]["support_interventions"] for x in rows),"support_trials":sum(x["oracle_metadata"]["support_trials"] for x in rows),"queries":sum(x["oracle_metadata"]["queries"] for x in rows),"heldout_trials":sum(x["oracle_metadata"]["heldout_trials"] for x in rows)}
 if counts!=lock["expected_counts"] or corpus_hash(rows)!=lock["expected_corpus_sha256"]: errors.append("V47 corpus differs from lock")
 if any("heldout_outcome_ids" in q or "true_joint_distribution" in q for x in rows for q in x["agent_input"]["queries"]): errors.append("Query outcomes exposed")
 if any(len(s["realized_outcome_ids"])!=128 or not set(s["realized_outcome_ids"])<=set(s["outcome_catalog"]) for x in rows for s in x["agent_input"]["support_interventions"]): errors.append("Support realization invalid")
 audit={"schema_version":47,"experiment":"v47_corpus_audit","passed":not errors,"decision":"authorize_v47_corpus_seal" if not errors else "reject_v47_corpus","errors":errors,"implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"artifacts":artifacts,"structural_checks":counts,"data_access":{"sampled_development_runs":0,"model_forward_passes":0,"adapter_training_runs":0}}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
