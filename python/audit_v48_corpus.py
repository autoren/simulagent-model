#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v48_composition import corpus_hash,counts
def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v48-implementation-lock.json"); p.add_argument("--output",default="outputs/v48-stochastic-language-composition/corpus-audit.json"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); lock=json.loads(lock_path.read_text()); base=PROJECT_ROOT/"data/v48-stochastic-language-composition"; rows=[]; artifacts={}; errors=[]
 for split in ("development_fit","development_evaluation"):
  path=base/f"{split}.jsonl"; selected=read(path); rows+=selected; artifacts[split]={"path":str(path.relative_to(PROJECT_ROOT)),"records":len(selected),"sha256":file_sha256(path)}
  if len(selected)!=24: errors.append(f"V48 {split} count")
 actual=counts(rows)
 if any(actual[k]!=v for k,v in lock["expected_counts"].items()) or corpus_hash(rows)!=lock["expected_corpus_sha256"]: errors.append("V48 corpus differs from lock")
 if any("outcome" in key.lower() for row in rows for query in row["agent_input"]["queries"] for key in query): errors.append("V48 query exposes outcomes")
 if any("unit_0" in json.dumps(row["agent_input"]) for row in rows): errors.append("Canonical entity exposed")
 audit={"schema_version":48,"experiment":"v48_corpus_audit","passed":not errors,"decision":"authorize_v48_corpus_seal" if not errors else "reject_v48_corpus","errors":errors,"implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"artifacts":artifacts,"structural_checks":actual,"data_access":{"development_runs":0,"model_forward_passes":0,"adapter_training_runs":0}}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
