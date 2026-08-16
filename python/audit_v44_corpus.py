#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v44_delayed import corpus_hash
def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v44-implementation-lock.json"); p.add_argument("--output",default="outputs/v44-deterministic-delayed-effects/corpus-audit.json"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); lock=json.loads(lock_path.read_text()); base=PROJECT_ROOT/"data/v44-deterministic-delayed-effects"; errors=[]; artifacts={}; rows=[]
 for split,expected in (("development_fit",24),("development_evaluation",16)):
  path=base/f"{split}.jsonl"; selected=read(path); rows.extend(selected); artifacts[split]={"path":str(path.relative_to(PROJECT_ROOT)),"records":len(selected),"sha256":file_sha256(path)}
  if len(selected)!=expected: errors.append(f"V44 {split} count mismatch")
 if corpus_hash(rows)!=lock["expected_corpus_sha256"]: errors.append("V44 corpus hash mismatch")
 counts={"mechanics":len(rows),"support_sequences":sum(len(x["agent_input"]["support_sequences"]) for x in rows),"query_sequences":sum(len(x["agent_input"]["queries"]) for x in rows),"wait_counterfactual_pairs":sum(x["oracle_metadata"]["wait_counterfactual_pairs"] for x in rows)}
 if counts!=lock["expected_counts"]: errors.append("V44 corpus count mismatch")
 if any("target" in q for row in rows for q in row["agent_input"]["queries"]): errors.append("V44 query target exposed")
 audit={"schema_version":44,"experiment":"v44_corpus_audit","passed":not errors,"decision":"authorize_v44_corpus_seal" if not errors else "reject_v44_corpus","errors":errors,"implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"artifacts":artifacts,"structural_checks":{**counts,"family_counts":dict(Counter(x["construction_family"] for x in rows)),"oracle_development_runs":0},"data_access":{"oracle_development_runs":0,"model_forward_passes":0,"adapter_training_runs":0,"v43_records_read":0}}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
