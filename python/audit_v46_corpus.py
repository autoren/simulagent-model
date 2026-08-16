#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter
from fractions import Fraction
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v46_stochastic import corpus_hash
def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def normal(trajectory): return all(sum((Fraction(x["mass"]["numerator"],x["mass"]["denominator"]) for x in step),Fraction(0))==1 for step in trajectory)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v46-implementation-lock.json"); p.add_argument("--output",default="outputs/v46-oracle-stochastic-transitions/corpus-audit.json"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); lock=json.loads(lock_path.read_text()); base=PROJECT_ROOT/"data/v46-oracle-stochastic-transitions"; errors=[]; artifacts={}; rows=[]
 for split,expected in (("development_fit",24),("development_evaluation",16)):
  path=base/f"{split}.jsonl"; selected=read(path); rows.extend(selected); artifacts[split]={"path":str(path.relative_to(PROJECT_ROOT)),"records":len(selected),"sha256":file_sha256(path)}
  if len(selected)!=expected: errors.append(f"V46 {split} count mismatch")
 if corpus_hash(rows)!=lock["expected_corpus_sha256"]: errors.append("V46 corpus hash mismatch")
 counts={"mechanics":len(rows),"support_sequences":sum(len(x["agent_input"]["support_sequences"]) for x in rows),"query_sequences":sum(len(x["agent_input"]["queries"]) for x in rows),"probability_sensitive_queries":sum(x["oracle_metadata"]["probability_sensitive_queries"] for x in rows),"uniform_sensitive_queries":sum(x["oracle_metadata"]["uniform_sensitive_queries"] for x in rows),"timing_sensitive_queries":sum(x["oracle_metadata"]["timing_sensitive_queries"] for x in rows)}
 if counts!=lock["expected_counts"]: errors.append("V46 corpus count mismatch")
 if any("target" in q for row in rows for q in row["agent_input"]["queries"]): errors.append("V46 query target exposed")
 trajectories=[s["observed_step_distributions"] for row in rows for s in row["agent_input"]["support_sequences"]]+[q["target"] for row in rows for q in row["oracle_queries"]]
 if not all(normal(t) for t in trajectories): errors.append("V46 probability mass does not normalize exactly")
 audit={"schema_version":46,"experiment":"v46_corpus_audit","passed":not errors,"decision":"authorize_v46_corpus_seal" if not errors else "reject_v46_corpus","errors":errors,"implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"artifacts":artifacts,"structural_checks":{**counts,"exact_mass_normalization":all(normal(t) for t in trajectories),"family_counts":dict(Counter(x["construction_family"] for x in rows)),"probability_counts":dict(Counter(x["oracle_metadata"]["probability"] for x in rows)),"sampled_realizations":0,"oracle_development_runs":0},"data_access":{"oracle_development_runs":0,"sampled_realizations":0,"model_forward_passes":0,"adapter_training_runs":0}}
 output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
