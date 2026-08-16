#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v46_stochastic import build_population,corpus_hash
from v46_stochastic import mechanic_registry
def main():
 p=argparse.ArgumentParser(); p.add_argument("--design-lock",default="configs/v46-design-lock.json"); p.add_argument("--output",default="outputs/v46-oracle-stochastic-transitions/implementation-audit.json"); a=p.parse_args(); design_path=(PROJECT_ROOT/a.design_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); d=json.loads(design_path.read_text()); errors=[]
 if not d["authorization"]["write_oracle_implementation"]: errors.append("V46 design does not authorize implementation")
 source=PROJECT_ROOT/d["source_v45_outcome_lock"]
 if file_sha256(source)!=d["source_v45_outcome_lock_sha256"]: errors.append("V45 source outcome changed")
 registry=mechanic_registry(); rows=build_population(d["config_payload"]); counts={"mechanics":len(rows),"support_sequences":sum(len(x["agent_input"]["support_sequences"]) for x in rows),"query_sequences":sum(len(x["agent_input"]["queries"]) for x in rows),"probability_sensitive_queries":sum(x["oracle_metadata"]["probability_sensitive_queries"] for x in rows),"uniform_sensitive_queries":sum(x["oracle_metadata"]["uniform_sensitive_queries"] for x in rows),"timing_sensitive_queries":sum(x["oracle_metadata"]["timing_sensitive_queries"] for x in rows)}
 if counts["mechanics"]!=40 or counts["query_sequences"]!=960 or counts["support_sequences"]>800: errors.append("V46 dry-run population quotas are invalid")
 if counts["probability_sensitive_queries"]<40 or counts["timing_sensitive_queries"]<20: errors.append("V46 sensitivity contract failed")
 if len(registry)!=40 or len({x["key"] for x in registry})!=40: errors.append("V46 registry is not a 40-program population")
 overlaps=sum(bool({s["structural_key"] for s in row["agent_input"]["support_sequences"]}&{q["structural_key"] for q in row["agent_input"]["queries"]}) for row in rows)
 if overlaps: errors.append("V46 support/query structural overlap")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v46-implementation-lock.json","configs/v46-corpus-seal.json","data/v46-oracle-stochastic-transitions","outputs/v46-oracle-stochastic-transitions/development")): errors.append("V46 downstream artifact exists before implementation lock")
 dry={**counts,"expected_corpus_sha256":corpus_hash(rows),"family_counts":dict(Counter(x["construction_family"] for x in rows)),"split_counts":dict(Counter(x["split"] for x in rows)),"probability_counts":dict(Counter(x["oracle_metadata"]["probability"] for x in rows)),"support_query_structural_overlap":overlaps,"sampled_realizations":0,"oracle_development_predictions":0}; audit={"schema_version":46,"experiment":"v46_implementation_audit","passed":not errors,"decision":"authorize_v46_implementation_lock" if not errors else "repair_v46_implementation","errors":errors,"design_lock":str(design_path.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design_path),"dry_run":dry,"data_access":{"oracle_development_runs":0,"sampled_realizations":0,"model_forward_passes":0,"adapter_training_runs":0}}
 output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
