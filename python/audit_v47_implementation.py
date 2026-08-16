#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v47_sampled import build_population,corpus_hash
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry
def main():
 p=argparse.ArgumentParser(); p.add_argument("--design-lock",default="configs/v47-design-lock.json"); p.add_argument("--output",default="outputs/v47-sampled-transition-estimation/implementation-audit.json"); a=p.parse_args(); design_path=(PROJECT_ROOT/a.design_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); d=json.loads(design_path.read_text()); errors=[]
 if not d["authorization"]["write_estimator_implementation"]: errors.append("V47 design does not authorize implementation")
 source=PROJECT_ROOT/d["source_v46_outcome_lock"]
 if file_sha256(source)!=d["source_v46_outcome_lock_sha256"]: errors.append("V46 outcome changed")
 registry=mechanic_registry(); rows=build_population(d["config_payload"]); counts={"mechanics":len(rows),"support_interventions":sum(x["oracle_metadata"]["support_interventions"] for x in rows),"support_trials":sum(x["oracle_metadata"]["support_trials"] for x in rows),"queries":sum(x["oracle_metadata"]["queries"] for x in rows),"heldout_trials":sum(x["oracle_metadata"]["heldout_trials"] for x in rows)}
 if counts!={"mechanics":48,"support_interventions":576,"support_trials":73728,"queries":1152,"heldout_trials":73728}: errors.append("V47 dry-run quotas differ")
 if {x["key"] for x in registry}&{x["key"] for x in v46_registry()}: errors.append("V47 reuses a V46 program")
 if dict(Counter(x["probability"] for x in registry))!={"1/4":16,"1/2":16,"3/4":16}: errors.append("V47 probabilities are not balanced")
 overlaps=sum(bool({s["structural_key"] for s in x["agent_input"]["support_interventions"]}&{q["structural_key"] for q in x["agent_input"]["queries"]}) for x in rows)
 if overlaps: errors.append("V47 support/query overlap")
 if any("oracle" in canonical.lower() for x in rows for canonical in x["agent_input"]): errors.append("V47 agent input field exposes oracle data")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v47-implementation-lock.json","configs/v47-corpus-seal.json","data/v47-sampled-transition-estimation","outputs/v47-sampled-transition-estimation/development")): errors.append("V47 downstream artifact exists")
 audit={"schema_version":47,"experiment":"v47_implementation_audit","passed":not errors,"decision":"authorize_v47_implementation_lock" if not errors else "repair_v47_implementation","errors":errors,"design_lock":str(design_path.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design_path),"dry_run":{**counts,"expected_corpus_sha256":corpus_hash(rows),"family_counts":dict(Counter(x["construction_family"] for x in rows)),"probability_counts":dict(Counter(x["oracle_metadata"]["probability"] for x in rows)),"support_query_overlap":overlaps,"sampled_development_runs":0},"data_access":{"sampled_development_runs":0,"model_forward_passes":0,"adapter_training_runs":0}}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
