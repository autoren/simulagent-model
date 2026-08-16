#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json
from generate_v48_composition import build_population,corpus_hash,counts
from evaluate_v48_composition import compile_record,mean
from v46_stochastic import mechanic_registry as v46_registry
from v47_sampling import mechanic_registry as v47_registry
from v48_composition import mechanic_registry
def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--design-lock",default="configs/v48-design-lock.json"); p.add_argument("--reporting-lock",default="configs/v48-reporting-supplement-lock.json"); p.add_argument("--output",default="outputs/v48-stochastic-language-composition/implementation-audit.json"); a=p.parse_args(); design=(PROJECT_ROOT/a.design_lock).resolve(); reporting=(PROJECT_ROOT/a.reporting_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); d=json.loads(design.read_text()); errors=[]
 if not d["authorization"]["write_composition_implementation"]: errors.append("V48 implementation unauthorized")
 source=PROJECT_ROOT/d["source_v47_outcome_lock"]
 if file_sha256(source)!=d["source_v47_outcome_lock_sha256"]: errors.append("V47 outcome changed")
 reporting_payload=json.loads(reporting.read_text())
 if not reporting_payload["non_gating"] or reporting_payload["source_design_lock_sha256"]!=file_sha256(design): errors.append("V48 reporting supplement invalid")
 registry=mechanic_registry(); prior={x["key"] for x in v46_registry()}|{x["key"] for x in v47_registry()}; rows=build_population(d["config_payload"]); population=counts(rows)
 if {x["key"] for x in registry}&prior or len(registry)!=48: errors.append("V48 programs are not fresh")
 expected={"mechanics":48,"support_interventions":576,"support_trials":18432,"queries":1152,"heldout_trials":73728,"safety_challenges":432}
 if any(population[k]!=v for k,v in expected.items()): errors.append("V48 population quotas differ")
 interfaces=[]
 for row in rows: interfaces.append(compile_record(row)[1])
 interface_rates={key:mean(x for metric in interfaces for x in metric[key]) for key in ("clauses","graphs","commands","sequences","catalogs","safety","alignment")}
 if any(value!=1.0 for value in interface_rates.values()): errors.append("V48 dry language or alignment audit failed")
 v47_keys=set()
 for path in (PROJECT_ROOT/"data/v47-sampled-transition-estimation").glob("development_*.jsonl"):
  for row in read(path): v47_keys|={x["structural_key"] for x in row["agent_input"]["support_interventions"]}; v47_keys|={x["structural_key"] for x in row["agent_input"]["queries"]}
 v48_keys={x["source_structural_key"] for row in rows for x in row["reference"]["support_interventions"]}|{x["source_structural_key"] for row in rows for x in row["reference"]["queries"]}
 case_overlap=len(v47_keys&v48_keys)
 if case_overlap: errors.append("V48 reuses a V47 structural case")
 raw_exposure=sum("unit_0" in canonical_json(row["agent_input"]) for row in rows)
 if raw_exposure: errors.append("V48 exposes canonical entity IDs")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v48-implementation-lock.json","configs/v48-corpus-seal.json","data/v48-stochastic-language-composition","outputs/v48-stochastic-language-composition/development")): errors.append("V48 downstream artifact exists")
 audit={"schema_version":48,"experiment":"v48_implementation_audit","passed":not errors,"decision":"authorize_v48_implementation_lock" if not errors else "repair_v48_implementation","errors":errors,"design_lock":str(design.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design),"reporting_supplement_lock":str(reporting.relative_to(PROJECT_ROOT)),"reporting_supplement_lock_sha256":file_sha256(reporting),"dry_run":{**population,"expected_corpus_sha256":corpus_hash(rows),"interface_rates":interface_rates,"program_overlap_with_v46_v47":0,"case_overlap_with_v47":case_overlap,"raw_canonical_entity_exposure":raw_exposure,"development_predictions":0},"data_access":{"development_runs":0,"model_forward_passes":0,"adapter_training_runs":0}}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
