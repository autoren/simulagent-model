#!/usr/bin/env python3
"""Construct the compact sealed V47 sampled-transition population."""
from __future__ import annotations
import argparse,json
from collections import Counter
from itertools import product
from typing import Any,Sequence
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json,sha256_text
from v42_stateful import ONTOLOGY,action_bindings,deterministic_world,entities,epistemic_rows
from v46_stochastic import ACTIONS
from v47_sampling import execute_joint_distribution,joint_map,mechanic_registry,sample_trajectory,trial_seed

def make_actions(entity_rows,pattern,token):
 bindings=action_bindings(entity_rows); start=int(sha256_text(f"binding|{token}")[:8],16)%len(bindings); rows=[]
 for index,action in enumerate(pattern): rows.append({"id":"wait","binding":{}} if action=="wait" else {"id":action,"binding":dict(bindings[(start+index)%len(bindings)])})
 return rows

def structural_key(entity_rows,state,actions): return sha256_text(canonical_json({"entities":entity_rows,"initial_state":state,"actions":actions}))

def intervention(index,seed,prefix):
 length=2+(index%3); patterns=[p for p in product(ACTIONS,repeat=length) if ("pulse" in p or "route" in p)]; pattern=patterns[(index//3)%len(patterns)]; count=2+(index%3); es=entities(count); token=f"v47-{prefix}|{seed}|{index}"; world=deterministic_world(es,token); actions=make_actions(es,pattern,token); state=epistemic_rows(world)
 return {"id":f"{prefix}_{sha256_text(token)[:16]}","entities":es,"initial_world":world,"initial_state":state,"actions":actions,"structural_key":structural_key(es,state,actions),"sequence_length":length,"entity_count":count}

def exact(program,case): return execute_joint_distribution(program,case["entities"],case["initial_world"],case["actions"])

def informative_support_cases(target,registry,pool,signatures,count):
 ranked=[]
 for case in pool:
  target_distribution=signatures[(target["id"],case["id"])]
  if len(target_distribution)<2: continue
  target_key=canonical_json(target_distribution); matches=sum(canonical_json(signatures[(m["id"],case["id"])])==target_key for m in registry)
  ranked.append((matches,sha256_text(f"{target['id']}|{case['id']}"),case))
 ranked.sort(key=lambda x:(x[0],x[1])); chosen=[]; seen=set()
 for _,_,case in ranked:
  action_shape=tuple(x["id"] for x in case["actions"])
  signature=(case["sequence_length"],case["entity_count"],action_shape)
  if signature in seen and len(chosen)<6: continue
  chosen.append(case); seen.add(signature)
  if len(chosen)==count: return chosen
 raise RuntimeError(f"V47 lacks informative supports for {target['id']}")

def catalog(distribution):
 rows={}
 for row in distribution:
  outcome_id=f"outcome_{sha256_text(canonical_json(row['trajectory']))[:16]}"; rows[outcome_id]=row["trajectory"]
 return rows

def sampled_ids(mechanic,case,trial_count,sampling_seed,outcome_catalog):
 reverse={canonical_json(value):key for key,value in outcome_catalog.items()}; result=[]
 for trial in range(trial_count):
  trajectory=sample_trajectory(mechanic["program"],case["entities"],case["initial_world"],case["actions"],trial_seed(sampling_seed,mechanic["id"],case["id"],trial)); result.append(reverse[canonical_json(trajectory)])
 return result

def support_row(mechanic,case,trials,sampling_seed):
 dist=exact(mechanic["program"],case); outcomes=catalog(dist)
 return {"id":case["id"],"entities":case["entities"],"initial_state":case["initial_state"],"actions":case["actions"],"structural_key":case["structural_key"],"outcome_catalog":outcomes,"realized_outcome_ids":sampled_ids(mechanic,case,trials,sampling_seed,outcomes)}

def query_case(mechanic,ordinal,seed,forbidden):
 for attempt in range(4096):
  case=intervention(10000+ordinal+attempt*24,seed,f"query-{mechanic['id']}-{ordinal}")
  stochastic_rule=next(rule for rule in mechanic["program"]["rules"] if rule["stochastic_immediate"] or rule["stochastic_delayed"]); trigger=stochastic_rule["action"]; branch=(stochastic_rule["stochastic_immediate"] or stochastic_rule["stochastic_delayed"])[0]; delay=branch.get("delay",0); other="route" if trigger=="pulse" else "pulse"; length=case["sequence_length"]
  if delay and length>delay: pattern=[trigger]+["wait"]*delay+[other]*(length-delay-1)
  elif delay: pattern=[trigger]+[other]*(length-1)
  else: pattern=[trigger]+[other]*(length-1)
  case["actions"]=make_actions(case["entities"],pattern,case["id"]); case["structural_key"]=structural_key(case["entities"],case["initial_state"],case["actions"])
  if case["structural_key"] in forbidden: continue
  dist=exact(mechanic["program"],case)
  # When timing permits, require a genuinely stochastic prediction.
  delayed=mechanic["timing"]=="delayed"; required=not delayed or length>delay
  if required and len(dist)<2: continue
  forbidden.add(case["structural_key"]); return case,dist
 raise RuntimeError(f"V47 lacks query {mechanic['id']}/{ordinal}")

def build_population(config):
 registry=mechanic_registry(); p=config["population"]; generator_seed=p["generatorSeed"]; sampling_seed=p["samplingSeed"]; max_trials=max(p["nestedTrialsPerIntervention"]); rows=[]; pool=[intervention(i,generator_seed,"support") for i in range(768)]; signatures={(m["id"],case["id"]):exact(m["program"],case) for m in registry for case in pool}
 for mechanic in registry:
  cases=informative_support_cases(mechanic,registry,pool,signatures,p["supportInterventionsPerMechanic"]); supports=[support_row(mechanic,case,max_trials,sampling_seed) for case in cases]; forbidden={x["structural_key"] for x in supports}; queries=[]; oracle=[]
  for ordinal in range(p["queryInterventionsPerMechanic"]):
   case,dist=query_case(mechanic,ordinal,generator_seed,forbidden); outcomes=catalog(dist); query_id=case["id"]
   queries.append({k:v for k,v in case.items() if k not in ("initial_world",)})
   oracle.append({"id":query_id,"true_joint_distribution":dist,"outcome_catalog":outcomes,"heldout_outcome_ids":sampled_ids(mechanic,case,p["heldoutTrialsPerQuery"],sampling_seed,outcomes)})
  split="development_fit" if mechanic["ordinal"]<6 else "development_evaluation"
  rows.append({"id":mechanic["id"],"schema_version":47,"split":split,"construction_family":mechanic["family"],"agent_input":{"task":"infer_a_posterior_over_stochastic_transition_programs_from_realized_trials","ontology":ONTOLOGY,"probability_vocabulary":["1/4","1/2","3/4"],"nested_trial_budgets":p["nestedTrialsPerIntervention"],"support_interventions":supports,"queries":queries},"target":{"program":mechanic["program"],"program_key":mechanic["key"]},"oracle_queries":oracle,"oracle_metadata":{"family_ordinal":mechanic["ordinal"],"probability":mechanic["probability"],"timing":mechanic["timing"],"support_interventions":len(supports),"support_trials":len(supports)*max_trials,"queries":len(queries),"heldout_trials":len(queries)*p["heldoutTrialsPerQuery"]}})
 if len(rows)!=48 or len({x["target"]["program_key"] for x in rows})!=48: raise RuntimeError("V47 population invalid")
 return sorted(rows,key=lambda x:x["id"])

def corpus_hash(rows:Sequence[dict[str,Any]]): return sha256_text("".join(canonical_json(x)+"\n" for x in sorted(rows,key=lambda x:x["id"])))

def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v47-implementation-lock.json"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); lock=json.loads(lock_path.read_text())
 if not lock["authorization"]["construct_sampled_population"]: raise RuntimeError("V47 implementation lock does not authorize construction")
 for path,expected in lock["implementation"].items():
  if file_sha256(PROJECT_ROOT/path)!=expected: raise RuntimeError(f"V47 implementation changed: {path}")
 output=PROJECT_ROOT/"data/v47-sampled-transition-estimation"
 if output.exists(): raise RuntimeError("V47 population already exists")
 rows=build_population(lock["config_payload"])
 if corpus_hash(rows)!=lock["expected_corpus_sha256"]: raise RuntimeError("V47 corpus differs from lock")
 output.mkdir(parents=True); artifacts={}
 for split in ("development_fit","development_evaluation"):
  selected=[x for x in rows if x["split"]==split]; path=output/f"{split}.jsonl"; path.write_text("".join(canonical_json(x)+"\n" for x in selected)); artifacts[split]={"path":str(path.relative_to(PROJECT_ROOT)),"records":len(selected),"sha256":file_sha256(path)}
 counts={"mechanics":48,"support_interventions":sum(x["oracle_metadata"]["support_interventions"] for x in rows),"support_trials":sum(x["oracle_metadata"]["support_trials"] for x in rows),"queries":sum(x["oracle_metadata"]["queries"] for x in rows),"heldout_trials":sum(x["oracle_metadata"]["heldout_trials"] for x in rows),"families":dict(Counter(x["construction_family"] for x in rows)),"probabilities":dict(Counter(x["oracle_metadata"]["probability"] for x in rows)),"splits":dict(Counter(x["split"] for x in rows))}; manifest={"schema_version":47,"experiment":lock["config_payload"]["experiment"],"implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"artifacts":artifacts,"counts":counts,"data_access":{"sampled_development_runs":0,"support_realizations_constructed":counts["support_trials"],"heldout_realizations_constructed":counts["heldout_trials"],"model_forward_passes":0,"adapter_training_runs":0}}; (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=="__main__": main()
