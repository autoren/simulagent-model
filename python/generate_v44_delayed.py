#!/usr/bin/env python3
"""Construct the V44 oracle deterministic delayed-effect population."""
from __future__ import annotations
import argparse,json
from collections import Counter
from typing import Any,Sequence
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json,sha256_text
from v42_stateful import ONTOLOGY,action_bindings,atom_universe,deterministic_world,entities,epistemic_rows,world_signature
from v44_delayed import ACTIONS,execute_partial,execute_sequence,mechanic_registry

def complete_rows(world): return [{"atom":atom,"value":value} for atom,value in sorted(world.items())]

def make_actions(entity_rows,pattern,token):
 bindings=action_bindings(entity_rows); start=int(sha256_text(f"binding|{token}")[:8],16)%len(bindings); rows=[]
 for index,action in enumerate(pattern):
  rows.append({"id":"wait","binding":{}} if action=="wait" else {"id":action,"binding":dict(bindings[(start+index)%len(bindings)])})
 return rows

def structural_key(entity_rows,state,actions): return sha256_text(canonical_json({"entities":list(entity_rows),"initial_state":list(state),"actions":list(actions)}))

def support_case(index,seed):
 count=2+(index%2); entity_rows=entities(count); token=f"v44-support|{seed}|{index}"; world=deterministic_world(entity_rows,token); length=3+(index%4); patterns=(("pulse","wait","route","wait","pulse","route"),("route","pulse","wait","route","wait","pulse"),("wait","pulse","route","wait","route","pulse")); pattern=list(patterns[index%len(patterns)][:length]); actions=make_actions(entity_rows,pattern,token); state=epistemic_rows(world)
 return {"id":f"support_{sha256_text(token)[:16]}","entities":entity_rows,"initial_world":world,"actions":actions,"structural_key":structural_key(entity_rows,state,actions)}

def trajectory_key(program,case,control="queued"): return tuple(world_signature(x) for x in execute_sequence(program,case["entities"],case["initial_world"],case["actions"],control))

def identifying_support(target,registry,seed,maximum,pool=None,signatures=None):
 pool=list(pool) if pool is not None else [support_case(i,seed) for i in range(768)]; signatures=signatures if signatures is not None else {(m["id"],c["id"]):trajectory_key(m["program"],c) for m in registry for c in pool}; survivors=list(registry); selected=[]; used=set()
 while len(survivors)>1:
  choices=[]
  for case in pool:
   if case["id"] in used: continue
   signature=signatures[(target["id"],case["id"])]; matching=[m for m in survivors if signatures[(m["id"],case["id"])]==signature]
   if len(matching)<len(survivors): choices.append((len(matching),sha256_text(f"{target['id']}|{case['id']}"),case,matching))
  if not choices: raise RuntimeError(f"V44 target not identifiable: {target['id']}")
  _,_,case,survivors=min(choices,key=lambda x:(x[0],x[1])); used.add(case["id"]); trajectory=execute_sequence(target["program"],case["entities"],case["initial_world"],case["actions"]); selected.append({"id":case["id"],"entities":case["entities"],"initial_state":epistemic_rows(case["initial_world"]),"actions":case["actions"],"observed_step_states":[complete_rows(x) for x in trajectory],"structural_key":case["structural_key"]})
  if len(selected)>maximum: raise RuntimeError(f"V44 support budget exceeded: {target['id']}")
 if survivors[0]["id"]!=target["id"]: raise RuntimeError("V44 support selected wrong mechanic")
 return selected

def query_pair(mechanic,pair_index,seed,forbidden):
 length=3+(pair_index%4); count=2+(pair_index%4); entity_rows=entities(count); delay=mechanic["delay"]; trigger=mechanic["trigger_action"]; other="route" if trigger=="pulse" else "pulse"; filler=[other]*(length-delay-1); forward_pattern=filler+[trigger]+["wait"]*delay; reverse_pattern=["wait"]*delay+filler+[trigger]
 for attempt in range(4096):
  token=f"v44-query|{seed}|{mechanic['id']}|{pair_index}|{attempt}"; world=deterministic_world(entity_rows,token); binding=action_bindings(entity_rows)[int(sha256_text(f"pair-binding|{token}")[:8],16)%len(action_bindings(entity_rows))]
  def actions(pattern): return [{"id":"wait","binding":{}} if action=="wait" else {"id":action,"binding":dict(binding)} for action in pattern]
  forward_actions=actions(forward_pattern); reverse_actions=actions(reverse_pattern); unknown=[]
  if pair_index%3==2: unknown=[atom_universe(entity_rows)[int(sha256_text(f"unknown|{token}")[:8],16)%len(atom_universe(entity_rows))]]
  initial=epistemic_rows(world,unknown); forward_key=structural_key(entity_rows,initial,forward_actions); reverse_key=structural_key(entity_rows,initial,reverse_actions)
  if forward_key in forbidden or reverse_key in forbidden or forward_key==reverse_key: continue
  target_f=execute_partial([mechanic["program"]],entity_rows,initial,forward_actions); target_r=execute_partial([mechanic["program"]],entity_rows,initial,reverse_actions)
  if target_f["possible_final_observations"]==target_r["possible_final_observations"]: continue
  collapsed=(execute_partial([mechanic["program"]],entity_rows,initial,forward_actions,"collapsed_delay"),execute_partial([mechanic["program"]],entity_rows,initial,reverse_actions,"collapsed_delay")); end_flush=(execute_partial([mechanic["program"]],entity_rows,initial,forward_actions,"end_flush"),execute_partial([mechanic["program"]],entity_rows,initial,reverse_actions,"end_flush"))
  if all(pred["possible_final_observations"]==target["possible_final_observations"] for pred,target in zip(collapsed,(target_f,target_r))) or all(pred["possible_final_observations"]==target["possible_final_observations"] for pred,target in zip(end_flush,(target_f,target_r))): continue
  group=f"wait_{sha256_text(f'{mechanic['id']}|{pair_index}')[:16]}"
  def row(role,acts,target,key): return {"id":f"query_{sha256_text(f'{token}|{role}')[:16]}","entities":entity_rows,"initial_state":initial,"actions":acts,"structural_key":key,"sequence_length":length,"entity_count":count,"partial_initial_state":bool(unknown),"wait_counterfactual_group":group,"wait_counterfactual_role":role,"wait_placement_effect":True,"target":target}
  forbidden.update((forward_key,reverse_key)); return row("delivered",forward_actions,target_f,forward_key),row("pending",reverse_actions,target_r,reverse_key)
 raise RuntimeError(f"Could not construct V44 causal pair: {mechanic['id']}/{pair_index}")

def build_population(config):
 registry=mechanic_registry(); seed=config["population"]["generatorSeed"]; pool=[support_case(i,seed) for i in range(768)]; signatures={(m["id"],c["id"]):trajectory_key(m["program"],c) for m in registry for c in pool}; records=[]
 for mechanic in registry:
  support=identifying_support(mechanic,registry,seed,config["population"]["supportSequencesPerMechanicMaximum"],pool,signatures); forbidden={x["structural_key"] for x in support}; queries=[]
  for pair_index in range(config["population"]["querySequencesPerMechanic"]//2): queries.extend(query_pair(mechanic,pair_index,seed,forbidden))
  split="development_fit" if mechanic["ordinal"]<6 else "development_evaluation"; records.append({"id":mechanic["id"],"schema_version":44,"split":split,"construction_family":mechanic["family"],"agent_input":{"task":"infer_a_deterministic_delayed_effect_mechanic_and_predict_each_observed_trajectory","ontology":ONTOLOGY,"action_schemas":[{"id":"pulse","parameters":ONTOLOGY["action"]["parameters"]},{"id":"route","parameters":ONTOLOGY["action"]["parameters"]},{"id":"wait","parameters":[]}],"tick_semantics":config["tickSemantics"],"support_sequences":support,"queries":[{k:v for k,v in q.items() if k!="target"} for q in queries]},"target":{"program":mechanic["program"],"program_key":mechanic["key"]},"oracle_queries":[{"id":q["id"],"target":q["target"]} for q in queries],"oracle_metadata":{"family_ordinal":mechanic["ordinal"],"trigger_action":mechanic["trigger_action"],"delay":mechanic["delay"],"support_sequences":len(support),"query_sequences":len(queries),"wait_counterfactual_pairs":len(queries)//2}})
 if len(records)!=40 or len({r["target"]["program_key"] for r in records})!=40: raise RuntimeError("V44 population must contain 40 unique mechanics")
 return sorted(records,key=lambda x:x["id"])

def corpus_hash(rows:Sequence[dict[str,Any]]): return sha256_text("".join(canonical_json(row)+"\n" for row in sorted(rows,key=lambda x:x["id"])))

def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v44-implementation-lock.json"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); lock=json.loads(lock_path.read_text());
 if not lock["authorization"]["construct_development_population"]: raise RuntimeError("V44 implementation lock does not authorize construction")
 for path,expected in lock["implementation"].items():
  if file_sha256(PROJECT_ROOT/path)!=expected: raise RuntimeError(f"V44 implementation changed: {path}")
 output=PROJECT_ROOT/"data/v44-deterministic-delayed-effects"
 if output.exists(): raise RuntimeError("V44 population already exists")
 rows=build_population(lock["config_payload"])
 if corpus_hash(rows)!=lock["expected_corpus_sha256"]: raise RuntimeError("V44 corpus differs from implementation lock")
 output.mkdir(parents=True); artifacts={}
 for split in ("development_fit","development_evaluation"):
  selected=[x for x in rows if x["split"]==split]; path=output/f"{split}.jsonl"; path.write_text("".join(canonical_json(x)+"\n" for x in selected)); artifacts[split]={"path":str(path.relative_to(PROJECT_ROOT)),"records":len(selected),"sha256":file_sha256(path)}
 counts={"mechanics":len(rows),"support_sequences":sum(len(x["agent_input"]["support_sequences"]) for x in rows),"query_sequences":sum(len(x["agent_input"]["queries"]) for x in rows),"wait_counterfactual_pairs":sum(x["oracle_metadata"]["wait_counterfactual_pairs"] for x in rows),"families":dict(Counter(x["construction_family"] for x in rows)),"splits":dict(Counter(x["split"] for x in rows))}; manifest={"schema_version":44,"experiment":lock["config_payload"]["experiment"],"implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"artifacts":artifacts,"counts":counts,"data_access":{"oracle_development_runs":0,"model_forward_passes":0,"adapter_training_runs":0,"v43_records_read":0}}; (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=="__main__": main()
