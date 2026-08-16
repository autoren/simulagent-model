#!/usr/bin/env python3
"""Construct paired language and symbolic V48 stochastic records."""
from __future__ import annotations
import argparse,json
from collections import Counter
from typing import Any,Sequence
from v10_protocol import file_sha256
from v22_relational import canonical_json,sha256_text
from v22r2_grounding import PROJECT_ROOT
from v43_language import operator_ontology,predicate_ontology,public_entities,render_state
from v45_language import action_ontology,render_action_sequence,safety_challenges
from v47_sampling import execute_joint_distribution,sample_trajectory
from generate_v47_sampled import intervention,informative_support_cases,query_case,catalog
from v48_composition import alias_distribution,alias_signature,alias_state,aliases_for,mechanic_registry,signature_rows

def v48_seed(seed,mechanic_id,case_id,trial): return int(sha256_text(f"v48|{seed}|{mechanic_id}|{case_id}|{trial}"),16)
def exact(program,case): return execute_joint_distribution(program,case["entities"],case["initial_world"],case["actions"])
def previous_structural_keys():
 keys=set(); base=PROJECT_ROOT/"data/v47-sampled-transition-estimation"
 for path in base.glob("development_*.jsonl"):
  for line in path.read_text().splitlines():
   if not line.strip(): continue
   row=json.loads(line); keys|={x["structural_key"] for x in row["agent_input"]["support_interventions"]}; keys|={x["structural_key"] for x in row["agent_input"]["queries"]}
 return keys
def sampled_ids(mechanic,case,count,seed,outcomes):
 reverse={canonical_json(value):key for key,value in outcomes.items()}; return [reverse[canonical_json(sample_trajectory(mechanic["program"],case["entities"],case["initial_world"],case["actions"],v48_seed(seed,mechanic["id"],case["id"],trial)))] for trial in range(count)]

def render_catalog(outcomes,aliases,predicate,operator_cues,token):
 public,reference,language_references={},{},{}
 for outcome_id,trajectory in sorted(outcomes.items()):
  public_steps=[]; reference_steps=[]; clause_steps=[]
  for index,signature in enumerate(trajectory):
   language,ref=render_state(signature_rows(signature),aliases,predicate,operator_cues,f"{token}|{outcome_id}|{index}"); public_steps.append(language); reference_steps.append(alias_signature(signature,aliases)); clause_steps.append(ref)
  public[outcome_id]=public_steps; reference[outcome_id]=reference_steps; language_references[outcome_id]=clause_steps
 return public,reference,language_references

def transform_mechanic(mechanic,support_cases,query_pairs,config):
 entity_ids={e["id"] for case in support_cases for e in case["entities"]}|{e["id"] for case,_ in query_pairs for e in case["entities"]}; aliases=aliases_for(mechanic["id"],entity_ids); predicate=predicate_ontology(f"v48|{mechanic['id']}"); operator,operator_cues=operator_ontology(f"v48|{mechanic['id']}"); action,action_cues=action_ontology(f"v48|{mechanic['id']}"); trials=config["population"]["realizedTrialsPerSupportIntervention"]; sampling_seed=config["population"]["samplingSeed"]; public_support=[]; symbolic_support=[]; references_support=[]; clause_count=0; command_count=0
 for case in support_cases:
  es=public_entities(case["entities"],aliases); initial,initial_ref=render_state(case["initial_state"],aliases,predicate,operator_cues,f"v48|{case['id']}|initial"); action_text,action_ref=render_action_sequence(case["actions"],aliases,action_cues,case["id"]); distribution=exact(mechanic["program"],case); outcomes=catalog(distribution); ids=sampled_ids(mechanic,case,trials,sampling_seed,outcomes); language_catalog,symbolic_catalog,outcome_references=render_catalog(outcomes,aliases,predicate,operator_cues,f"v48|{case['id']}|outcomes"); paired_key=sha256_text(f"v48-pair|{case['id']}")
  public_support.append({"id":case["id"],"entities":es,"initial_state_language":initial,"action_language":action_text,"outcome_catalog_language":language_catalog,"realized_outcome_ids":ids,"paired_structural_key":paired_key})
  symbolic={"id":case["id"],"entities":es,"initial_state":initial_ref["epistemic_state"],"actions":action_ref["actions"],"outcome_catalog":symbolic_catalog,"realized_outcome_ids":ids,"structural_key":paired_key}; symbolic_support.append(symbolic); references_support.append({"id":case["id"],"initial_state":initial_ref,"actions":action_ref,"outcome_catalog":symbolic_catalog,"outcome_state_references":outcome_references,"source_structural_key":case["structural_key"]})
  clause_count+=len(initial_ref["clauses"])+sum(len(signature_rows(sig)) for trajectory in outcomes.values() for sig in trajectory); command_count+=len(action_ref["actions"])
 public_queries=[]; symbolic_queries=[]; reference_queries=[]; oracle_queries=[]
 for case,distribution in query_pairs:
  es=public_entities(case["entities"],aliases); initial,initial_ref=render_state(case["initial_state"],aliases,predicate,operator_cues,f"v48|{case['id']}|initial"); action_text,action_ref=render_action_sequence(case["actions"],aliases,action_cues,case["id"]); paired_key=sha256_text(f"v48-pair|{case['id']}"); public_queries.append({"id":case["id"],"entities":es,"initial_state_language":initial,"action_language":action_text,"paired_structural_key":paired_key,"sequence_length":case["sequence_length"],"entity_count":case["entity_count"]}); symbolic_queries.append({"id":case["id"],"entities":es,"initial_state":initial_ref["epistemic_state"],"actions":action_ref["actions"],"structural_key":paired_key,"sequence_length":case["sequence_length"],"entity_count":case["entity_count"]}); reference_queries.append({"id":case["id"],"initial_state":initial_ref,"actions":action_ref,"source_structural_key":case["structural_key"]}); outcomes=catalog(distribution); ids=sampled_ids(mechanic,case,config["population"]["heldoutTrialsPerQuery"],sampling_seed,outcomes); oracle_queries.append({"id":case["id"],"true_joint_distribution":alias_distribution(distribution,aliases),"outcome_catalog":{key:[alias_signature(sig,aliases) for sig in trajectory] for key,trajectory in outcomes.items()},"heldout_outcome_ids":ids}); clause_count+=len(initial_ref["clauses"]); command_count+=len(action_ref["actions"])
 entity_catalog=[{"id":alias,"entity_type":"unit"} for alias in sorted(aliases.values())]; challenges=safety_challenges(entity_catalog[:2],predicate,operator,operator_cues,action,action_cues); split="development_fit" if mechanic["ordinal"]<6 else "development_evaluation"
 return {"id":mechanic["id"],"schema_version":48,"split":split,"construction_family":mechanic["family"],"agent_input":{"task":"compile_declared_stochastic_trajectory_records_then_infer_a_program_posterior","entity_catalog":entity_catalog,"predicate_ontology":predicate,"operator_ontology":operator,"action_ontology":action,"probability_vocabulary":["1/4","1/2","3/4"],"support_interventions":public_support,"queries":public_queries,"safety_challenges":challenges},"target":{"program":mechanic["program"],"program_key":mechanic["key"]},"oracle_queries":oracle_queries,"reference":{"entity_aliases":aliases,"support_interventions":references_support,"queries":reference_queries,"matched_symbolic":{"support_interventions":symbolic_support,"queries":symbolic_queries}},"oracle_metadata":{"family_ordinal":mechanic["ordinal"],"probability":mechanic["probability"],"timing":mechanic["timing"],"support_interventions":len(public_support),"support_trials":len(public_support)*trials,"queries":len(public_queries),"heldout_trials":len(public_queries)*config["population"]["heldoutTrialsPerQuery"],"state_clauses":clause_count,"action_commands":command_count,"safety_challenges":len(challenges)}}

def build_population(config):
 registry=mechanic_registry(); p=config["population"]; seed=p["generatorSeed"]; previous=previous_structural_keys(); pool=[case for i in range(1536) if (case:=intervention(i,seed,"v48-support"))["structural_key"] not in previous][:768]; signatures={(m["id"],case["id"]):exact(m["program"],case) for m in registry for case in pool}; rows=[]
 for mechanic in registry:
  supports=informative_support_cases(mechanic,registry,pool,signatures,p["supportInterventionsPerMechanic"]); forbidden=previous|{x["structural_key"] for x in supports}; queries=[query_case(mechanic,i,seed,forbidden) for i in range(p["queryInterventionsPerMechanic"])]; rows.append(transform_mechanic(mechanic,supports,queries,config))
 if len(rows)!=48 or len({x["target"]["program_key"] for x in rows})!=48: raise RuntimeError("V48 population invalid")
 return sorted(rows,key=lambda x:x["id"])

def corpus_hash(rows:Sequence[dict[str,Any]]): return sha256_text("".join(canonical_json(x)+"\n" for x in sorted(rows,key=lambda x:x["id"])))
def counts(rows): return {"mechanics":len(rows),"support_interventions":sum(x["oracle_metadata"]["support_interventions"] for x in rows),"support_trials":sum(x["oracle_metadata"]["support_trials"] for x in rows),"queries":sum(x["oracle_metadata"]["queries"] for x in rows),"heldout_trials":sum(x["oracle_metadata"]["heldout_trials"] for x in rows),"state_clauses":sum(x["oracle_metadata"]["state_clauses"] for x in rows),"action_commands":sum(x["oracle_metadata"]["action_commands"] for x in rows),"safety_challenges":sum(x["oracle_metadata"]["safety_challenges"] for x in rows),"families":dict(Counter(x["construction_family"] for x in rows)),"probabilities":dict(Counter(x["oracle_metadata"]["probability"] for x in rows)),"splits":dict(Counter(x["split"] for x in rows))}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v48-implementation-lock.json"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); lock=json.loads(lock_path.read_text())
 if not lock["authorization"]["construct_development_population"]: raise RuntimeError("V48 construction unauthorized")
 for path,expected in lock["implementation"].items():
  if file_sha256(PROJECT_ROOT/path)!=expected: raise RuntimeError(f"V48 implementation changed: {path}")
 output=PROJECT_ROOT/"data/v48-stochastic-language-composition"
 if output.exists(): raise RuntimeError("V48 population exists")
 rows=build_population(lock["config_payload"])
 if corpus_hash(rows)!=lock["expected_corpus_sha256"]: raise RuntimeError("V48 corpus differs from lock")
 output.mkdir(parents=True); artifacts={}
 for split in ("development_fit","development_evaluation"):
  selected=[x for x in rows if x["split"]==split]; path=output/f"{split}.jsonl"; path.write_text("".join(canonical_json(x)+"\n" for x in selected)); artifacts[split]={"path":str(path.relative_to(PROJECT_ROOT)),"records":len(selected),"sha256":file_sha256(path)}
 manifest={"schema_version":48,"experiment":lock["config_payload"]["experiment"],"implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"artifacts":artifacts,"counts":counts(rows),"data_access":{"development_runs":0,"model_forward_passes":0,"adapter_training_runs":0}}; (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); print(json.dumps(manifest,indent=2,sort_keys=True))
if __name__=="__main__": main()
