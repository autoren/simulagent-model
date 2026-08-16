#!/usr/bin/env python3
"""Evaluate paired declared-language and symbolic V48 stochastic inference."""
from __future__ import annotations
import argparse,json,math,time
from decimal import Decimal
from fractions import Fraction
import numpy as np
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json
from v43_language import compile_state
from v43r1_measurement import graph_equal
from v45_language import compile_action_sequence,evaluate_safety_challenge
from v47_sampling import execute_joint_distribution,joint_map,posterior,posterior_predictive
from v48_composition import mechanic_registry,probability_posterior

def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def mean(values): values=list(values); return sum(values)/len(values) if values else 0.0
def compiled_signature(compiled):
 if compiled.get("status")!="ok" or any(len(x["allowed_values"])!=1 for x in compiled["epistemic_state"]): return None
 return canonical_json([{"atom":x["atom"],"value":x["allowed_values"][0]} for x in sorted(compiled["epistemic_state"],key=lambda x:x["atom"])])
def clause_checks(compiled,reference):
 expected={x["id"]:x for x in reference["clauses"]}; return [x["id"] in expected and x["compiler_result"].get("status")=="ok" and x["compiler_result"].get("parse")==expected[x["id"]]["expected_parse"] for x in compiled["clauses"]]

def compile_record(record):
 agent=record["agent_input"]; predicate,operator,action=agent["predicate_ontology"],agent["operator_ontology"],agent["action_ontology"]; refs={x["id"]:x for x in record["reference"]["support_interventions"]}; qrefs={x["id"]:x for x in record["reference"]["queries"]}; matched=record["reference"]["matched_symbolic"]; matched_support={x["id"]:x for x in matched["support_interventions"]}; matched_queries={x["id"]:x for x in matched["queries"]}; clauses=[]; graphs=[]; commands=[]; sequences=[]; catalog_checks=[]; alignment=[]; supports=[]
 for public in agent["support_interventions"]:
  ref=refs[public["id"]]; symbolic=matched_support[public["id"]]; initial=compile_state(public["initial_state_language"],public["entities"],predicate,operator); clauses+=clause_checks(initial,ref["initial_state"]); initial_ok=graph_equal(initial["epistemic_state"],ref["initial_state"]["epistemic_state"]); graphs.append(initial_ok); actions=compile_action_sequence(public["action_language"],public["entities"],action); action_ok=actions.get("status")=="ok" and actions.get("actions")==ref["actions"]["actions"]; sequences.append(action_ok); commands += [action_ok]*len(ref["actions"]["actions"]); compiled_catalog={}
  for outcome_id,trajectory in public["outcome_catalog_language"].items():
   signatures=[]
   for step_index,(language_state,expected_signature) in enumerate(zip(trajectory,ref["outcome_catalog"][outcome_id],strict=True)):
    state=compile_state(language_state,public["entities"],predicate,operator); expected_rows=[{"atom":x["atom"],"allowed_values":[x["value"]]} for x in json.loads(expected_signature)]; clauses+=clause_checks(state,ref["outcome_state_references"][outcome_id][step_index])
    signature=compiled_signature(state); correct=signature==expected_signature and graph_equal(state["epistemic_state"],expected_rows); graphs.append(correct); catalog_checks.append(correct); signatures.append(signature or "__compile_failure__")
   compiled_catalog[outcome_id]=signatures
  trial_ok=public["realized_outcome_ids"]==symbolic["realized_outcome_ids"] and set(public["realized_outcome_ids"])<=set(compiled_catalog); alignment += [public["id"]==symbolic["id"],initial_ok,action_ok,compiled_catalog==symbolic["outcome_catalog"],trial_ok]
  supports.append({"id":public["id"],"entities":public["entities"],"initial_state":initial["epistemic_state"],"actions":actions.get("actions",[]),"outcome_catalog":compiled_catalog,"realized_outcome_ids":public["realized_outcome_ids"],"structural_key":public["paired_structural_key"]})
 queries=[]
 for public in agent["queries"]:
  ref=qrefs[public["id"]]; symbolic=matched_queries[public["id"]]; initial=compile_state(public["initial_state_language"],public["entities"],predicate,operator); clauses+=clause_checks(initial,ref["initial_state"]); initial_ok=graph_equal(initial["epistemic_state"],ref["initial_state"]["epistemic_state"]); graphs.append(initial_ok); actions=compile_action_sequence(public["action_language"],public["entities"],action); action_ok=actions.get("status")=="ok" and actions.get("actions")==ref["actions"]["actions"]; sequences.append(action_ok); commands += [action_ok]*len(ref["actions"]["actions"]); alignment += [public["id"]==symbolic["id"],initial_ok,action_ok,public["paired_structural_key"]==symbolic["structural_key"],any(x["id"]==public["id"] for x in record["oracle_queries"])]
  queries.append({"id":public["id"],"entities":public["entities"],"initial_state":initial["epistemic_state"],"actions":actions.get("actions",[]),"structural_key":public["paired_structural_key"],"sequence_length":public["sequence_length"],"entity_count":public["entity_count"]})
 safety=[evaluate_safety_challenge(x,agent["entity_catalog"],predicate,operator,action) for x in agent["safety_challenges"]]
 return {"support_interventions":supports,"queries":queries},{"clauses":clauses,"graphs":graphs,"commands":commands,"sequences":sequences,"catalogs":catalog_checks,"safety":safety,"alignment":alignment}

def true_map(distribution): return {key:Decimal(value.numerator)/Decimal(value.denominator) for key,value in joint_map(distribution).items()}
def tv(pred,true): return float(sum(abs(pred.get(k,0)-true.get(k,0)) for k in set(pred)|set(true))/2)
def log_loss(pred,outcomes):
 probabilities=[pred.get(x,0) for x in outcomes]
 return math.inf if any(not x for x in probabilities) else -mean(math.log(float(x)) for x in probabilities)
def brier(pred,outcomes): return mean(float(sum((pred.get(k,0)-(1 if k==observed else 0))**2 for k in set(pred)|set(outcomes))) for observed in outcomes)
def calibration(pairs):
 result=0.0; total=len(pairs)
 for i in range(10):
  rows=[x for x in pairs if i/10<=x[0]<(i+1)/10 or (i==9 and x[0]==1.0)]
  if rows: result+=len(rows)/total*abs(mean(x[0] for x in rows)-mean(x[1] for x in rows))
 return result
def program_skeleton(program):
 value=json.loads(canonical_json(program))
 for rule in value["rules"]:
  for branch in rule["stochastic_immediate"]+rule["stochastic_delayed"]: branch["probability"]="*"
 return canonical_json(value)

def evaluate_condition(record,condition,registry):
 supports=condition["support_interventions"]; queries=condition["queries"]; weights=posterior(registry,supports,32); target_index=next(i for i,x in enumerate(registry) if x["key"]==record["target"]["program_key"]); map_index=min(range(len(weights)),key=lambda i:(-weights[i],registry[i]["key"])); oracle={x["id"]:x for x in record["oracle_queries"]}; rows=[]; pairs=[]; shuffled=[]
 for query in queries:
  from v42_stateful import compatible_worlds
  worlds=compatible_worlds(query["initial_state"]); true_row=oracle[query["id"]]; true=true_map(true_row["true_joint_distribution"]); pred=posterior_predictive(registry,weights,query["entities"],worlds[0],query["actions"]); heldout=[canonical_json(true_row["outcome_catalog"][x]) for x in true_row["heldout_outcome_ids"]]; frequencies={key:heldout.count(key)/len(heldout) for key in set(pred)|set(true)}; pairs += [(float(pred.get(key,0)),frequencies.get(key,0.0)) for key in set(pred)|set(true)]; uniform={key:Decimal(1)/len(pred) for key in pred}; shuffled_pred=posterior_predictive(registry,weights,query["entities"],worlds[0],list(reversed(query["actions"]))); shuffled.append(tv(shuffled_pred,true)<=1e-12); rows.append({"id":query["id"],"sequence_length":query["sequence_length"],"tv":tv(pred,true),"log_loss":log_loss(pred,heldout),"brier":brier(pred,heldout),"uniform_log_loss":log_loss(uniform,heldout)})
 sorted_indices=sorted(range(len(weights)),key=lambda i:(-weights[i],registry[i]["key"])); runner=sorted_indices[1]; entropy=-sum(float(w)*math.log(float(w)) for w in weights if w); target_skeleton=program_skeleton(registry[target_index]["program"]); same_structure=sum(float(w) for w,m in zip(weights,registry,strict=True) if program_skeleton(m["program"])==target_skeleton)
 expected_probability=sum(float(w)*float(Fraction(m["probability"])) for w,m in zip(weights,registry,strict=True))
 return {"queries":rows,"map_schema_recovered":map_index==target_index,"target_program_mass":float(weights[target_index]),"probability_mae":abs(expected_probability-float(Fraction(record["oracle_metadata"]["probability"]))),"calibration_error":calibration(pairs),"uniform_disadvantage":mean(x["uniform_log_loss"]-x["log_loss"] for x in rows),"shuffled_exact":mean(shuffled),"literal_coverage":mean(q["structural_key"] in {s["structural_key"] for s in supports} for q in queries),"posterior_entropy_nats":entropy,"runner_up_mass":float(weights[runner]),"runner_up_program_key":registry[runner]["key"],"probability_posterior":probability_posterior(registry,weights),"structural_uncertainty_mass":max(0.0,1-same_structure),"within_structure_probability_uncertainty_mass":max(0.0,same_structure-float(weights[target_index]))}

def evaluate_record(record,registry):
 language_compiled,interface=compile_record(record); symbolic=record["reference"]["matched_symbolic"]; language=evaluate_condition(record,language_compiled,registry); baseline=evaluate_condition(record,symbolic,registry); language_tv=mean(x["tv"] for x in language["queries"]); symbolic_tv=mean(x["tv"] for x in baseline["queries"]); language_loss=mean(x["log_loss"] for x in language["queries"]); symbolic_loss=mean(x["log_loss"] for x in baseline["queries"])
 return {"id":record["id"],"split":record["split"],"family":record["construction_family"],"probability":record["oracle_metadata"]["probability"],"timing":record["oracle_metadata"]["timing"],"interface":interface,"language":language,"symbolic":baseline,"summary":{"language_tv":language_tv,"symbolic_tv":symbolic_tv,"tv_delta":language_tv-symbolic_tv,"language_log_loss":language_loss,"symbolic_log_loss":symbolic_loss,"log_loss_delta":language_loss-symbolic_loss}}

def bootstrap(values,seed):
 array=np.asarray(values,float); rng=np.random.default_rng(seed); samples=np.mean(array[rng.integers(0,len(array),size=(10000,len(array)))],axis=1); return {"mean":float(np.mean(array)),"lower":float(np.quantile(samples,.025)),"upper":float(np.quantile(samples,.975))}
def aggregate(records):
 clauses=[x for r in records for x in r["interface"]["clauses"]]; graphs=[x for r in records for x in r["interface"]["graphs"]]; commands=[x for r in records for x in r["interface"]["commands"]]; sequences=[x for r in records for x in r["interface"]["sequences"]]; safety=[x for r in records for x in r["interface"]["safety"]]; alignment=[x for r in records for x in r["interface"]["alignment"]]
 def condition(name,selected):
  query=[q for r in selected for q in r[name]["queries"]]; by_family={family:mean(q["tv"] for r in selected if r["family"]==family for q in r[name]["queries"]) for family in sorted({r["family"] for r in selected})}; by_probability={value:mean(q["tv"] for r in selected if r["probability"]==value for q in r[name]["queries"]) for value in sorted({r["probability"] for r in selected})}; by_timing={value:mean(q["tv"] for r in selected if r["timing"]==value for q in r[name]["queries"]) for value in sorted({r["timing"] for r in selected})}; by_length={str(value):mean(q["tv"] for r in selected for q in r[name]["queries"] if q["sequence_length"]==value) for value in sorted({q["sequence_length"] for r in selected for q in r[name]["queries"]})}; return {"mean_tv":mean(q["tv"] for q in query),"heldout_log_loss":mean(q["log_loss"] for q in query),"heldout_brier":mean(q["brier"] for q in query),"calibration_error":mean(r[name]["calibration_error"] for r in selected),"map_schema_recovery":mean(r[name]["map_schema_recovered"] for r in selected),"mean_target_program_posterior":mean(r[name]["target_program_mass"] for r in selected),"probability_mae":mean(r[name]["probability_mae"] for r in selected),"every_family_mean_tv":by_family,"every_probability_mean_tv":by_probability,"every_timing_mean_tv":by_timing,"every_sequence_length_mean_tv":by_length,"uniformized_log_loss_disadvantage":mean(r[name]["uniform_disadvantage"] for r in selected),"shuffled_action_order_exact":mean(r[name]["shuffled_exact"] for r in selected),"literal_lookup_coverage":mean(r[name]["literal_coverage"] for r in selected)}
 def report(selected):
  language=condition("language",selected); symbolic=condition("symbolic",selected); worst=max(selected,key=lambda r:r["summary"]["language_tv"]); return {"mechanics":len(selected),"language":language,"symbolic":symbolic,"language_minus_symbolic_tv":language["mean_tv"]-symbolic["mean_tv"],"language_minus_symbolic_log_loss":language["heldout_log_loss"]-symbolic["heldout_log_loss"],"paired_tv_bootstrap":bootstrap([r["summary"]["tv_delta"] for r in selected],4847),"paired_log_loss_bootstrap":bootstrap([r["summary"]["log_loss_delta"] for r in selected],4848),"worst_mechanic":{"id":worst["id"],"tv":worst["summary"]["language_tv"]},"worst_family":max(language["every_family_mean_tv"],key=language["every_family_mean_tv"].get),"worst_probability":max(language["every_probability_mean_tv"],key=language["every_probability_mean_tv"].get),"worst_timing":max(language["every_timing_mean_tv"],key=language["every_timing_mean_tv"].get),"worst_sequence_length":max(language["every_sequence_length_mean_tv"],key=language["every_sequence_length_mean_tv"].get),"posterior_diagnostics":[{"id":r["id"],"target_mass":r["language"]["target_program_mass"],"entropy":r["language"]["posterior_entropy_nats"],"runner_up_mass":r["language"]["runner_up_mass"],"runner_up_program_key":r["language"]["runner_up_program_key"],"probability_posterior":r["language"]["probability_posterior"],"structural_uncertainty_mass":r["language"]["structural_uncertainty_mass"],"within_structure_probability_uncertainty_mass":r["language"]["within_structure_probability_uncertainty_mass"]} for r in selected]}
 return {"interface":{"state_clause_exact_parse":mean(clauses),"canonical_graph_exact":mean(graphs),"action_command_accuracy":mean(commands),"action_sequence_exact":mean(sequences),"fail_closed_safety_accuracy":mean(safety),"trial_alignment_exact":mean(alignment),"counts":{"clauses":len(clauses),"graphs":len(graphs),"commands":len(commands),"sequences":len(sequences),"safety":len(safety),"alignment":len(alignment)}},"all_mechanics":report(records),"development_evaluation":report([r for r in records if r["split"]=="development_evaluation"])}

def qualification(m,g):
 i=m["interface"]; x=m["all_mechanics"]; l=x["language"]; checks={"clause_parse":i["state_clause_exact_parse"]>=g["minimumClauseParseAccuracy"],"canonical_graph":i["canonical_graph_exact"]>=g["minimumCanonicalGraphExact"],"action_command":i["action_command_accuracy"]>=g["minimumActionCommandAccuracy"],"action_sequence":i["action_sequence_exact"]>=g["minimumActionSequenceExact"],"safety":i["fail_closed_safety_accuracy"]>=g["minimumFailClosedSafetyAccuracy"],"trial_alignment":i["trial_alignment_exact"]>=1.0,"map_schema":l["map_schema_recovery"]>=g["minimumMapSchemaRecovery"],"target_posterior":l["mean_target_program_posterior"]>=g["minimumMeanTargetProgramPosterior"],"probability_mae":l["probability_mae"]<=g["maximumProbabilityParameterMeanAbsoluteError"],"mean_tv":l["mean_tv"]<=g["maximumMeanJointDistributionTotalVariation"],"every_family_tv":max(l["every_family_mean_tv"].values())<=g["maximumEveryFamilyMeanJointDistributionTotalVariation"],"calibration":l["calibration_error"]<=g["maximumCalibrationError"],"composition_tv":x["language_minus_symbolic_tv"]<=g["maximumLanguageMinusSymbolicMeanTv"],"composition_log_loss":x["language_minus_symbolic_log_loss"]<=g["maximumLanguageMinusSymbolicLogLoss"],"uniformized_inadequate":l["uniformized_log_loss_disadvantage"]>=g["minimumUniformizedLogLossDisadvantageNats"],"shuffled_inadequate":l["shuffled_action_order_exact"]<=g["maximumShuffledActionOrderExactDistributionMatch"],"literal_inadequate":l["literal_lookup_coverage"]<=g["maximumLiteralLanguageLookupCoverage"]}; return {"passed":all(checks.values()),"checks":checks}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--corpus-seal",default="configs/v48-corpus-seal.json"); p.add_argument("--output-dir",default="outputs/v48-stochastic-language-composition/development"); a=p.parse_args(); seal_path=(PROJECT_ROOT/a.corpus_seal).resolve(); output=(PROJECT_ROOT/a.output_dir).resolve(); attempt=output.parent/"development-attempt.json"
 if output.exists() or attempt.exists(): raise RuntimeError("V48 already attempted")
 seal=json.loads(seal_path.read_text()); impl_path=PROJECT_ROOT/seal["implementation_lock"]; impl=json.loads(impl_path.read_text()); records=[]
 for path,expected in impl["implementation"].items():
  if file_sha256(PROJECT_ROOT/path)!=expected: raise RuntimeError(f"V48 implementation changed: {path}")
 for artifact in seal["corpora"].values():
  path=PROJECT_ROOT/artifact["path"]
  if file_sha256(path)!=artifact["sha256"]: raise RuntimeError("V48 sealed corpus changed")
  records+=read(path)
 output.parent.mkdir(parents=True,exist_ok=True); attempt.write_text(json.dumps({"schema_version":48,"status":"started","development_run":1,"corpus_seal_sha256":file_sha256(seal_path)},indent=2,sort_keys=True)+"\n"); started=time.perf_counter(); registry=mechanic_registry(); details=[evaluate_record(x,registry) for x in sorted(records,key=lambda x:x["id"])]; metrics=aggregate(details); q=qualification(metrics,impl["config_payload"]["gates"])
 if q["passed"]: decision="stochastic_language_composition_pass_preregister_passive_partial_observation"
 elif metrics["all_mechanics"]["symbolic"]["mean_tv"]<=impl["config_payload"]["gates"]["maximumMeanJointDistributionTotalVariation"]: decision="repair_declared_language_interface_only"
 else: decision="revisit_sampled_stochastic_identifiability"
 output.mkdir(parents=True); detail_path=output/"mechanic-metrics.jsonl"; detail_path.write_text("".join(canonical_json(x)+"\n" for x in details)); result={"schema_version":48,"experiment":impl["config_payload"]["experiment"],"corpus_seal":str(seal_path.relative_to(PROJECT_ROOT)),"corpus_seal_sha256":file_sha256(seal_path),"development_run_number":1,"metrics":metrics,"qualification":q,"decision":decision,"mechanic_metrics":str(detail_path.relative_to(PROJECT_ROOT)),"mechanic_metrics_sha256":file_sha256(detail_path),"runtime_seconds":time.perf_counter()-started,"data_access":{"development_runs":1,"mechanics_scored":len(records),"selection_on_development_evaluation":0,"model_forward_passes":0,"adapter_training_runs":0},"authorization":{"preregister_passive_partial_observation":q["passed"],"construct_partial_observation_population":False,"active_intervention_selection":False,"open_ontology":False,"final_evaluation":False,"model_access":False}}; result_path=output/"result.json"; result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); state=json.loads(attempt.read_text()); state.update({"status":"completed","result_sha256":file_sha256(result_path)}); attempt.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
