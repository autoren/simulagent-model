#!/usr/bin/env python3
"""Run the single sealed V47 finite-sample stochastic development."""
from __future__ import annotations
import argparse,json,math,time
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
import numpy as np
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json
from v42_stateful import compatible_worlds
from v47_sampling import execute_joint_distribution,joint_map,mechanic_registry,posterior,posterior_predictive

def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def mean(x): return sum(x)/len(x) if x else 0.0
def world(case):
 rows=compatible_worlds(case["initial_state"])
 if len(rows)!=1: raise ValueError("V47 requires complete initial states")
 return rows[0]
def float_truth(distribution): return {k:Decimal(v.numerator)/Decimal(v.denominator) for k,v in joint_map(distribution).items()}
def tv(pred,true): return float(sum(abs(pred.get(k,0)-true.get(k,0)) for k in set(pred)|set(true))/2)
def log_loss(pred,outcomes):
 values=[pred.get(key,0.0) for key in outcomes]
 return math.inf if any(x<=0 for x in values) else -mean([math.log(x) for x in values])
def brier(pred,outcomes):
 keys=set(pred)|set(outcomes); scores=[]
 for observed in outcomes: scores.append(sum((pred.get(key,0)-(1 if key==observed else 0))**2 for key in keys))
 return float(mean(scores))
def uniformized(pred): return {key:1/len(pred) for key in pred}
def calibration_error(pairs):
 total=len(pairs); result=0.0
 for index in range(10):
  lower=index/10; upper=(index+1)/10; selected=[x for x in pairs if lower<=x[0]<(upper if index<9 else upper+1e-12)]
  if selected: result+=len(selected)/total*abs(mean([x[0] for x in selected])-mean([x[1] for x in selected]))
 return result
def bootstrap(values,seed=4747):
 rng=np.random.default_rng(seed); array=np.asarray(values,float); samples=np.mean(array[rng.integers(0,len(array),size=(10000,len(array)))],axis=1); return {"mean":float(np.mean(array)),"lower":float(np.quantile(samples,.025)),"upper":float(np.quantile(samples,.975))}

def evaluate_record(record,registry,budgets):
 target_index=next(i for i,m in enumerate(registry) if m["key"]==record["target"]["program_key"]); target_probability=float(Fraction(record["oracle_metadata"]["probability"])); supports=record["agent_input"]["support_interventions"]; queries=record["agent_input"]["queries"]; oracle={x["id"]:x for x in record["oracle_queries"]}; results={}
 for budget in budgets:
  weights=posterior(registry,supports,budget); map_index=min(range(len(weights)),key=lambda i:(-weights[i],registry[i]["key"])); rows=[]; calibration=[]
  for query in queries:
   initial=world(query); truth_row=oracle[query["id"]]; true=float_truth(truth_row["true_joint_distribution"]); pred=posterior_predictive(registry,weights,query["entities"],initial,query["actions"]); map_pred={k:float(v) for k,v in joint_map(execute_joint_distribution(registry[map_index]["program"],query["entities"],initial,query["actions"])).items()}; uniform=uniformized(pred); catalog=truth_row["outcome_catalog"]; heldout=[canonical_json(catalog[x]) for x in truth_row["heldout_outcome_ids"]]; frequencies={key:heldout.count(key)/len(heldout) for key in set(true)|set(pred)}
   calibration.extend((float(pred.get(key,0)),frequencies.get(key,0.0)) for key in set(true)|set(pred))
   rows.append({"sequence_length":query["sequence_length"],"tv":tv(pred,true),"log_loss":log_loss(pred,heldout),"brier":brier(pred,heldout),"map_log_loss":log_loss(map_pred,heldout),"uniform_log_loss":log_loss(uniform,heldout),"normalized":abs(sum(pred.values())-1)<1e-12})
  expected_probability=sum(float(weight)*float(Fraction(m["probability"])) for weight,m in zip(weights,registry,strict=True)); results[str(budget)]={"target_posterior":float(weights[target_index]),"map_schema_recovered":map_index==target_index,"probability_mae":abs(expected_probability-target_probability),"calibration_error":calibration_error(calibration),"queries":rows,"literal_lookup_coverage":mean([q["structural_key"] in {s["structural_key"] for s in supports} for q in queries])}
 return {"id":record["id"],"family":record["construction_family"],"probability":record["oracle_metadata"]["probability"],"timing":record["oracle_metadata"]["timing"],"split":record["split"],"budgets":results}

def aggregate(records,budgets):
 output={"mechanics":len(records)}
 for budget in budgets:
  key=str(budget); mechanics=[r["budgets"][key] for r in records]; queries=[q for m in mechanics for q in m["queries"]]; family={}
  for value in sorted({r["family"] for r in records}): family[value]=mean([q["tv"] for r in records if r["family"]==value for q in r["budgets"][key]["queries"]])
  mechanic_tvs=[mean([row["tv"] for row in r["budgets"][key]["queries"]]) for r in records]
  output[key]={"predictive_mass_normalization":mean([q["normalized"] for q in queries]),"mean_joint_distribution_tv":mean([q["tv"] for q in queries]),"every_family_mean_tv":family,"heldout_log_loss":mean([q["log_loss"] for q in queries]),"heldout_brier":mean([q["brier"] for q in queries]),"map_log_loss":mean([q["map_log_loss"] for q in queries]),"uniformized_log_loss":mean([q["uniform_log_loss"] for q in queries]),"log_loss_improvement_over_uniformized":mean([q["uniform_log_loss"]-q["log_loss"] for q in queries]),"map_schema_recovery":mean([m["map_schema_recovered"] for m in mechanics]),"mean_target_program_posterior":mean([m["target_posterior"] for m in mechanics]),"probability_parameter_mae":mean([m["probability_mae"] for m in mechanics]),"calibration_error":mean([m["calibration_error"] for m in mechanics]),"literal_lookup_coverage":mean([m["literal_lookup_coverage"] for m in mechanics]),"mechanic_cluster_bootstrap_mean_tv":bootstrap(mechanic_tvs,4747+budget)}
 return output

def qualification(m,g):
 x=m["128"]; checks={"mass_normalization":x["predictive_mass_normalization"]>=g["minimumPredictiveMassNormalization"],"mean_tv":x["mean_joint_distribution_tv"]<=g["maximumMeanJointDistributionTotalVariation"],"every_family_tv":max(x["every_family_mean_tv"].values())<=g["maximumEveryFamilyMeanJointDistributionTotalVariation"],"probability_mae":x["probability_parameter_mae"]<=g["maximumProbabilityParameterMeanAbsoluteError"],"map_schema_recovery":x["map_schema_recovery"]>=g["minimumMapSchemaRecovery"],"target_posterior":x["mean_target_program_posterior"]>=g["minimumMeanTargetProgramPosteriorMass"],"calibration":x["calibration_error"]<=g["maximumCalibrationError"],"uniformized_inadequate":x["log_loss_improvement_over_uniformized"]>=g["minimumLogLossImprovementOverUniformizedNats"],"literal_lookup_inadequate":x["literal_lookup_coverage"]<=g["maximumLiteralLookupCoverage"],"tv_improves_8_to_32":m["32"]["mean_joint_distribution_tv"]<m["8"]["mean_joint_distribution_tv"],"tv_improves_32_to_128":m["128"]["mean_joint_distribution_tv"]<m["32"]["mean_joint_distribution_tv"],"posterior_no_worse_than_map":x["heldout_log_loss"]<=x["map_log_loss"]}
 return {"passed":all(checks.values()),"checks":checks}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--corpus-seal",default="configs/v47-corpus-seal.json"); p.add_argument("--output-dir",default="outputs/v47-sampled-transition-estimation/development"); a=p.parse_args(); seal_path=(PROJECT_ROOT/a.corpus_seal).resolve(); output=(PROJECT_ROOT/a.output_dir).resolve(); attempt=output.parent/"development-attempt.json"
 if output.exists() or attempt.exists(): raise RuntimeError("V47 development already attempted")
 seal=json.loads(seal_path.read_text()); impl_path=PROJECT_ROOT/seal["implementation_lock"]; impl=json.loads(impl_path.read_text())
 for path,expected in impl["implementation"].items():
  if file_sha256(PROJECT_ROOT/path)!=expected: raise RuntimeError(f"V47 implementation changed: {path}")
 records=[]
 for artifact in seal["corpora"].values():
  path=PROJECT_ROOT/artifact["path"]
  if file_sha256(path)!=artifact["sha256"]: raise RuntimeError("V47 sealed corpus changed")
  records.extend(read(path))
 output.parent.mkdir(parents=True,exist_ok=True); attempt.write_text(json.dumps({"schema_version":47,"status":"started","sampled_development_run":1,"corpus_seal_sha256":file_sha256(seal_path)},indent=2,sort_keys=True)+"\n"); started=time.perf_counter(); registry=mechanic_registry(); budgets=impl["config_payload"]["population"]["nestedTrialsPerIntervention"]; details=[evaluate_record(record,registry,budgets) for record in sorted(records,key=lambda x:x["id"])]; metrics=aggregate(details,budgets); q=qualification(metrics,impl["config_payload"]["gatesAt128TrialsPerIntervention"])
 if q["passed"]: decision="sampled_estimation_pass_preregister_stochastic_language_composition"
 elif not q["checks"]["mass_normalization"]: decision="repair_joint_likelihood_or_predictive_normalization"
 elif q["checks"]["map_schema_recovery"] and not q["checks"]["calibration"]: decision="revise_posterior_calibration_interface"
 else: decision="revisit_stochastic_identifiability_before_architecture_expansion"
 output.mkdir(parents=True); predictions=output/"mechanic-metrics.jsonl"; predictions.write_text("".join(canonical_json(x)+"\n" for x in details)); result={"schema_version":47,"experiment":impl["config_payload"]["experiment"],"corpus_seal":str(seal_path.relative_to(PROJECT_ROOT)),"corpus_seal_sha256":file_sha256(seal_path),"sampled_development_run_number":1,"metrics":metrics,"qualification":q,"decision":decision,"mechanic_metrics":str(predictions.relative_to(PROJECT_ROOT)),"mechanic_metrics_sha256":file_sha256(predictions),"runtime_seconds":time.perf_counter()-started,"data_access":{"sampled_development_runs":1,"mechanics_scored":len(records),"selection_on_development_evaluation":0,"model_forward_passes":0,"adapter_training_runs":0},"authorization":{"preregister_stochastic_language_composition":q["passed"],"construct_stochastic_language_population":False,"active_intervention_selection":False,"open_ontology":False,"final_evaluation":False,"model_access":False}}; result_path=output/"result.json"; result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); state=json.loads(attempt.read_text()); state.update({"status":"completed","result_sha256":file_sha256(result_path)}); attempt.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
