#!/usr/bin/env python3
"""Run the single sealed V44 oracle delayed-effect development."""
from __future__ import annotations
import argparse,json,time
from collections import defaultdict
from typing import Any,Sequence
import numpy as np
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json
from v42_stateful import compatible_worlds
from v44_delayed import execute_partial,execute_sequence,mechanic_registry,world_signature

def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def rate(values:Sequence[bool]): return sum(values)/len(values) if values else 0.0
def observed_signatures(support): return [canonical_json(sorted(step,key=lambda x:x["atom"])) for step in support["observed_step_states"]]
def predicted_signatures(program,support,control="queued"):
 worlds=compatible_worlds(support["initial_state"])
 if len(worlds)!=1: raise ValueError("V44 support states must be complete")
 return [world_signature(x) for x in execute_sequence(program,support["entities"],worlds[0],support["actions"],control)]
def version_space(record,registry):
 survivors=list(registry); prefixes=[]; target=record["target"]["program_key"]
 for support in record["agent_input"]["support_sequences"]:
  observed=observed_signatures(support); survivors=[m for m in survivors if predicted_signatures(m["program"],support)==observed]; prefixes.append({"prefix":len(prefixes)+1,"version_space":len(survivors),"target_retained":any(m["key"]==target for m in survivors)})
 return survivors,prefixes
def evaluate_record(record,registry):
 by_key={m["key"]:m for m in registry}; target_key=record["target"]["program_key"]; target_program=by_key[target_key]["program"]; supports=record["agent_input"]["support_sequences"]; queries=record["agent_input"]["queries"]; targets={x["id"]:x["target"] for x in record["oracle_queries"]}; validation=[predicted_signatures(target_program,s)==observed_signatures(s) for s in supports]
 for q in queries: validation.append(execute_partial([target_program],q["entities"],q["initial_state"],q["actions"])==targets[q["id"]])
 survivors,prefixes=version_space(record,registry); programs=[x["program"] for x in survivors]; lookup={s["structural_key"]:observed_signatures(s)[-1] for s in supports}; query_rows=[]
 for q in queries:
  target=targets[q["id"]]; primary=execute_partial(programs,q["entities"],q["initial_state"],q["actions"]); collapsed=execute_partial([target_program],q["entities"],q["initial_state"],q["actions"],"collapsed_delay"); end_flush=execute_partial([target_program],q["entities"],q["initial_state"],q["actions"],"end_flush"); literal=[lookup[q["structural_key"]]] if q["structural_key"] in lookup else ["__unseen_sequence__"]
  query_rows.append({"id":q["id"],"family":record["construction_family"],"split":record["split"],"sequence_length":q["sequence_length"],"partial_initial_state":q["partial_initial_state"],"wait_group":q["wait_counterfactual_group"],"wait_role":q["wait_counterfactual_role"],"step_exact":primary["possible_step_states"]==target["possible_step_states"],"final_exact":primary["possible_final_observations"]==target["possible_final_observations"],"collapsed_final_exact":collapsed["possible_final_observations"]==target["possible_final_observations"],"end_flush_final_exact":end_flush["possible_final_observations"]==target["possible_final_observations"],"lookup_final_exact":literal==target["possible_final_observations"],"predicted_final":primary["possible_final_observations"],"target_final":target["possible_final_observations"]})
 groups=defaultdict(list)
 for row in query_rows: groups[row["wait_group"]].append(row)
 wait_checks=[]
 for rows in groups.values():
  valid=len(rows)==2 and {x["wait_role"] for x in rows}=={"delivered","pending"}; wait_checks.append(valid and rows[0]["target_final"]!=rows[1]["target_final"] and rows[0]["predicted_final"]!=rows[1]["predicted_final"] and all(x["final_exact"] for x in rows))
 target_retained=any(x["key"]==target_key for x in survivors); schema=len(survivors)==1 and survivors[0]["key"]==target_key; metrics={"program_validation":all(validation),"target_retained":target_retained,"schema_recovered":schema,"empty_version_space":not survivors,"version_space":len(survivors),"queries":query_rows,"wait_checks":wait_checks,"complete_mechanic_exact":all(x["step_exact"] and x["final_exact"] for x in query_rows),"prefixes":prefixes}; prediction={"id":record["id"],"family":record["construction_family"],"split":record["split"],"target_retained":target_retained,"schema_recovered":schema,"version_space":len(survivors),"next_state_exact":rate([x["step_exact"] for x in query_rows]),"final_exact":rate([x["final_exact"] for x in query_rows]),"wait_counterfactual_accuracy":rate(wait_checks),"collapsed_delay_final_exact":rate([x["collapsed_final_exact"] for x in query_rows]),"end_flush_final_exact":rate([x["end_flush_final_exact"] for x in query_rows]),"literal_lookup_final_exact":rate([x["lookup_final_exact"] for x in query_rows])}; return metrics,prediction
def aggregate(metrics):
 queries=[x for m in metrics for x in m["queries"]]; waits=[x for m in metrics for x in m["wait_checks"]]
 def grouped(field,metric): return {str(value):rate([x[metric] for x in queries if x[field]==value]) for value in sorted({x[field] for x in queries},key=str)}
 return {"mechanics":len(metrics),"queries":len(queries),"oracle_program_validation":rate([m["program_validation"] for m in metrics]),"queued_target_retention":rate([m["target_retained"] for m in metrics]),"queued_schema_recovery":rate([m["schema_recovered"] for m in metrics]),"queued_empty_version_space":rate([m["empty_version_space"] for m in metrics]),"queued_next_state_exact":rate([x["step_exact"] for x in queries]),"queued_final_observation_exact":rate([x["final_exact"] for x in queries]),"queued_complete_mechanic_exact":rate([m["complete_mechanic_exact"] for m in metrics]),"queued_by_family_final_exact":grouped("family","final_exact"),"queued_by_sequence_length_final_exact":grouped("sequence_length","final_exact"),"queued_by_split_final_exact":grouped("split","final_exact"),"queued_partial_initial_final_exact":grouped("partial_initial_state","final_exact"),"wait_counterfactual_pairs":len(waits),"wait_placement_counterfactual_accuracy":rate(waits),"collapsed_delay_final_exact":rate([x["collapsed_final_exact"] for x in queries]),"end_flush_final_exact":rate([x["end_flush_final_exact"] for x in queries]),"literal_lookup_final_exact":rate([x["lookup_final_exact"] for x in queries]),"median_final_version_space":float(np.median([m["version_space"] for m in metrics]))}
def qualification(m,g):
 checks={"oracle_program_validation":m["oracle_program_validation"]>=g["minimumOracleProgramValidation"],"queued_target_retention":m["queued_target_retention"]>=g["minimumQueuedTargetRetention"],"queued_schema_recovery":m["queued_schema_recovery"]>=g["minimumQueuedSchemaRecovery"],"queued_empty_version_space":m["queued_empty_version_space"]<=g["maximumQueuedEmptyVersionSpace"],"queued_next_state_exact":m["queued_next_state_exact"]>=g["minimumQueuedNextStateExact"],"queued_final_observation_exact":m["queued_final_observation_exact"]>=g["minimumQueuedFinalObservationExact"],"queued_every_family":min(m["queued_by_family_final_exact"].values())>=g["minimumQueuedEveryFamilyExact"],"queued_every_sequence_length":min(m["queued_by_sequence_length_final_exact"].values())>=g["minimumQueuedEverySequenceLengthExact"],"wait_placement_counterfactual_accuracy":m["wait_placement_counterfactual_accuracy"]>=g["minimumWaitPlacementCounterfactualAccuracy"],"collapsed_delay_inadequate":m["collapsed_delay_final_exact"]<=g["maximumCollapsedDelayFinalExact"],"end_flush_inadequate":m["end_flush_final_exact"]<=g["maximumEndFlushFinalExact"],"literal_lookup_inadequate":m["literal_lookup_final_exact"]<=g["maximumLiteralLookupFinalExact"]}; return {"passed":all(checks.values()),"checks":checks}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--corpus-seal",default="configs/v44-corpus-seal.json"); p.add_argument("--output-dir",default="outputs/v44-deterministic-delayed-effects/development"); a=p.parse_args(); seal_path=(PROJECT_ROOT/a.corpus_seal).resolve(); output=(PROJECT_ROOT/a.output_dir).resolve(); attempt=output.parent/"development-attempt.json"
 if output.exists() or attempt.exists(): raise RuntimeError("V44 development already attempted")
 seal=json.loads(seal_path.read_text()); impl_path=PROJECT_ROOT/seal["implementation_lock"]; impl=json.loads(impl_path.read_text())
 for path,expected in impl["implementation"].items():
  if file_sha256(PROJECT_ROOT/path)!=expected: raise RuntimeError(f"V44 implementation changed: {path}")
 records=[]
 for artifact in seal["corpora"].values():
  path=PROJECT_ROOT/artifact["path"]
  if file_sha256(path)!=artifact["sha256"]: raise RuntimeError("V44 sealed corpus changed")
  records.extend(read(path))
 records.sort(key=lambda x:x["id"]); output.parent.mkdir(parents=True,exist_ok=True); attempt.write_text(json.dumps({"schema_version":44,"status":"started","oracle_development_run":1,"corpus_seal_sha256":file_sha256(seal_path)},indent=2,sort_keys=True)+"\n"); started=time.perf_counter(); registry=mechanic_registry(); all_metrics=[]; predictions=[]
 for record in records:
  metric,prediction=evaluate_record(record,registry); all_metrics.append(metric); predictions.append(prediction)
 metrics=aggregate(all_metrics); qualified=qualification(metrics,impl["config_payload"]["gates"]); checks=qualified["checks"]
 if qualified["passed"]: decision="delayed_foundation_pass_preregister_delayed_language_grounding"
 elif not all(checks[k] for k in ("queued_target_retention","queued_schema_recovery","queued_next_state_exact","queued_final_observation_exact")): decision="repair_event_queue_or_delayed_dsl"
 elif not checks["collapsed_delay_inadequate"] or not checks["end_flush_inadequate"]: decision="redesign_timing_causal_requirements"
 else: decision="redesign_structural_generalization"
 output.mkdir(parents=True,exist_ok=False); predictions_path=output/"mechanic-predictions.jsonl"; predictions_path.write_text("".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in predictions)); result={"schema_version":44,"experiment":impl["config_payload"]["experiment"],"corpus_seal":str(seal_path.relative_to(PROJECT_ROOT)),"corpus_seal_sha256":file_sha256(seal_path),"oracle_development_run_number":1,"metrics":metrics,"qualification":qualified,"decision":decision,"predictions":str(predictions_path.relative_to(PROJECT_ROOT)),"predictions_sha256":file_sha256(predictions_path),"runtime_seconds":time.perf_counter()-started,"data_access":{"oracle_development_runs":1,"mechanics_scored":len(records),"selection_on_development_evaluation":0,"model_forward_passes":0,"adapter_training_runs":0,"v43_records_read":0},"authorization":{"preregister_delayed_language_grounding":qualified["passed"],"construct_delayed_language_grounding":False,"add_stochasticity":False,"active_intervention_selection":False,"open_ontology":False,"final_evaluation":False,"model_access":False}}; result_path=output/"result.json"; result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); state=json.loads(attempt.read_text()); state.update({"status":"completed","result_sha256":file_sha256(result_path)}); attempt.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
