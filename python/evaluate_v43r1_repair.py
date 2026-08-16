#!/usr/bin/env python3
"""Run the single V43r1 graph-measurement repair rescore."""
from __future__ import annotations
import argparse,json,time
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v42_stateful import mechanic_registry
from v43_language import compile_state
from v43r1_measurement import duplicate_free,graph_equal
from evaluate_v43_language import aggregate,evaluate_record,qualification as v43_qualification,read

def graph_pairs(record):
 agent=record["agent_input"]; support_refs={x["id"]:x for x in record["reference"]["support_sequences"]}; query_refs={x["id"]:x for x in record["reference"]["queries"]}
 for public in agent["support_sequences"]:
  ref=support_refs[public["id"]]
  yield compile_state(public["initial_state_language"],public["entities"],agent["predicate_ontology"],agent["operator_ontology"])["epistemic_state"],ref["initial_state"]["epistemic_state"]
  for language,expected in zip(public["observed_step_state_language"],ref["observed_step_states"]):
   yield compile_state(language,public["entities"],agent["predicate_ontology"],agent["operator_ontology"])["epistemic_state"],expected["epistemic_state"]
 for public in agent["queries"]:
  ref=query_refs[public["id"]]
  yield compile_state(public["initial_state_language"],public["entities"],agent["predicate_ontology"],agent["operator_ontology"])["epistemic_state"],ref["initial_state"]["epistemic_state"]

def compute(records,original):
 registry=mechanic_registry(); reproduced=aggregate([evaluate_record(row,registry)[0] for row in records]); pairs=[pair for row in records for pair in graph_pairs(row)]
 other_metric_keys=sorted(set(original["metrics"])-{"state_graph_exact"})
 other_metric_checks={key:reproduced[key]==original["metrics"][key] for key in other_metric_keys}
 reproduced_gates=v43_qualification(reproduced,original["registered_gates"])["checks"]
 other_gate_keys=sorted(set(original["gate_checks"])-{"state_graph_exact"})
 other_gate_checks={key:reproduced_gates[key]==original["gate_checks"][key] and reproduced_gates[key] for key in other_gate_keys}
 canonical_checks=[graph_equal(left,right) for left,right in pairs]
 duplicate_checks=[duplicate_free(left) and duplicate_free(right) for left,right in pairs]
 permutation_checks=[graph_equal(left,list(reversed(right))) and graph_equal(list(reversed(left)),right) for left,right in pairs]
 return {"graph_pairs":len(pairs),"original_registered_ordered_graph_exact":original["metrics"]["state_graph_exact"],"canonical_graph_exact":sum(canonical_checks)/len(canonical_checks),"duplicate_free":sum(duplicate_checks)/len(duplicate_checks),"semantic_content_mismatches":sum(not x for x in canonical_checks),"comparator_permutation_invariance":sum(permutation_checks)/len(permutation_checks),"other_v43_metrics_reproduced":sum(other_metric_checks.values())/len(other_metric_checks),"other_v43_gate_checks_passed":sum(other_gate_checks.values())/len(other_gate_checks),"other_metric_checks":other_metric_checks,"other_gate_checks":other_gate_checks,"reproduced_v43_metrics":reproduced}

def qualify(metrics,gates):
 checks={"canonical_graph_exact":metrics["canonical_graph_exact"]>=gates["minimumCanonicalGraphExact"],"duplicate_free":metrics["duplicate_free"]>=gates["minimumDuplicateFree"],"semantic_content_mismatches":metrics["semantic_content_mismatches"]<=gates["maximumSemanticContentMismatches"],"other_v43_metrics_reproduced":metrics["other_v43_metrics_reproduced"]>=gates["minimumOtherV43MetricsReproduced"],"other_v43_gate_checks_passed":metrics["other_v43_gate_checks_passed"]>=gates["minimumOtherV43GateChecksPassed"],"comparator_permutation_invariance":metrics["comparator_permutation_invariance"]>=gates["minimumComparatorPermutationInvariance"]}
 return {"passed":all(checks.values()),"checks":checks}

def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v43r1-implementation-lock.json"); p.add_argument("--output-dir",default="outputs/v43r1-graph-measurement-repair/rescore"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); output=(PROJECT_ROOT/a.output_dir).resolve(); attempt=output.parent/"rescore-attempt.json"
 if output.exists() or attempt.exists(): raise RuntimeError("V43r1 repair rescore already attempted")
 lock=json.loads(lock_path.read_text())
 for path,expected in lock["implementation"].items():
  if file_sha256(PROJECT_ROOT/path)!=expected: raise RuntimeError(f"V43r1 implementation changed: {path}")
 outcome=json.loads((PROJECT_ROOT/lock["source_v43_outcome_lock"]).read_text()); original_result_path=PROJECT_ROOT/outcome["result"]; original_result=json.loads(original_result_path.read_text()); seal_path=PROJECT_ROOT/lock["source_v43_corpus_seal"]; seal=json.loads(seal_path.read_text()); records=[]
 for artifact in seal["corpora"].values():
  path=PROJECT_ROOT/artifact["path"]
  if file_sha256(path)!=artifact["sha256"]: raise RuntimeError("Sealed V43 corpus changed")
  records.extend(read(path))
 records.sort(key=lambda x:x["id"]); attempt.parent.mkdir(parents=True,exist_ok=True); attempt.write_text(json.dumps({"schema_version":"43r1","status":"started","repair_rescore":1,"implementation_lock_sha256":file_sha256(lock_path)},indent=2,sort_keys=True)+"\n"); started=time.perf_counter()
 original={"metrics":original_result["metrics"],"gate_checks":original_result["qualification"]["checks"],"registered_gates":lock["v43_registered_gates"]}; metrics=compute(records,original); qualified=qualify(metrics,lock["config_payload"]["gates"]); decision="measurement_repair_pass_preregister_deterministic_delay" if qualified["passed"] else "measurement_repair_rejected"
 result={"schema_version":"43r1","experiment":"v43r1_graph_measurement_repair","implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"source_v43_result":str(original_result_path.relative_to(PROJECT_ROOT)),"source_v43_result_sha256":file_sha256(original_result_path),"original_v43_qualification_passed":False,"original_v43_outcome_unchanged":True,"repair_rescore_number":1,"metrics":metrics,"qualification":qualified,"decision":decision,"runtime_seconds":time.perf_counter()-started,"data_access":{"repair_rescores":1,"v43_records_read":40,"model_forward_passes":0,"adapter_training_runs":0,"new_corpus_records":0},"authorization":{"preregister_deterministic_delayed_effects":qualified["passed"],"construct_delayed_effects_benchmark":False,"add_stochasticity":False,"final_evaluation":False,"model_access":False}}
 output.mkdir(parents=True,exist_ok=False); result_path=output/"result.json"; result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); state=json.loads(attempt.read_text()); state.update({"status":"completed","result_sha256":file_sha256(result_path)}); attempt.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
