#!/usr/bin/env python3
"""Fit on V38 focus fit only and perform the single frozen validation."""
from __future__ import annotations
import argparse,json,time
import numpy as np
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v35_binding import make_ridge
from v38_focus_parser import extract_literal_candidates
from v38_evaluation import METHODS,cross_validate,predict_method,qualification,score,select_method
def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--features-lock",default="configs/v38-features-lock.json"); p.add_argument("--output-dir",default="outputs/v38-ontology-anchored-focus-parser/evaluation"); a=p.parse_args(); feature_lock_path=(PROJECT_ROOT/a.features_lock).resolve(); output=(PROJECT_ROOT/a.output_dir).resolve(); attempt=output.parent/"evaluation-attempt.json"
 if output.exists() or attempt.exists(): raise RuntimeError("V38 evaluation already attempted")
 fl=json.loads(feature_lock_path.read_text()); seal_path=PROJECT_ROOT/fl["corpus_seal"]; seal=json.loads(seal_path.read_text()); impl_path=PROJECT_ROOT/seal["implementation_lock"]; impl=json.loads(impl_path.read_text())
 for path,expected in impl["implementation"].items():
  if file_sha256(PROJECT_ROOT/path)!=expected: raise RuntimeError(f"V38 implementation changed: {path}")
 meta_path=PROJECT_ROOT/fl["feature_metadata"]; artifact=PROJECT_ROOT/fl["feature_artifact"]
 if file_sha256(meta_path)!=fl["feature_metadata_sha256"] or file_sha256(artifact)!=fl["feature_artifact_sha256"]: raise RuntimeError("V38 features changed")
 npz=np.load(artifact); ids=npz["record_ids"].tolist(); fit=sorted(read(PROJECT_ROOT/seal["corpora"]["ontology_focus_fit"]["path"]),key=lambda r:r["id"]); validation=sorted(read(PROJECT_ROOT/seal["corpora"]["ontology_focus_validation"]["path"]),key=lambda r:r["id"]); fi=np.asarray([ids.index(r["id"]) for r in fit]); vi=np.asarray([ids.index(r["id"]) for r in validation])
 attempt.parent.mkdir(parents=True,exist_ok=True); attempt.write_text(json.dumps({"schema_version":38,"status":"started","validation_evaluations":1,"features_lock_sha256":file_sha256(feature_lock_path)},indent=2,sort_keys=True)+"\n"); started=time.perf_counter()
 bundle={"hidden":npz["candidate_hidden"],"margin":npz["candidate_margin"],"mask":npz["candidate_mask"],"deterministic":npz["deterministic_index"]}; fit_bundle={k:v[fi] for k,v in bundle.items()}; val_bundle={k:v[vi] for k,v in bundle.items()}; targets=np.asarray([r["target"]["focus_candidate_index"] for r in fit],dtype=np.int64); groups=[r["oracle_metadata"]["surface_family"] for r in fit]; config=impl["config_payload"]
 reports=cross_validate(fit_bundle,targets,groups,config["selection"]["alphas"]); selected=select_method(reports,config["selection"]["methods"])
 # Reconstruct the already selected V37 outer-operation readout from its locked fit features.
 v37_fl_path=PROJECT_ROOT/impl["v37_features_lock"]
 if file_sha256(v37_fl_path)!=impl["v37_features_lock_sha256"]: raise RuntimeError("V37 feature lock changed")
 v37_fl=json.loads(v37_fl_path.read_text()); v37_seal=json.loads((PROJECT_ROOT/v37_fl["corpus_seal"]).read_text()); v37_rows=sorted(read(PROJECT_ROOT/v37_seal["corpora"]["fit"]["path"]),key=lambda r:r["id"]); v37_npz=np.load(PROJECT_ROOT/v37_fl["feature_artifact"]); v37_ids=v37_npz["record_ids"].tolist(); v37_i=np.asarray([v37_ids.index(r["id"]) for r in v37_rows]); operations=impl["v32_config_payload"]["sharedHead"]["outerOperationClasses"]; op_targets=np.asarray([operations.index(r["target"]["factorization"]["outer_operation"]) for r in v37_rows]); op_model=make_ridge(1000.0); op_model.fit(v37_npz["direct_operation_hidden"][v37_i],op_targets); op_indices=op_model.predict(npz["operation_hidden"][vi]); op_predictions=[operations[int(i)] for i in op_indices]
 candidates=[[c for c in extract_literal_candidates(r)] for r in validation]; methods={}; method_predictions={}
 for method in METHODS:
  eligible=[r for r in reports if r["method"]==method]; choice=select_method(eligible,config["selection"]["methods"]); indices=predict_method(method,choice["alpha"],fit_bundle,targets,val_bundle); method_predictions[method]=indices; methods[method]={"fit_choice":choice,"validation":score(validation,indices,candidates,op_predictions,impl["v32_config_payload"])}
 indices=method_predictions[selected["method"]]; metrics=methods[selected["method"]]["validation"]; qualified=qualification(metrics,config); deterministic_qualified=qualification(methods["deterministic_discourse_parser"]["validation"],config)
 if qualified["passed"]: decision="ontology_focus_parser_qualified_preregister_semantic_confirmation_only"
 elif deterministic_qualified["checks"]["focus"] and deterministic_qualified["checks"]["lexical_sign"]: decision="ontology_focus_parser_succeeds_but_frozen_operation_blocks_full_gate"
 else: decision="ontology_focus_parser_failed_compare_stronger_frozen_grounder"
 output.mkdir(parents=True,exist_ok=False); pred_path=output/"predictions.jsonl"; pred_path.write_text("".join(json.dumps({"id":r["id"],"selected_candidate_index":int(i),"outer_operation":op},sort_keys=True,separators=(",",":"))+"\n" for r,i,op in zip(validation,indices,op_predictions,strict=True)))
 result={"schema_version":38,"experiment":config["experiment"],"features_lock":str(feature_lock_path.relative_to(PROJECT_ROOT)),"features_lock_sha256":file_sha256(feature_lock_path),"evaluation_number":1,"focus_selection":selected,"cv_reports":reports,"methods":methods,"selected_validation":metrics,"qualification":qualified,"deterministic_qualification":deterministic_qualified,"decision":decision,"predictions":str(pred_path.relative_to(PROJECT_ROOT)),"predictions_sha256":file_sha256(pred_path),"runtime_seconds":time.perf_counter()-started,"data_access":{"focus_cv_fits":50,"final_focus_fits":1,"fixed_operation_refits":1,"fit_records_used":240,"validation_records_scored":240,"validation_evaluations":1,"selection_on_validation":0,"model_forward_passes":0,"v32_evaluation_records_read":0,"v28_runs":0,"adapter_training_runs":0},"authorization":{"preregister_fresh_semantic_confirmation":qualified["passed"],"construct_confirmation":False,"end_to_end_relational_suite":False,"v32_evaluation":False,"v28":False,"adapter_training":False,"change_backbone":False}}
 result_path=output/"result.json"; result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); state=json.loads(attempt.read_text()); state.update({"status":"completed","result_sha256":file_sha256(result_path)}); attempt.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
