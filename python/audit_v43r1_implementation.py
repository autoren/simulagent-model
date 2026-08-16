#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v43r1_measurement import duplicate_free,graph_equal
def main():
 p=argparse.ArgumentParser(); p.add_argument("--design-lock",default="configs/v43r1-design-lock.json"); p.add_argument("--output",default="outputs/v43r1-graph-measurement-repair/implementation-audit.json"); a=p.parse_args(); design_path=(PROJECT_ROOT/a.design_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); d=json.loads(design_path.read_text()); errors=[]
 for path,key in ((d["source_v43_outcome_lock"],"source_v43_outcome_lock_sha256"),(d["source_v43_corpus_seal"],"source_v43_corpus_seal_sha256"),(d["source_post_hoc_diagnostic"],"source_post_hoc_diagnostic_sha256")):
  if file_sha256(PROJECT_ROOT/path)!=d[key]: errors.append(f"Frozen V43r1 source changed: {path}")
 left=[{"atom":"b","allowed_values":[True]},{"atom":"a","allowed_values":[False]}]; right=list(reversed(left)); permutation=graph_equal(left,right) and graph_equal(right,left); content_sensitive=not graph_equal(left,[{"atom":"a","allowed_values":[True]}]); duplicate_safe=not duplicate_free([left[0],left[0]])
 if not all((permutation,content_sensitive,duplicate_safe)): errors.append("Canonical graph comparator contract failed")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v43r1-implementation-lock.json","configs/v43r1-outcome-lock.json","outputs/v43r1-graph-measurement-repair/rescore")): errors.append("V43r1 downstream artifact exists before implementation lock")
 audit={"schema_version":"43r1","experiment":"v43r1_implementation_audit","passed":not errors,"decision":"authorize_v43r1_implementation_lock" if not errors else "repair_v43r1_implementation","errors":errors,"design_lock":str(design_path.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design_path),"comparator_checks":{"permutation_invariant":permutation,"content_sensitive":content_sensitive,"duplicate_safe":duplicate_safe},"data_access":{"repair_rescores":0,"v43_records_read":0,"model_forward_passes":0,"adapter_training_runs":0}}
 output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
