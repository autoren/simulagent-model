#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v38_focus_parser import corpus_hash
def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v38-implementation-lock.json"); p.add_argument("--output",default="outputs/v38-ontology-anchored-focus-parser/corpus-audit.json"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); lock=json.loads(lock_path.read_text()); base=PROJECT_ROOT/"data/v38-ontology-anchored-focus-parser"; errors=[]; artifacts={}; rows={}
 for name in ("ontology_focus_fit","ontology_focus_validation"):
  path=base/f"{name}.jsonl"; rows[name]=read(path); artifacts[name]={"path":str(path.relative_to(PROJECT_ROOT)),"records":len(rows[name]),"sha256":file_sha256(path)}
  if len(rows[name])!=240 or corpus_hash(rows[name])!=lock["expected_corpus_sha256"][name]: errors.append(f"V38 {name} differs from lock")
 evidence={r["agent_input"]["evidence_text"] for r in rows["ontology_focus_fit"]}&{r["agent_input"]["evidence_text"] for r in rows["ontology_focus_validation"]}; templates={r["oracle_metadata"]["normalized_template"] for r in rows["ontology_focus_fit"]}&{r["oracle_metadata"]["normalized_template"] for r in rows["ontology_focus_validation"]}
 if evidence or templates: errors.append("V38 split overlap")
 if any("target" in r["agent_input"] for values in rows.values() for r in values): errors.append("V38 target exposed in agent input")
 audit={"schema_version":38,"experiment":"v38_corpus_audit","passed":not errors,"decision":"authorize_v38_corpus_seal" if not errors else "repair_v38_corpus","errors":errors,"implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"artifacts":artifacts,"overlap_checks":{"exact_evidence_overlap":len(evidence),"normalized_template_overlap":len(templates)},"data_access":{"model_forward_passes":0,"fit_runs":0,"validation_evaluations":0,"v32_evaluation_records_read":0,"v28_runs":0}}
 output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
