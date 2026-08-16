#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v41_confirmation import corpus_hash,old_program_keys
def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--implementation-lock",default="configs/v41-implementation-lock.json"); p.add_argument("--output",default="outputs/v41-relational-mechanic-confirmation/corpus-audit.json"); a=p.parse_args(); lock_path=(PROJECT_ROOT/a.implementation_lock).resolve(); output=(PROJECT_ROOT/a.output).resolve(); lock=json.loads(lock_path.read_text()); path=PROJECT_ROOT/"data/v41-relational-mechanic-confirmation/relational_confirmation.jsonl"; rows=read(path); errors=[]
 if len(rows)!=40 or corpus_hash(rows)!=lock["expected_corpus_sha256"]: errors.append("V41 corpus differs from implementation lock")
 if {row["target"]["program_key"] for row in rows}&old_program_keys(lock["config_payload"]): errors.append("V41 corpus reuses a V22 target program")
 counts={"mechanics":len(rows),"support_scenes":sum(len(r["agent_input"]["support_traces"]) for r in rows),"query_scenes":sum(len(r["agent_input"]["queries"]) for r in rows),"language_clauses":sum(len(s["evidence_packets"]) for r in rows for s in r["agent_input"]["support_traces"]+r["agent_input"]["queries"])}
 if counts!=lock["expected_counts"]: errors.append("V41 materialized counts differ from lock")
 if any("target" in r["agent_input"] or "oracle_grounding" in r["agent_input"] or "language_reference" in r["agent_input"] for r in rows): errors.append("V41 oracle data appears inside agent input")
 if file_sha256(PROJECT_ROOT/lock["frozen_compiler"])!=lock["frozen_compiler_sha256"] or file_sha256(PROJECT_ROOT/lock["frozen_semantic_kernel"])!=lock["frozen_semantic_kernel_sha256"]: errors.append("V41 frozen component changed")
 audit={"schema_version":41,"experiment":"v41_corpus_audit","passed":not errors,"decision":"authorize_v41_corpus_seal" if not errors else "reject_v41_corpus","errors":errors,"implementation_lock":str(lock_path.relative_to(PROJECT_ROOT)),"implementation_lock_sha256":file_sha256(lock_path),"artifact":{"path":str(path.relative_to(PROJECT_ROOT)),"records":len(rows),"sha256":file_sha256(path)},"structural_checks":{**counts,"families":dict(Counter(r["construction_family"] for r in rows)),"confirmation_records_scored":0},"data_access":{"confirmation_scoring_runs":0,"model_forward_passes":0,"v22r2_evaluation_records_read":0,"v28_runs":0}}
 output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
