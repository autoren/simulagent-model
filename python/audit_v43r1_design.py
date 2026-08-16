#!/usr/bin/env python3
"""Audit the V43r1 measurement-repair design."""
from __future__ import annotations
import argparse, json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v43r1-graph-measurement-repair.json"); p.add_argument("--output",default="outputs/v43r1-graph-measurement-repair/design-audit.json"); a=p.parse_args()
    config_path=(PROJECT_ROOT/a.config).resolve(); output=(PROJECT_ROOT/a.output).resolve(); c=json.loads(config_path.read_text()); errors=[]
    outcome_path=PROJECT_ROOT/c["sourceV43OutcomeLock"]; seal_path=PROJECT_ROOT/c["sourceV43CorpusSeal"]; diagnostic_path=PROJECT_ROOT/c["sourcePostHocDiagnostic"]
    outcome=json.loads(outcome_path.read_text()); diagnostic=json.loads(diagnostic_path.read_text())
    failed=[key for key,value in outcome["gate_checks"].items() if not value]
    if outcome["qualification_passed"] or failed != ["state_graph_exact"]: errors.append("V43 failure is not isolated to the registered graph gate")
    if diagnostic["canonical_row_set_exact"]!=1.0 or diagnostic["semantic_content_mismatches"]!=0 or diagnostic["duplicate_free"]!=1.0: errors.append("Post-hoc evidence does not support an ordering-only repair")
    if diagnostic["sealed_result_sha256"]!=outcome["result_sha256"]: errors.append("Diagnostic is not bound to the frozen V43 result")
    allowed=set(c["registeredChange"]["allowedImplementationChanges"])
    if allowed!={"graph_comparator","repair_reporting_and_audit_plumbing"}: errors.append("V43r1 repair scope is too broad")
    if c["immutableInputs"]["newCorpusConstruction"] or c["immutableInputs"]["newModelAccess"]: errors.append("V43r1 introduces new data or model access")
    if any((PROJECT_ROOT/path).exists() for path in ("configs/v43r1-design-lock.json","configs/v43r1-implementation-lock.json","configs/v43r1-outcome-lock.json","outputs/v43r1-graph-measurement-repair/rescore")): errors.append("V43r1 downstream artifact exists before design lock")
    audit={"schema_version":"43r1","experiment":"v43r1_design_audit","passed":not errors,"decision":"authorize_v43r1_design_lock" if not errors else "reject_v43r1_design","errors":errors,"config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"source_v43_outcome_lock":str(outcome_path.relative_to(PROJECT_ROOT)),"source_v43_outcome_lock_sha256":file_sha256(outcome_path),"source_v43_corpus_seal":str(seal_path.relative_to(PROJECT_ROOT)),"source_v43_corpus_seal_sha256":file_sha256(seal_path),"source_post_hoc_diagnostic":str(diagnostic_path.relative_to(PROJECT_ROOT)),"source_post_hoc_diagnostic_sha256":file_sha256(diagnostic_path),"checks":{"sole_failed_v43_gate":failed==["state_graph_exact"],"canonical_diagnostic_exact":diagnostic["canonical_row_set_exact"]==1.0,"original_v43_remains_immutable":c["firewall"]["V43ArtifactModification"]=="forbidden","non_final":c["firewall"]["finalEvaluation"]=="forbidden"},"data_access":{"repair_rescores":0,"model_forward_passes":0,"adapter_training_runs":0}}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
    if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
