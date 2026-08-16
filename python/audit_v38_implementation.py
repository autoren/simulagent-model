#!/usr/bin/env python3
"""Audit the complete V38 implementation before construction or model access."""

from __future__ import annotations
import argparse, copy, json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from generate_v38_focus_parser import build_population, corpus_hash
from v38_focus_parser import candidate_prompt, extract_literal_candidates


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--design-lock",default="configs/v38-ontology-anchored-focus-parser-lock.json"); parser.add_argument("--output",default="outputs/v38-ontology-anchored-focus-parser/implementation-audit.json"); args=parser.parse_args()
    design_path=(PROJECT_ROOT/args.design_lock).resolve(); output=(PROJECT_ROOT/args.output).resolve(); design=json.loads(design_path.read_text()); config=design["config_payload"]; errors=[]
    if not design["authorization"]["write_implementation"]: errors.append("V38 design does not authorize implementation")
    v32_path=PROJECT_ROOT/"configs/v32-factorized-semantics.json"; v32=json.loads(v32_path.read_text())
    fit=build_population(config,v32,"ontology_focus_fit"); validation=build_population(config,v32,"ontology_focus_validation")
    exact_overlap={row["agent_input"]["evidence_text"] for row in fit}&{row["agent_input"]["evidence_text"] for row in validation}
    template_overlap={row["oracle_metadata"]["normalized_template"] for row in fit}&{row["oracle_metadata"]["normalized_template"] for row in validation}
    if exact_overlap or template_overlap: errors.append("V38 fit/validation language overlaps")
    candidate_views=sum(row["oracle_metadata"]["grounded_literal_candidates"] for row in fit+validation)
    operation_views=len(fit)+len(validation); forwards=candidate_views+operation_views
    if (candidate_views,operation_views,forwards)!=(800,480,1280): errors.append("V38 forward budget mismatch")
    row=validation[0]; candidate=extract_literal_candidates(row)[0]; changed=copy.deepcopy(row); changed["target"]={"sentinel":True}
    if candidate_prompt(row,candidate)!=candidate_prompt(changed,candidate): errors.append("V38 candidate prompt depends on target")
    forbidden=(PROJECT_ROOT/"configs/v38-implementation-lock.json",PROJECT_ROOT/"data/v38-ontology-anchored-focus-parser",PROJECT_ROOT/"outputs/v38-ontology-anchored-focus-parser/features",PROJECT_ROOT/"outputs/v38-ontology-anchored-focus-parser/evaluation")
    if any(path.exists() for path in forbidden): errors.append("V38 artifact exists before implementation lock")
    audit={"schema_version":38,"experiment":"v38_implementation_audit","passed":not errors,"decision":"authorize_v38_implementation_lock" if not errors else "repair_v38_implementation","errors":errors,"design_lock":str(design_path.relative_to(PROJECT_ROOT)),"design_lock_sha256":file_sha256(design_path),"dry_run":{"fit_records":len(fit),"validation_records":len(validation),"candidate_views":candidate_views,"operation_views":operation_views,"backbone_forward_passes":forwards,"fit_corpus_sha256":corpus_hash(fit),"validation_corpus_sha256":corpus_hash(validation)},"overlap_checks":{"exact_evidence_overlap":len(exact_overlap),"normalized_template_overlap":len(template_overlap)},"data_access":{"model_forward_passes":0,"fit_runs":0,"validation_evaluations":0,"v32_evaluation_records_read":0,"v28_runs":0}}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True))
    if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
