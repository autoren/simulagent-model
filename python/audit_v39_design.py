#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v39-declared-language-compiler.json"); p.add_argument("--output",default="outputs/v39-declared-language-compiler/design-audit.json"); a=p.parse_args(); config_path=(PROJECT_ROOT/a.config).resolve(); output=(PROJECT_ROOT/a.output).resolve(); c=json.loads(config_path.read_text()); outcome_path=PROJECT_ROOT/c["sourceV38OutcomeLock"]; outcome=json.loads(outcome_path.read_text()); errors=[]
 if outcome["scientific_decision"]!="ontology_focus_parser_succeeds_but_frozen_operation_blocks_full_gate": errors.append("V38 does not support V39")
 m=outcome["selected_validation"]
 if any(m[key]!=1.0 for key in ("focus_accuracy","lexical_sign_accuracy","focus_first_accuracy","focus_second_accuracy","exact_opposite_decoy_accuracy","different_atom_decoy_accuracy","worst_surface_family_accuracy")): errors.append("V38 focus parser did not pass every local gate")
 if m["outer_operation_accuracy"]>=0.95: errors.append("V38 does not isolate outer operation as the blocker")
 stage=c["stageAuthorization"]
 if not stage["writeAndAuditImplementation"] or any(stage[k] for k in ("constructEvaluation","scoreEvaluation","preregisterConfirmation")): errors.append("V39 design authorization too broad")
 if any(c["firewall"][k]!="forbidden" for k in ("v32CalibrationOrEvaluationUse","v28Use","adapterTraining","backboneChange","endToEndRelationalSuite","claimExpansionBeyondDeclaredGrammar")): errors.append("V39 firewall incomplete")
 forbidden=(PROJECT_ROOT/"configs/v39-declared-language-compiler-lock.json",PROJECT_ROOT/"configs/v39-implementation-lock.json",PROJECT_ROOT/"data/v39-declared-language-compiler",PROJECT_ROOT/"outputs/v39-declared-language-compiler/evaluation")
 if any(path.exists() for path in forbidden): errors.append("V39 artifact exists before design lock")
 audit={"schema_version":39,"experiment":"v39_design_audit","passed":not errors,"decision":"authorize_v39_design_lock" if not errors else "repair_v39_design","errors":errors,"source":{"config_sha256":file_sha256(config_path),"v38_outcome_sha256":file_sha256(outcome_path),"plan_sha256":file_sha256(PROJECT_ROOT/"docs/v39-declared-language-compiler-plan.md")},"data_access":{"v39_records_constructed":0,"evaluations":0,"v32_evaluation_records_read":0,"v28_runs":0}}
 output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
