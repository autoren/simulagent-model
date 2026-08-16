#!/usr/bin/env python3
"""Audit the isolated V42 sequential-foundation design."""
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v42-sequential-state-foundation.json"); p.add_argument("--output",default="outputs/v42-sequential-state-foundation/design-audit.json"); a=p.parse_args(); config_path=(PROJECT_ROOT/a.config).resolve(); output=(PROJECT_ROOT/a.output).resolve(); c=json.loads(config_path.read_text()); errors=[]; source_path=PROJECT_ROOT/c["sourceV41OutcomeLock"]; source=json.loads(source_path.read_text()) if source_path.is_file() else {}
 if not source.get("qualification_passed") or not source.get("authorization",{}).get("begin_architecture_breaking_benchmark"): errors.append("V41 does not authorize architecture-breaking benchmark design")
 s=c["scope"]
 if not s["persistentStateMutation"] or not s["deterministicImmediateEffects"]: errors.append("V42 does not isolate deterministic persistent mutation")
 if any(s[key] for key in ("stochasticEffects","delayedEffects","activeInterventionSelection","openConceptInventory")): errors.append("V42 confounds the isolated sequential axis")
 pop=c["population"]
 if pop["mechanics"]!=40 or pop["mechanicsPerFamily"]*len(pop["families"])!=40 or pop["fitMechanics"]+pop["developmentEvaluationMechanics"]!=40: errors.append("V42 population quotas are inconsistent")
 if c["comparisons"]["frozenV22MemorylessExecutor"]!="registered_inadequacy_control": errors.append("V42 lacks the memoryless inadequacy control")
 if c["gates"]["maximumMemorylessFinalObservationExact"]>=1.0 or c["gates"]["maximumLiteralLookupFinalObservationExact"]>=1.0: errors.append("V42 controls are not required to fail")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v42-design-lock.json","data/v42-sequential-state-foundation","outputs/v42-sequential-state-foundation/development")): errors.append("V42 downstream artifact exists before design lock")
 audit={"schema_version":42,"experiment":"v42_design_audit","passed":not errors,"decision":"authorize_v42_design_lock" if not errors else "repair_v42_design","errors":errors,"config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"source_v41_outcome_lock":str(source_path.relative_to(PROJECT_ROOT)),"source_v41_outcome_lock_sha256":file_sha256(source_path) if source_path.is_file() else None,"checks":{"sequential_mutation_isolated":s["persistentStateMutation"] and not any(s[k] for k in ("stochasticEffects","delayedEffects","activeInterventionSelection","openConceptInventory")),"stateful_and_inadequacy_controls_registered":set(c["comparisons"])=={"statefulVersionSpaceExecutor","frozenV22MemorylessExecutor","literalSequenceLookup","jointNeuralChallenger"},"population_fixed":pop["mechanics"]==40,"oracle_first":s["oracleFirst"] and not s["languageGrounding"],"non_final":c["firewall"]["finalEvaluation"]=="forbidden"},"data_access":{"development_mechanics_constructed":0,"oracle_development_runs":0,"language_model_forward_passes":0,"adapter_training_runs":0}}
 output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
