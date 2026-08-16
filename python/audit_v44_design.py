#!/usr/bin/env python3
"""Audit the isolated V44 deterministic-delay design."""
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v44-deterministic-delayed-effects.json"); p.add_argument("--output",default="outputs/v44-deterministic-delayed-effects/design-audit.json"); a=p.parse_args(); config_path=(PROJECT_ROOT/a.config).resolve(); output=(PROJECT_ROOT/a.output).resolve(); c=json.loads(config_path.read_text()); errors=[]; source_path=PROJECT_ROOT/c["sourceV43r1OutcomeLock"]; source=json.loads(source_path.read_text()) if source_path.is_file() else {}
 if not source.get("qualification_passed") or not source.get("authorization",{}).get("preregister_deterministic_delayed_effects"): errors.append("V43r1 does not authorize V44 preregistration")
 scope=c["scope"]
 if not scope["oracleFirst"] or not scope["deterministicDelayedEffects"] or scope["languageGrounding"]: errors.append("V44 does not isolate oracle deterministic delay")
 if any(scope[key] for key in ("stochasticEffects","activeInterventionSelection","openConceptInventory")): errors.append("V44 confounds deterministic delay with another new axis")
 tick=c["tickSemantics"]
 if tick["sequenceEnd"]!="pending_events_are_not_automatically_flushed" or tick["sameTargetSameTickConflict"]!="forbidden_by_generator_and_program_validator": errors.append("V44 timing or conflict semantics are underspecified")
 pop=c["population"]
 if pop["mechanics"]!=40 or pop["mechanicsPerFamily"]*len(pop["families"])!=40 or pop["fitMechanics"]+pop["developmentEvaluationMechanics"]!=40: errors.append("V44 population quotas are inconsistent")
 causal=c["causalRequirements"]
 if "wait" not in pop["actions"] or not all(causal[key] for key in ("everyMechanicHasDelaySensitiveQuery","everyFamilyHasWaitPlacementCounterfactual","pairedQueriesUseSameInitialStateAndActionMultiset","waitPlacementChangesAtLeastOneRegisteredFinalObservation")) or causal["supportQueryStructuralOverlap"]!=0: errors.append("V44 lacks registered causal timing requirements")
 gates=c["gates"]
 if gates["maximumCollapsedDelayFinalExact"]>=1.0 or gates["maximumEndFlushFinalExact"]>=1.0 or gates["maximumLiteralLookupFinalExact"]>=1.0: errors.append("V44 inadequacy controls are not required to fail")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v44-design-lock.json","configs/v44-implementation-lock.json","data/v44-deterministic-delayed-effects","outputs/v44-deterministic-delayed-effects/development")): errors.append("V44 downstream artifact exists before design lock")
 audit={"schema_version":44,"experiment":"v44_design_audit","passed":not errors,"decision":"authorize_v44_design_lock" if not errors else "repair_v44_design","errors":errors,"config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"source_v43r1_outcome_lock":str(source_path.relative_to(PROJECT_ROOT)),"source_v43r1_outcome_lock_sha256":file_sha256(source_path) if source_path.is_file() else None,"checks":{"deterministic_delay_isolated":scope["deterministicDelayedEffects"] and not any(scope[k] for k in ("stochasticEffects","activeInterventionSelection","openConceptInventory")),"tick_order_explicit":len(tick["phaseOrder"])==4,"pending_events_not_flushed":tick["sequenceEnd"]=="pending_events_are_not_automatically_flushed","wait_counterfactuals_registered":c["causalRequirements"]["everyFamilyHasWaitPlacementCounterfactual"],"timing_controls_registered":set(c["comparisons"])=={"queuedVersionSpaceExecutor","collapsedDelayExecutor","endFlushExecutor","literalSequenceLookup","jointNeuralChallenger"},"non_final":c["firewall"]["finalEvaluation"]=="forbidden"},"data_access":{"development_mechanics_constructed":0,"oracle_development_runs":0,"model_forward_passes":0,"adapter_training_runs":0}}
 output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
