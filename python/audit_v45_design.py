#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v45-delayed-language-grounding.json"); p.add_argument("--output",default="outputs/v45-delayed-language-grounding/design-audit.json"); a=p.parse_args(); config_path=(PROJECT_ROOT/a.config).resolve(); output=(PROJECT_ROOT/a.output).resolve(); c=json.loads(config_path.read_text()); errors=[]; outcome_path=PROJECT_ROOT/c["sourceV44OutcomeLock"]; seal_path=PROJECT_ROOT/c["sourceV44CorpusSeal"]; outcome=json.loads(outcome_path.read_text()) if outcome_path.is_file() else {}; seal=json.loads(seal_path.read_text()) if seal_path.is_file() else {}
 if not outcome.get("qualification_passed") or not outcome.get("authorization",{}).get("preregister_delayed_language_grounding"): errors.append("V44 does not authorize V45 preregistration")
 paired=c["pairedDesign"]; metrics=outcome.get("metrics",{})
 if (paired["mechanics"],paired["querySequences"],paired["waitCounterfactualPairs"])!=(metrics.get("mechanics"),metrics.get("queries"),metrics.get("wait_counterfactual_pairs")): errors.append("V45 paired counts do not match frozen V44")
 impl_path=PROJECT_ROOT/seal.get("implementation_lock","missing"); impl=json.loads(impl_path.read_text()) if impl_path.is_file() else {}
 if paired["supportSequences"]!=impl.get("expected_counts",{}).get("support_sequences"): errors.append("V45 support count does not match V44")
 if not paired["reuseSealedV44MechanicsAndCases"] or not paired["noV44TargetDrivenSelection"] or c["frozenReasoning"]["reasonerModification"]!="forbidden": errors.append("V45 does not isolate the representation boundary")
 if c["languageInterface"]["openParaphrase"] or c["firewall"]["stochasticEffectsInV45"]!="forbidden": errors.append("V45 expands beyond declared deterministic delayed language")
 if c["nextAxisIfPassed"]["axis"]!="stochastic_transition_effects" or not c["nextAxisIfPassed"]["activeSelectionStillDeferred"]: errors.append("V45 does not isolate the next axis")
 gates=c["gates"]
 if any(gates[key]>=1.0 for key in ("maximumCollapsedDelayFinalExact","maximumEndFlushFinalExact","maximumLiteralLanguageLookupFinalExact")): errors.append("V45 controls are not required to fail")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v45-design-lock.json","configs/v45-implementation-lock.json","data/v45-delayed-language-grounding","outputs/v45-delayed-language-grounding/development")): errors.append("V45 downstream artifact exists before design lock")
 audit={"schema_version":45,"experiment":"v45_design_audit","passed":not errors,"decision":"authorize_v45_design_lock" if not errors else "repair_v45_design","errors":errors,"config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"source_v44_outcome_lock":str(outcome_path.relative_to(PROJECT_ROOT)),"source_v44_outcome_lock_sha256":file_sha256(outcome_path) if outcome_path.is_file() else None,"source_v44_corpus_seal":str(seal_path.relative_to(PROJECT_ROOT)),"source_v44_corpus_seal_sha256":file_sha256(seal_path) if seal_path.is_file() else None,"checks":{"paired_representation_isolation":paired["reuseSealedV44MechanicsAndCases"],"v44_reasoner_frozen":c["frozenReasoning"]["reasonerModification"]=="forbidden","wait_language_registered":c["languageInterface"]["waitExtension"]=="declared_zero_argument_wait_cue_and_exact_step_production","canonical_graph_metric_registered":c["languageInterface"]["stateGraphComparator"]=="frozen_v43r1_canonical_duplicate_safe_comparator","safety_suite_registered":all(c["safetyChallenges"].values()),"non_final":c["firewall"]["finalEvaluation"]=="forbidden"},"data_access":{"v44_records_read":0,"paired_development_runs":0,"model_forward_passes":0,"adapter_training_runs":0}}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
