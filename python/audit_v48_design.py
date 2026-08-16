#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v48-stochastic-language-composition.json"); p.add_argument("--output",default="outputs/v48-stochastic-language-composition/design-audit.json"); a=p.parse_args(); config_path=(PROJECT_ROOT/a.config).resolve(); output=(PROJECT_ROOT/a.output).resolve(); c=json.loads(config_path.read_text()); errors=[]; source_path=PROJECT_ROOT/c["sourceV47OutcomeLock"]; source=json.loads(source_path.read_text()) if source_path.is_file() else {}
 if not source.get("qualification_passed") or not source.get("authorization",{}).get("preregister_stochastic_language_composition"): errors.append("V47 does not authorize V48 preregistration")
 frozen=c["frozenComponents"]
 if not frozen["languageCompiler"].startswith("v45_") or not frozen["probabilisticEstimator"].startswith("v47_"): errors.append("V48 does not compose the frozen positive components")
 boundary=c["claimBoundary"]
 if boundary["learnProbabilityWords"] or boundary["openOntology"] or boundary["activeInterventionSelection"] or boundary["languageModelAccess"] or boundary["adapterTraining"]: errors.append("V48 claim is expanded")
 pop=c["population"]
 if pop["mechanics"]!=48 or pop["mechanicsPerFamily"]*len(pop["families"])!=48 or pop["developmentFitMechanics"]+pop["developmentEvaluationMechanics"]!=48 or len(set(pop["probabilityCounts"].values()))!=1: errors.append("V48 population quotas are invalid")
 language=c["languageContract"]
 if not language["freshEntityAliasesPerMechanic"] or not language["exactOrdinalActionOrderRequired"] or not language["unknownAmbiguousMalformedAndDuplicateOrdinalInputsFailClosed"] or not language["probabilityIsInferredFromRepeatedOutcomesNotStatedInLanguage"]: errors.append("V48 language contract is invalid")
 if set(c["conditions"])!={"languagePrimary","matchedSymbolicBaseline","uniformizedProbabilityControl","shuffledActionOrderControl","literalLanguageLookup"}: errors.append("V48 comparisons incomplete")
 gates=c["gates"]
 if any(gates[key]!=1.0 for key in ("minimumClauseParseAccuracy","minimumCanonicalGraphExact","minimumActionCommandAccuracy","minimumActionSequenceExact","minimumFailClosedSafetyAccuracy")): errors.append("V48 supported-language interface is not gated at exactness")
 if gates["maximumLanguageMinusSymbolicMeanTv"]>0.01 or gates["maximumLanguageMinusSymbolicLogLoss"]>0.01: errors.append("V48 composition degradation tolerance too broad")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v48-design-lock.json","configs/v48-implementation-lock.json","data/v48-stochastic-language-composition","outputs/v48-stochastic-language-composition/development")): errors.append("V48 downstream artifact exists")
 audit={"schema_version":48,"experiment":"v48_design_audit","passed":not errors,"decision":"authorize_v48_design_lock" if not errors else "repair_v48_design","errors":errors,"config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"source_v47_outcome_lock":str(source_path.relative_to(PROJECT_ROOT)),"source_v47_outcome_lock_sha256":file_sha256(source_path) if source_path.is_file() else None,"checks":{"frozen_composition":frozen["languageCompiler"].startswith("v45_") and frozen["probabilisticEstimator"].startswith("v47_"),"fresh_balanced_population":pop["programOverlapWithV46OrV47"]==0 and len(set(pop["probabilityCounts"].values()))==1,"probability_not_stated":language["probabilityIsInferredFromRepeatedOutcomesNotStatedInLanguage"],"matched_symbolic_baseline":c["conditions"]["matchedSymbolicBaseline"].startswith("same_programs"),"fail_closed":language["unknownAmbiguousMalformedAndDuplicateOrdinalInputsFailClosed"],"no_model_or_training":not boundary["languageModelAccess"] and not boundary["adapterTraining"],"non_final":c["firewall"]["finalEvaluation"]=="forbidden"},"data_access":{"development_mechanics_constructed":0,"sampled_realizations":0,"development_runs":0,"model_forward_passes":0,"adapter_training_runs":0}}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
