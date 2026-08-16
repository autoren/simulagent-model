#!/usr/bin/env python3
"""Audit the isolated V46 oracle stochastic-transition design."""
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v46-oracle-stochastic-transitions.json"); p.add_argument("--output",default="outputs/v46-oracle-stochastic-transitions/design-audit.json"); a=p.parse_args(); config_path=(PROJECT_ROOT/a.config).resolve(); output=(PROJECT_ROOT/a.output).resolve(); c=json.loads(config_path.read_text()); errors=[]; source_path=PROJECT_ROOT/c["sourceV45OutcomeLock"]; source=json.loads(source_path.read_text()) if source_path.is_file() else {}
 if not source.get("qualification_passed") or not source.get("authorization",{}).get("preregister_stochastic_transition_foundation"): errors.append("V45 does not authorize V46 preregistration")
 boundary=c["claimBoundary"]
 if not boundary["oracleDistributionValuedSupport"] or boundary["sampledOutcomeEstimation"] or boundary["probabilityCalibrationFromFiniteTrials"] or boundary["languageGrounding"] or boundary["activeInterventionSelection"]: errors.append("V46 does not isolate oracle probability semantics")
 semantics=c["probabilitySemantics"]
 if semantics["arithmetic"]!="exact_reduced_rational" or semantics["probabilityVocabulary"]!=["1/4","1/2","3/4"] or not semantics["massNormalizationRequired"]: errors.append("V46 probability semantics are not exact and finite")
 contract=c["supportAndQueryContract"]
 if not contract["supportProvidesExactPostActionWorldDistributions"] or not contract["queriesRequireExactPostActionWorldDistributions"] or contract["sampledRealizations"]!=0 or contract["partialInitialStates"]: errors.append("V46 oracle support/query contract is confounded")
 pop=c["population"]
 if pop["mechanics"]!=40 or pop["mechanicsPerFamily"]*len(pop["families"])!=40 or pop["fitMechanics"]+pop["developmentEvaluationMechanics"]!=40: errors.append("V46 population quotas are inconsistent")
 gates=c["gates"]
 if gates["maximumMeanTrajectoryTotalVariation"]!=0.0 or any(gates[key]>=1.0 for key in ("maximumUniformizedExactDistributionMatch","maximumMapExactDistributionMatch","maximumLiteralLookupExactDistributionMatch")): errors.append("V46 exactness or inadequacy gates are invalid")
 if c["nextStageIfPassed"]["stage"]!="sampled_transition_probability_estimation" or not c["nextStageIfPassed"]["requiresFreshPopulation"] or not c["nextStageIfPassed"]["requiresCalibrationAndProperScoring"]: errors.append("V46 does not separate the empirical next stage")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v46-design-lock.json","configs/v46-implementation-lock.json","data/v46-oracle-stochastic-transitions","outputs/v46-oracle-stochastic-transitions/development")): errors.append("V46 downstream artifact exists before design lock")
 audit={"schema_version":46,"experiment":"v46_design_audit","passed":not errors,"decision":"authorize_v46_design_lock" if not errors else "repair_v46_design","errors":errors,"config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"source_v45_outcome_lock":str(source_path.relative_to(PROJECT_ROOT)),"source_v45_outcome_lock_sha256":file_sha256(source_path) if source_path.is_file() else None,"checks":{"oracle_distribution_semantics_isolated":boundary["oracleDistributionValuedSupport"] and not boundary["sampledOutcomeEstimation"],"exact_rational_arithmetic":semantics["arithmetic"]=="exact_reduced_rational","finite_probability_vocabulary":len(semantics["probabilityVocabulary"])==3,"sampled_trials_forbidden":c["firewall"]["sampledOutcomeEstimation"]=="forbidden","probability_ablations_registered":set(c["comparisons"])=={"exactWeightedVersionSpaceExecutor","uniformizedSupportExecutor","mapDeterminizedExecutor","literalDistributionLookup","sampleFrequencyEstimator"},"non_final":c["firewall"]["finalEvaluation"]=="forbidden"},"data_access":{"development_mechanics_constructed":0,"oracle_development_runs":0,"sampled_realizations":0,"model_forward_passes":0,"adapter_training_runs":0}}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
