#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/v47-sampled-transition-estimation.json"); p.add_argument("--output",default="outputs/v47-sampled-transition-estimation/design-audit.json"); a=p.parse_args(); config_path=(PROJECT_ROOT/a.config).resolve(); output=(PROJECT_ROOT/a.output).resolve(); c=json.loads(config_path.read_text()); errors=[]; source_path=PROJECT_ROOT/c["sourceV46OutcomeLock"]; source=json.loads(source_path.read_text()) if source_path.is_file() else {}
 if not source.get("qualification_passed") or not source.get("authorization",{}).get("preregister_sampled_transition_estimation"): errors.append("V46 does not authorize V47 preregistration")
 boundary=c["claimBoundary"]
 if not boundary["realizedSupportTrials"] or boundary["oracleDistributionValuedSupport"] or boundary["languageGrounding"] or boundary["activeInterventionSelection"]: errors.append("V47 does not isolate finite-sample estimation")
 pop=c["population"]
 if pop["mechanics"]!=48 or pop["mechanicsPerFamily"]*len(pop["families"])!=48 or pop["developmentFitMechanics"]+pop["developmentEvaluationMechanics"]!=48: errors.append("V47 population quotas are inconsistent")
 if sum(pop["probabilityCounts"].values())!=48 or len(set(pop["probabilityCounts"].values()))!=1: errors.append("V47 probabilities are not balanced")
 if pop["nestedTrialsPerIntervention"]!=[8,32,128] or pop["heldoutTrialsPerQuery"]<32: errors.append("V47 sampling budgets are invalid")
 sampling=c["samplingContract"]
 if not sampling["supportContainsOnlyRealizedTrajectories"] or not sampling["queryAgentInputContainsNoOutcomes"] or not sampling["oracleJointDistributionsScorerOnly"] or sampling["supportQueryStructuralOverlap"]!=0: errors.append("V47 sampling firewall is invalid")
 estimator=c["estimator"]
 if "posterior" not in estimator["primary"] or estimator["selectionOnEvaluation"] or estimator["perMechanicOracleChoices"]: errors.append("V47 estimator is not a fixed uncertainty-preserving estimator")
 metrics=c["metrics"]
 if "heldout_joint_trajectory_log_loss" not in metrics["properScoring"] or "multiclass_brier_score" not in metrics["properScoring"] or metrics["uncertainty"]!="mechanic_cluster_bootstrap_95_percent_interval_with_10000_resamples_and_seed_4747": errors.append("V47 proper scoring or uncertainty is not preregistered")
 gates=c["gatesAt128TrialsPerIntervention"]
 if gates["maximumMeanJointDistributionTotalVariation"]<=0 or gates["minimumMeanTargetProgramPosteriorMass"]>=1 or not gates["requireMeanTvStrictImprovement8To32"] or not gates["requireMeanTvStrictImprovement32To128"]: errors.append("V47 finite-sample gates are invalid")
 if any((PROJECT_ROOT/path).exists() for path in ("configs/v47-design-lock.json","configs/v47-implementation-lock.json","data/v47-sampled-transition-estimation","outputs/v47-sampled-transition-estimation/development")): errors.append("V47 downstream artifact exists before design lock")
 audit={"schema_version":47,"experiment":"v47_design_audit","passed":not errors,"decision":"authorize_v47_design_lock" if not errors else "repair_v47_design","errors":errors,"config":str(config_path.relative_to(PROJECT_ROOT)),"config_sha256":file_sha256(config_path),"source_v46_outcome_lock":str(source_path.relative_to(PROJECT_ROOT)),"source_v46_outcome_lock_sha256":file_sha256(source_path) if source_path.is_file() else None,"checks":{"fresh_population_required":pop["freshProgramOverlapWithV46"]==0,"probabilities_balanced":len(set(pop["probabilityCounts"].values()))==1,"nested_sampling":sampling["nestedSupportTrials"].startswith("first_8"),"joint_likelihood":estimator["likelihoodUnit"].startswith("complete_realized_trajectory"),"proper_scoring":len(metrics["properScoring"])==2,"calibration_registered":len(metrics["calibration"])==1,"no_language_or_active_selection":not boundary["languageGrounding"] and not boundary["activeInterventionSelection"],"non_final":c["firewall"]["finalEvaluation"]=="forbidden"},"data_access":{"sampled_mechanics_constructed":0,"sampled_development_runs":0,"sampled_realizations":0,"model_forward_passes":0,"adapter_training_runs":0}}
 output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n"); print(json.dumps(audit,indent=2,sort_keys=True));
 if not audit["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
