#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v50-history-dependent-belief-filtering.json")
    parser.add_argument("--output", default="outputs/v50-history-dependent-belief-filtering/design-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text())
    errors = []
    source_path = PROJECT_ROOT / config["sourceV49OutcomeLock"]
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    metrics = source.get("metrics", {}).get("all_mechanics", {})
    checks = source.get("gate_checks", {})
    interpretation = config["v49Interpretation"]
    source_matches = (
        source.get("scientific_decision") == "prediction_may_pass_without_demonstrated_persistent_belief_use"
        and checks.get("oracle_filter") is True
        and checks.get("primary_tv") is True
        and checks.get("map_latent_collapse_inadequate") is True
        and checks.get("history_ablation_inadequate") is False
        and checks.get("partial_full_log_loss") is False
        and metrics.get("mean_conditional_latent_suffix_tv") == interpretation["meanConditionalLatentSuffixTv"]
        and metrics.get("partial_minus_full_log_loss") == interpretation["partialMinusFullRawLogLoss"]
    )
    if not source_matches:
        errors.append("V50 interpretation is not bound to the sealed V49 result")

    boundary = config["claimBoundary"]
    if not (
        boundary["partialObservation"] and boundary["historyDependenceRequired"]
        and boundary["persistentLatentWorldAndQueue"] and boundary["knownObservationSchedule"]
        and boundary["valueIndependentMasks"] and not boundary["missingCompletelyAtRandom"]
        and not boundary["languageGrounding"] and not boundary["activeInterventionSelection"]
        and not boundary["noisySensors"] and not boundary["continuousProbabilityLearning"]
        and not boundary["openOntology"] and not boundary["modelAccess"]
        and not boundary["adapterTraining"] and not boundary["finalEvaluation"]
    ):
        errors.append("V50 does not isolate history-dependent passive filtering")

    population = config["population"]
    if (
        population["mechanics"] != 48
        or population["mechanicsPerFamily"] * len(population["families"]) != 48
        or population["developmentFitMechanics"] + population["developmentEvaluationMechanics"] != 48
        or len(set(population["probabilityCounts"].values())) != 1
        or max(population["queryEvidencePrefixLengths"]) >= min(population["sequenceLengths"]) + 1
    ):
        errors.append("V50 population quotas are inconsistent")

    contract = config["historyDependenceContract"]
    gates = config["gates"]
    if (
        contract["minimumOracleHistoryDependentQueryFraction"] < 0.8
        or contract["minimumOracleFullHistoryVsLatestOnlyTv"] < 0.1
        or gates["minimumOracleHistoryDependentQueryFraction"] < 0.8
        or gates["minimumMeanOracleFullHistoryVsLatestOnlyTv"] < 0.1
    ):
        errors.append("V50 does not require a genuine history-dependent population")

    repair = config["metricRepair"]
    if (
        not repair["conditionMatchedRegretDefinition"].startswith("model_negative_log_probability_minus_target_program_oracle")
        or repair["rawPartialMinusFullLogLoss"] != "report_non_gating_as_the_empirical_value_of_additional_information"
        or config["firewall"]["rawCrossInformationLogLossAsGate"] != "forbidden"
    ):
        errors.append("V50 does not repair the V49 cross-information scoring error")

    inference = config["inference"]
    if (
        not inference["primary"].startswith("frozen_v49_exact")
        or inference["selectionOnDevelopmentEvaluation"]
        or inference["perEpisodeOracleChoices"]
    ):
        errors.append("V50 inference is not frozen and selection-free")
    if gates["maximumMeanConditionMatchedRegretNats"] > 0.02 or gates[
        "maximumPartialMinusFullConditionMatchedRegretNats"
    ] > 0.01:
        errors.append("V50 condition-matched regret tolerances are too broad")

    downstream = (
        "configs/v50-design-lock.json",
        "configs/v50-implementation-lock.json",
        "data/v50-history-dependent-belief-filtering",
        "outputs/v50-history-dependent-belief-filtering/development",
    )
    if any((PROJECT_ROOT / path).exists() for path in downstream):
        errors.append("V50 downstream artifact exists before design lock")
    audit = {
        "schema_version": 50,
        "experiment": "v50_design_audit",
        "passed": not errors,
        "decision": "authorize_v50_design_lock" if not errors else "repair_v50_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "source_v49_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v49_outcome_lock_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "checks": {
            "v49_result_bound": source_matches,
            "core_positive_preserved": checks.get("oracle_filter") is True and checks.get("primary_tv") is True,
            "failed_history_control_targeted": checks.get("history_ablation_inadequate") is False,
            "invalid_raw_log_loss_gate_removed": config["firewall"]["rawCrossInformationLogLossAsGate"] == "forbidden",
            "condition_matched_regret_gated": gates["maximumMeanConditionMatchedRegretNats"] <= 0.02,
            "oracle_history_construction_gated": contract["minimumOracleHistoryDependentQueryFraction"] >= 0.8,
            "no_language_active_selection_or_training": not boundary["languageGrounding"]
            and not boundary["activeInterventionSelection"] and not boundary["modelAccess"]
            and not boundary["adapterTraining"],
            "non_final": config["firewall"]["finalEvaluation"] == "forbidden",
        },
        "data_access": {
            "history_dependent_mechanics_constructed": 0,
            "sampled_realizations": 0,
            "development_runs": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
