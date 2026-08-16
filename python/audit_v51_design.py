#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v51-simulation-based-calibration.json")
    parser.add_argument("--output", default="outputs/v51-simulation-based-calibration/design-audit.json")
    args = parser.parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config = json.loads(config_path.read_text())
    errors = []
    source_path = PROJECT_ROOT / config["sourceV50r1OutcomeLock"]
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    metrics = source.get("metrics", {}).get("all_mechanics", {})
    interpretation = config["v50r1Interpretation"]
    source_matches = (
        source.get("qualification_passed") is True
        and metrics.get("complete_history_condition_matched_regret") == interpretation["completeHistoryConditionMatchedRegret"]
        and metrics.get("mean_complete_history_conditional_suffix_tv") == interpretation["meanCompleteHistoryConditionalSuffixTv"]
        and metrics.get("oracle_history_dependent_query_fraction") == interpretation["oracleHistoryDependentQueryFraction"]
        and metrics.get("latest_only_log_loss_disadvantage") == interpretation["latestOnlyLogLossDisadvantage"]
        and metrics.get("partial_minus_full_condition_matched_regret") == interpretation["partialMinusFullConditionMatchedRegret"]
        and metrics.get("raw_partial_minus_full_log_loss_non_gating") == interpretation["rawInformationValueGapNonGating"]
        and metrics.get("raw_log_loss_vs_oracle_entropy_gap_discrepancy") == interpretation["measurementIdentityDiscrepancy"]
    )
    if not source_matches:
        errors.append("V51 interpretation is not bound to sealed V50r1")
    boundary = config["claimBoundary"]
    if not (
        boundary["exactInference"] and boundary["simulationBasedCalibration"]
        and boundary["independentExactReference"] and not boundary["approximateParticleInference"]
        and not boundary["activeInterventionSelection"] and not boundary["rewardOrPlanning"]
        and not boundary["languageGrounding"] and not boundary["modelAccess"]
        and not boundary["adapterTraining"] and not boundary["finalEvaluation"]
    ):
        errors.append("V51 does not isolate exact inference calibration")
    simulation = config["simulation"]
    if (
        simulation["replications"] < 2000
        or simulation["posteriorDrawsPerReplication"] + 1 != simulation["rankSupportSize"]
        or simulation["rankSupportSize"] % simulation["rankBins"]
        or simulation["expectedRanksPerBin"] != simulation["replications"] // simulation["rankBins"]
        or simulation["expectedRanksPerBin"] < 5
    ):
        errors.append("V51 SBC rank design is underpowered or inconsistent")
    paths = config["inferencePaths"]
    if (
        paths["primary"] == paths["independent"]
        or len(paths["comparisonTargets"]) < 4
        or paths["selectionOnCalibrationReplications"]
        or paths["perReplicationOracleChoices"]
    ):
        errors.append("V51 lacks an independent frozen exact comparison")
    gates = config["gates"]
    if (
        gates["maximumExactPathTv"] > 1e-12
        or gates["minimumRankChiSquarePValue"] < 0.001
        or gates["maximumAbsoluteRankBinZ"] > 4.5
        or gates["maximumAbsoluteCoverageZ"] > 4.5
        or gates["minimumBugControlsRejected"] < 2
    ):
        errors.append("V51 calibration or sensitivity gates are too weak")
    downstream = (
        "configs/v51-design-lock.json",
        "configs/v51-implementation-lock.json",
        "data/v51-simulation-based-calibration",
        "outputs/v51-simulation-based-calibration/calibration",
    )
    if any((PROJECT_ROOT / path).exists() for path in downstream):
        errors.append("V51 downstream artifact exists before design lock")
    audit = {
        "schema_version": 51,
        "experiment": "v51_design_audit",
        "passed": not errors,
        "decision": "authorize_v51_design_lock" if not errors else "repair_v51_design",
        "errors": errors,
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": file_sha256(config_path),
        "source_v50r1_outcome_lock": str(source_path.relative_to(PROJECT_ROOT)),
        "source_v50r1_outcome_lock_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "checks": {
            "v50r1_result_bound": source_matches,
            "exact_sbc_isolated": boundary["exactInference"] and boundary["simulationBasedCalibration"],
            "independent_exact_path_required": paths["primary"] != paths["independent"],
            "rank_bins_well_populated": simulation["expectedRanksPerBin"] >= 5,
            "discrete_ties_randomized": config["sbc"]["discreteTieHandling"].startswith("randomized_insertion_rank"),
            "bug_sensitivity_gated": gates["minimumBugControlsRejected"] >= 2,
            "no_approximation_active_selection_language_or_model": not boundary["approximateParticleInference"]
            and not boundary["activeInterventionSelection"] and not boundary["languageGrounding"]
            and not boundary["modelAccess"],
            "non_final": config["firewall"]["finalEvaluation"] == "forbidden",
        },
        "data_access": {
            "calibration_replications_constructed": 0,
            "calibration_runs": 0,
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
