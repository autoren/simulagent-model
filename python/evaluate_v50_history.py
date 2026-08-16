#!/usr/bin/env python3
"""Run the single sealed V50 history-dependent belief evaluation."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from decimal import Decimal
from fractions import Fraction

import numpy as np

from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v49_belief import conditional_suffix_distribution, full_evidence
from v50_belief import (
    decimal_map,
    effective_count,
    map_latent_predictive,
    mechanic_registry,
    query_predictive,
    safe_query_predictive,
    support_posterior,
    total_variation,
    trajectory_map,
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def initial_world(case):
    return {row["atom"]: row["allowed_values"][0] for row in case["initial_state"]}


def truth_map(rows):
    return {
        canonical_json(row["suffix"]): Fraction(row["mass"]["numerator"], row["mass"]["denominator"])
        for row in rows
    }


def log_loss(prediction, outcomes):
    values = [float(prediction.get(key, 0)) for key in outcomes]
    return math.inf if any(value <= 0 for value in values) else -mean([math.log(value) for value in values])


def calibration_error(pairs):
    if not pairs:
        return 0.0
    result = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        selected = [
            row for row in pairs
            if lower <= row[0] < (upper if index < 9 else upper + 1e-12)
        ]
        if selected:
            result += len(selected) / len(pairs) * abs(
                mean([row[0] for row in selected]) - mean([row[1] for row in selected])
            )
    return result


def bootstrap(values, seed=5063):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.mean(array[rng.integers(0, len(array), size=(10000, len(array)))], axis=1)
    return {
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def merge_supports(record):
    references = {
        row["id"]: row for row in record["reference"]["matched_fully_observed_support_interventions"]
    }
    return [{
        **row,
        "full_trajectory_catalog": references[row["id"]]["full_trajectory_catalog"],
        "realized_full_trajectory_ids": references[row["id"]]["realized_full_trajectory_ids"],
    } for row in record["agent_input"]["support_interventions"]]


def exact_full_baseline(registry, full_weights, target, query, compatible_full):
    world = initial_world(query)
    prefix_length = query["query_prefix_length"]
    prefix_mass: dict[str, Fraction] = {}
    for trajectory_key, mass in compatible_full.items():
        prefix_key = canonical_json(json.loads(trajectory_key)[:prefix_length])
        prefix_mass[prefix_key] = prefix_mass.get(prefix_key, Fraction(0)) + mass
    total = sum(prefix_mass.values(), Fraction(0))
    weighted_tv = 0.0
    predictions, truths = {}, {}
    for prefix_key, mass in prefix_mass.items():
        prefix = json.loads(prefix_key)
        evidence = full_evidence(prefix, prefix_length)
        prediction, _, _ = query_predictive(
            registry, full_weights, query["entities"], world, query["actions"], evidence, prefix_length
        )
        _, target_truth = conditional_suffix_distribution(
            target["program"], query["entities"], world, query["actions"], evidence, prefix_length
        )
        target_decimal = decimal_map(target_truth)
        predictions[prefix_key] = prediction
        truths[prefix_key] = target_decimal
        weighted_tv += float(mass / total) * total_variation(prediction, target_decimal)
    return weighted_tv, predictions, truths


def evaluate_record(record, registry):
    target_index = next(
        index for index, row in enumerate(registry) if row["key"] == record["target"]["program_key"]
    )
    target = registry[target_index]
    supports = merge_supports(record)
    partial_weights = support_posterior(registry, supports, fully_observed=False)
    full_weights = support_posterior(registry, supports, fully_observed=True)
    partial_map = min(range(len(registry)), key=lambda index: (-partial_weights[index], registry[index]["key"]))
    target_probability = float(Fraction(record["oracle_metadata"]["probability"]))
    expected_probability = sum(
        float(weight) * float(Fraction(mechanic["probability"]))
        for weight, mechanic in zip(partial_weights, registry, strict=True)
    )
    oracle_by_id = {row["id"]: row for row in record["oracle_queries"]}
    calibration_pairs = []
    query_rows = []
    for query in record["agent_input"]["queries"]:
        oracle = oracle_by_id[query["id"]]
        world = initial_world(query)
        prefix_length = query["query_prefix_length"]
        evidence = query["masked_prefix_observations"]
        truth_fraction = truth_map(oracle["true_complete_history_conditional_suffix_distribution"])
        truth = decimal_map(truth_fraction)
        prediction, query_weights, _ = query_predictive(
            registry, partial_weights, query["entities"], world, query["actions"], evidence, prefix_length
        )
        oracle_prediction, _, _ = query_predictive(
            [target], [Decimal(1)], query["entities"], world, query["actions"], evidence, prefix_length
        )
        latest_prediction, _, _ = query_predictive(
            registry, partial_weights, query["entities"], world, query["actions"],
            query["latest_only_prefix_observations"], prefix_length,
        )
        shuffled_prediction, _, _ = safe_query_predictive(
            registry, partial_weights, query["entities"], world, query["actions"],
            query["time_shuffled_prefix_observations"], prefix_length,
        )
        collapsed_prediction = map_latent_predictive(
            registry, partial_weights, query["entities"], world, query["actions"], evidence, prefix_length
        )
        catalog = oracle["compatible_full_trajectory_catalog"]
        heldout_full = [catalog[value] for value in oracle["heldout_full_trajectory_ids"]]
        heldout_suffix = [canonical_json(value[prefix_length:]) for value in heldout_full]
        frequencies = Counter(heldout_suffix)
        calibration_pairs.extend(
            (float(prediction.get(key, 0)), frequencies.get(key, 0) / len(heldout_suffix))
            for key in set(prediction) | set(frequencies)
        )

        target_full = trajectory_map(target["program"], query["entities"], world, query["actions"])
        compatible_full = {
            key: mass
            for key, mass in target_full.items()
            if all(
                {row["atom"]: row["value"] for row in json.loads(json.loads(key)[step])}.get(observed["atom"])
                is observed["value"]
                for step, observations in enumerate(evidence)
                for observed in observations
            )
        }
        compatible_total = sum(compatible_full.values(), Fraction(0))
        compatible_full = {key: value / compatible_total for key, value in compatible_full.items()}
        full_tv, full_predictions, full_truths = exact_full_baseline(
            registry, full_weights, target, query, compatible_full
        )
        full_model_values, full_oracle_values = [], []
        for trajectory in heldout_full:
            prefix_key = canonical_json(trajectory[:prefix_length])
            suffix_key = canonical_json(trajectory[prefix_length:])
            full_model_values.append(-math.log(float(full_predictions[prefix_key].get(suffix_key, 0))))
            full_oracle_values.append(-math.log(float(full_truths[prefix_key].get(suffix_key, 0))))

        primary_log_loss = log_loss(prediction, heldout_suffix)
        oracle_log_loss = log_loss(oracle_prediction, heldout_suffix)
        full_model_log_loss = mean(full_model_values)
        full_oracle_log_loss = mean(full_oracle_values)
        query_rows.append({
            "id": query["id"],
            "family": record["construction_family"],
            "probability": record["oracle_metadata"]["probability"],
            "timing": record["oracle_metadata"]["timing"],
            "sequence_length": query["sequence_length"],
            "prefix_length": prefix_length,
            "earlier_evidence_distance": query["earlier_evidence_distance"],
            "construction_mode": query["construction_mode"],
            "complete_history_tv": total_variation(prediction, truth),
            "oracle_program_complete_history_tv": total_variation(oracle_prediction, truth),
            "full_condition_tv": full_tv,
            "complete_history_log_loss": primary_log_loss,
            "oracle_complete_history_log_loss": oracle_log_loss,
            "full_condition_log_loss": full_model_log_loss,
            "oracle_full_condition_log_loss": full_oracle_log_loss,
            "complete_history_condition_matched_regret": primary_log_loss - oracle_log_loss,
            "full_condition_matched_regret": full_model_log_loss - full_oracle_log_loss,
            "latest_only_log_loss": log_loss(latest_prediction, heldout_suffix),
            "time_shuffled_log_loss": log_loss(shuffled_prediction, heldout_suffix),
            "map_latent_log_loss": log_loss(collapsed_prediction, heldout_suffix),
            "oracle_full_history_vs_latest_only_tv": oracle["oracle_full_history_vs_latest_only_tv"],
            "oracle_history_value_kl_nats": oracle["oracle_history_value_kl_nats"],
            "oracle_time_shuffled_kl_nats": oracle["oracle_time_shuffled_kl_nats"],
            "oracle_map_collapse_kl_nats": oracle["oracle_map_collapse_kl_nats"],
            "predictive_normalized": abs(sum(prediction.values(), Decimal(0)) - Decimal(1)) < Decimal("1e-80"),
            "belief_normalized": abs(sum(query_weights, Decimal(0)) - Decimal(1)) < Decimal("1e-80"),
            "finite_log_loss": math.isfinite(primary_log_loss),
            "target_continuation_retained": all(prediction.get(value, 0) > 0 for value in heldout_suffix),
        })
    return {
        "id": record["id"],
        "split": record["split"],
        "family": record["construction_family"],
        "probability": record["oracle_metadata"]["probability"],
        "timing": record["oracle_metadata"]["timing"],
        "likelihood_normalized": True,
        "map_schema_recovered": partial_map == target_index,
        "target_program_posterior": float(partial_weights[target_index]),
        "probability_mae": abs(expected_probability - target_probability),
        "calibration_error": calibration_error(calibration_pairs),
        "effective_program_count_after_support": effective_count(partial_weights),
        "queries": query_rows,
    }


def grouped_mean(records, field, stratum):
    values = sorted({query[stratum] for record in records for query in record["queries"]}, key=str)
    return {
        str(value): mean([
            query[field] for record in records for query in record["queries"] if query[stratum] == value
        ])
        for value in values
    }


def aggregate(records, history_tv_threshold=0.10):
    queries = [query for record in records for query in record["queries"]]
    mechanic_tvs = [mean([query["complete_history_tv"] for query in record["queries"]]) for record in records]
    mechanic_regrets = [
        mean([query["complete_history_condition_matched_regret"] for query in record["queries"]])
        for record in records
    ]
    partial_regret = mean([query["complete_history_condition_matched_regret"] for query in queries])
    full_regret = mean([query["full_condition_matched_regret"] for query in queries])
    raw_gap = mean([query["complete_history_log_loss"] - query["full_condition_log_loss"] for query in queries])
    oracle_entropy_gap = mean([
        query["oracle_complete_history_log_loss"] - query["oracle_full_condition_log_loss"]
        for query in queries
    ])
    return {
        "mechanics": len(records),
        "queries": len(queries),
        "likelihood_normalization": mean([record["likelihood_normalized"] for record in records]),
        "belief_normalization": mean([query["belief_normalized"] for query in queries]),
        "predictive_normalization": mean([query["predictive_normalized"] for query in queries]),
        "finite_log_loss_rate": mean([query["finite_log_loss"] for query in queries]),
        "target_continuation_retention": mean([query["target_continuation_retained"] for query in queries]),
        "mean_complete_history_conditional_suffix_tv": mean([query["complete_history_tv"] for query in queries]),
        "oracle_program_mean_tv": mean([query["oracle_program_complete_history_tv"] for query in queries]),
        "complete_history_log_loss": mean([query["complete_history_log_loss"] for query in queries]),
        "complete_history_condition_matched_regret": partial_regret,
        "calibration_error": mean([record["calibration_error"] for record in records]),
        "oracle_history_dependent_query_fraction": mean([
            query["oracle_full_history_vs_latest_only_tv"] >= history_tv_threshold for query in queries
        ]),
        "mean_oracle_full_history_vs_latest_only_tv": mean([
            query["oracle_full_history_vs_latest_only_tv"] for query in queries
        ]),
        "latest_only_log_loss_disadvantage": mean([
            query["latest_only_log_loss"] - query["complete_history_log_loss"] for query in queries
        ]),
        "time_shuffled_log_loss_disadvantage": mean([
            query["time_shuffled_log_loss"] - query["complete_history_log_loss"] for query in queries
        ]),
        "map_latent_collapse_log_loss_disadvantage": mean([
            query["map_latent_log_loss"] - query["complete_history_log_loss"] for query in queries
        ]),
        "map_schema_recovery": mean([record["map_schema_recovered"] for record in records]),
        "mean_target_program_posterior": mean([record["target_program_posterior"] for record in records]),
        "probability_parameter_mae": mean([record["probability_mae"] for record in records]),
        "partial_condition_regret": partial_regret,
        "full_condition_regret": full_regret,
        "partial_minus_full_condition_matched_regret": partial_regret - full_regret,
        "raw_partial_minus_full_log_loss_non_gating": raw_gap,
        "oracle_conditional_entropy_gap_non_gating": oracle_entropy_gap,
        "raw_log_loss_vs_oracle_entropy_gap_discrepancy": abs(raw_gap - oracle_entropy_gap),
        "every_family_mean_tv": grouped_mean(records, "complete_history_tv", "family"),
        "every_probability_mean_tv": grouped_mean(records, "complete_history_tv", "probability"),
        "every_timing_mean_tv": grouped_mean(records, "complete_history_tv", "timing"),
        "every_sequence_length_mean_tv": grouped_mean(records, "complete_history_tv", "sequence_length"),
        "every_prefix_length_mean_tv": grouped_mean(records, "complete_history_tv", "prefix_length"),
        "every_earlier_evidence_distance_mean_tv": grouped_mean(records, "complete_history_tv", "earlier_evidence_distance"),
        "mechanic_cluster_bootstrap_mean_tv": bootstrap(mechanic_tvs, 5063),
        "mechanic_cluster_bootstrap_condition_matched_regret": bootstrap(mechanic_regrets, 5069),
    }


def qualification(metrics, gates):
    checks = {
        "likelihood_normalization": metrics["likelihood_normalization"] >= gates["minimumLikelihoodNormalization"],
        "belief_normalization": metrics["belief_normalization"] >= gates["minimumBeliefNormalization"],
        "predictive_normalization": metrics["predictive_normalization"] >= gates["minimumPredictiveNormalization"],
        "finite_log_loss": metrics["finite_log_loss_rate"] >= gates["minimumFiniteLogLossRate"],
        "target_continuation_retention": metrics["target_continuation_retention"] >= gates["minimumTargetContinuationRetention"],
        "oracle_filter": metrics["oracle_program_mean_tv"] <= gates["maximumOracleProgramMeanTv"],
        "primary_tv": metrics["mean_complete_history_conditional_suffix_tv"] <= gates["maximumPrimaryMeanTv"],
        "every_family_tv": max(metrics["every_family_mean_tv"].values()) <= gates["maximumEveryFamilyMeanTv"],
        "calibration": metrics["calibration_error"] <= gates["maximumCalibrationError"],
        "condition_matched_regret": metrics["complete_history_condition_matched_regret"] <= gates["maximumMeanConditionMatchedRegretNats"],
        "partial_full_condition_matched_regret": metrics["partial_minus_full_condition_matched_regret"] <= gates["maximumPartialMinusFullConditionMatchedRegretNats"],
        "oracle_history_fraction": metrics["oracle_history_dependent_query_fraction"] >= gates["minimumOracleHistoryDependentQueryFraction"],
        "oracle_history_tv": metrics["mean_oracle_full_history_vs_latest_only_tv"] >= gates["minimumMeanOracleFullHistoryVsLatestOnlyTv"],
        "latest_only_inadequate": metrics["latest_only_log_loss_disadvantage"] >= gates["minimumLatestOnlyLogLossDisadvantageNats"],
        "time_shuffle_inadequate": metrics["time_shuffled_log_loss_disadvantage"] >= gates["minimumTimeShuffledLogLossDisadvantageNats"],
        "map_latent_collapse_inadequate": metrics["map_latent_collapse_log_loss_disadvantage"] >= gates["minimumMapLatentCollapseLogLossDisadvantageNats"],
        "map_schema": metrics["map_schema_recovery"] >= gates["minimumMapSchemaRecovery"],
        "target_program": metrics["mean_target_program_posterior"] >= gates["minimumMeanTargetProgramPosterior"],
        "probability_mae": metrics["probability_parameter_mae"] <= gates["maximumProbabilityParameterMeanAbsoluteError"],
        "measurement_identity": metrics["raw_log_loss_vs_oracle_entropy_gap_discrepancy"] <= gates["maximumRawLogLossVsOracleEntropyGapDiscrepancyNats"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v50-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v50-history-dependent-belief-filtering/development")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "development-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V50 development already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V50 implementation changed: {path}")
    records = []
    for artifact in seal["corpora"].values():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError("V50 sealed corpus changed")
        records.extend(read(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 50,
        "status": "started",
        "development_run": 1,
        "corpus_seal_sha256": file_sha256(seal_path),
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    registry = mechanic_registry()
    details = [evaluate_record(record, registry) for record in sorted(records, key=lambda row: row["id"])]
    threshold = implementation["config_payload"]["historyDependenceContract"]["minimumOracleFullHistoryVsLatestOnlyTv"]
    all_metrics = aggregate(details, threshold)
    evaluation_metrics = aggregate([row for row in details if row["split"] == "development_evaluation"], threshold)
    q = qualification(all_metrics, implementation["config_payload"]["gates"])
    if q["passed"]:
        decision = "history_dependent_belief_filtering_pass_preregister_supported_language_composition"
    elif not q["checks"]["oracle_history_fraction"] or not q["checks"]["oracle_history_tv"]:
        decision = "repair_history_dependent_population_without_interpreting_prediction"
    elif not q["checks"]["oracle_filter"]:
        decision = "repair_world_or_queue_filter_semantics"
    elif not q["checks"]["latest_only_inadequate"] or not q["checks"]["time_shuffle_inadequate"]:
        decision = "prediction_may_pass_without_demonstrated_temporal_evidence_use"
    else:
        decision = "revisit_identifiability_or_condition_matched_scoring"
    output.mkdir(parents=True)
    mechanic_path = output / "mechanic-metrics.jsonl"
    mechanic_path.write_text("".join(canonical_json(row) + "\n" for row in details))
    result = {
        "schema_version": 50,
        "experiment": implementation["config_payload"]["experiment"],
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "development_run_number": 1,
        "metrics": {"all_mechanics": all_metrics, "development_evaluation": evaluation_metrics},
        "qualification": q,
        "decision": decision,
        "mechanic_metrics": str(mechanic_path.relative_to(PROJECT_ROOT)),
        "mechanic_metrics_sha256": file_sha256(mechanic_path),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "development_runs": 1,
            "mechanics_scored": len(records),
            "selection_on_development_evaluation": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
        "authorization": {
            "preregister_partial_observation_language_composition": q["passed"],
            "construct_language_composition_population": False,
            "active_intervention_selection": False,
            "noisy_sensors": False,
            "continuous_probabilities": False,
            "open_ontology": False,
            "final_evaluation": False,
            "model_access": False,
        },
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    state = json.loads(attempt.read_text())
    state.update({"status": "completed", "result_sha256": file_sha256(result_path)})
    attempt.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
