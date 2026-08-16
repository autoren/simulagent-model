#!/usr/bin/env python3
"""Run the single sealed V49 passive-partial-observation development evaluation."""
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
from v49_belief import (
    conditional_suffix_distribution,
    decimal_map,
    effective_count,
    full_evidence,
    map_latent_predictive,
    mechanic_registry,
    posterior_uncertainty,
    prefix_configurations,
    query_predictive,
    support_posterior,
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


def tv(prediction, truth):
    keys = set(prediction) | set(truth)
    return float(sum(abs(Decimal(prediction.get(key, 0)) - Decimal(truth.get(key, 0))) for key in keys) / 2)


def log_loss(prediction, outcomes):
    values = [float(prediction.get(key, 0)) for key in outcomes]
    return math.inf if any(value <= 0 for value in values) else -mean([math.log(value) for value in values])


def brier(prediction, outcomes):
    keys = set(prediction) | set(outcomes)
    return mean([
        sum((float(prediction.get(key, 0)) - (1 if key == observed else 0)) ** 2 for key in keys)
        for observed in outcomes
    ])


def uniformized(prediction):
    return {key: Decimal(1) / Decimal(len(prediction)) for key in prediction}


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
            result += len(selected) / len(pairs) * abs(mean([row[0] for row in selected]) - mean([row[1] for row in selected]))
    return result


def bootstrap(values, seed=4943):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.mean(array[rng.integers(0, len(array), size=(10000, len(array)))], axis=1)
    return {
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def filtered_configuration_entropy(registry, support_weights, query, evidence, prefix_length):
    masses = {}
    world = initial_world(query)
    for mechanic, support_weight in zip(registry, support_weights, strict=True):
        evidence_mass, configurations = prefix_configurations(
            mechanic["program"], query["entities"], world, query["actions"][:prefix_length], evidence
        )
        if not evidence_mass or not support_weight:
            continue
        for key, configuration in configurations.items():
            mass = support_weight * Decimal(configuration["mass"].numerator) / Decimal(configuration["mass"].denominator)
            masses[f"{mechanic['key']}|{key}"] = mass
    total = sum(masses.values(), Decimal(0))
    normalized = [float(value / total) for value in masses.values()]
    return -sum(value * math.log(value) for value in normalized if value)


def merge_supports(record):
    references = {
        row["id"]: row for row in record["reference"]["matched_fully_observed_support_interventions"]
    }
    return [{**row, **{
        "full_trajectory_catalog": references[row["id"]]["full_trajectory_catalog"],
        "realized_full_trajectory_ids": references[row["id"]]["realized_full_trajectory_ids"],
    }} for row in record["agent_input"]["support_interventions"]]


def exact_full_baseline(registry, full_weights, target, query, compatible_full):
    world = initial_world(query)
    prefix_length = query["query_prefix_length"]
    prefix_mass: dict[str, Fraction] = {}
    for trajectory_key, mass in compatible_full.items():
        prefix_key = canonical_json(json.loads(trajectory_key)[:prefix_length])
        prefix_mass[prefix_key] = prefix_mass.get(prefix_key, Fraction(0)) + mass
    total = sum(prefix_mass.values(), Fraction(0))
    weighted_tv = 0.0
    predictions = {}
    truths = {}
    for prefix_key, mass in prefix_mass.items():
        prefix = json.loads(prefix_key)
        evidence = full_evidence(prefix, prefix_length)
        prediction, _, _ = query_predictive(
            registry, full_weights, query["entities"], world, query["actions"], evidence, prefix_length
        )
        _, target_truth = conditional_suffix_distribution(
            target["program"], query["entities"], world, query["actions"], evidence, prefix_length
        )
        predictions[prefix_key] = prediction
        truths[prefix_key] = decimal_map(target_truth)
        weighted_tv += float(mass / total) * tv(prediction, decimal_map(target_truth))
    return weighted_tv, predictions, truths


def evaluate_record(record, registry):
    target_index = next(index for index, row in enumerate(registry) if row["key"] == record["target"]["program_key"])
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
    support_keys = {row["structural_key"] for row in record["agent_input"]["support_interventions"]}
    query_rows = []
    calibration_pairs = []
    for query in record["agent_input"]["queries"]:
        oracle = oracle_by_id[query["id"]]
        world = initial_world(query)
        prefix_length = query["query_prefix_length"]
        evidence = query["masked_prefix_observations"]
        truth_fraction = truth_map(oracle["true_partial_conditional_suffix_distribution"])
        truth = decimal_map(truth_fraction)
        prediction, query_weights, program_conditionals = query_predictive(
            registry, partial_weights, query["entities"], world, query["actions"], evidence, prefix_length
        )
        oracle_prediction, oracle_weights, oracle_conditionals = query_predictive(
            [target], [Decimal(1)], query["entities"], world, query["actions"], evidence, prefix_length
        )
        collapsed = map_latent_predictive(
            registry, partial_weights, query["entities"], world, query["actions"], evidence, prefix_length
        )
        latest_evidence = [[] for _ in range(prefix_length - 1)] + [evidence[-1]]
        history_ablated, _, _ = query_predictive(
            registry, partial_weights, query["entities"], world, query["actions"], latest_evidence, prefix_length
        )
        uniform = uniformized(prediction)
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
        full_tv, full_predictions, _ = exact_full_baseline(
            registry, full_weights, target, query, compatible_full
        )
        full_log_values = []
        for trajectory in heldout_full:
            prefix_key = canonical_json(trajectory[:prefix_length])
            suffix_key = canonical_json(trajectory[prefix_length:])
            full_log_values.append(-math.log(float(full_predictions[prefix_key].get(suffix_key, 0))))
        full_log_loss = mean(full_log_values)

        diagnostics = posterior_uncertainty(prediction, query_weights, program_conditionals)
        diagnostics["filtered_configuration_entropy"] = filtered_configuration_entropy(
            registry, partial_weights, query, evidence, prefix_length
        )
        diagnostics["target_program_posterior_after_prefix"] = float(query_weights[target_index])
        diagnostics["oracle_program_predictive_entropy"] = posterior_uncertainty(
            oracle_prediction, oracle_weights, oracle_conditionals
        )["predictive_entropy"]
        primary_log_loss = log_loss(prediction, heldout_suffix)
        query_rows.append({
            "id": query["id"],
            "family": record["construction_family"],
            "probability": record["oracle_metadata"]["probability"],
            "timing": record["oracle_metadata"]["timing"],
            "sequence_length": query["sequence_length"],
            "visible_fraction": query["visible_fraction"],
            "query_prefix_length": prefix_length,
            "partial_tv": tv(prediction, truth),
            "oracle_program_partial_tv": tv(oracle_prediction, truth),
            "full_tv": full_tv,
            "partial_log_loss": primary_log_loss,
            "full_log_loss": full_log_loss,
            "brier": brier(prediction, heldout_suffix),
            "map_latent_log_loss": log_loss(collapsed, heldout_suffix),
            "history_ablated_log_loss": log_loss(history_ablated, heldout_suffix),
            "uniformized_log_loss": log_loss(uniform, heldout_suffix),
            "predictive_normalized": abs(sum(prediction.values(), Decimal(0)) - Decimal(1)) < Decimal("1e-80"),
            "belief_normalized": abs(sum(query_weights, Decimal(0)) - Decimal(1)) < Decimal("1e-80"),
            "finite_log_loss": math.isfinite(primary_log_loss),
            "target_latent_continuation_retained": all(prediction.get(value, 0) > 0 for value in heldout_suffix),
            "literal_lookup": query["structural_key"] in support_keys,
            "diagnostics": diagnostics,
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


def aggregate(records):
    queries = [query for record in records for query in record["queries"]]
    mechanic_tvs = [mean([query["partial_tv"] for query in record["queries"]]) for record in records]
    mechanic_log_deltas = [
        mean([query["partial_log_loss"] - query["full_log_loss"] for query in record["queries"]])
        for record in records
    ]
    result = {
        "mechanics": len(records),
        "queries": len(queries),
        "likelihood_normalization": mean([record["likelihood_normalized"] for record in records]),
        "belief_normalization": mean([query["belief_normalized"] for query in queries]),
        "predictive_normalization": mean([query["predictive_normalized"] for query in queries]),
        "finite_log_loss_rate": mean([query["finite_log_loss"] for query in queries]),
        "target_latent_continuation_retention": mean([
            query["target_latent_continuation_retained"] for query in queries
        ]),
        "mean_conditional_latent_suffix_tv": mean([query["partial_tv"] for query in queries]),
        "oracle_program_partial_mean_tv": mean([query["oracle_program_partial_tv"] for query in queries]),
        "heldout_conditional_suffix_log_loss": mean([query["partial_log_loss"] for query in queries]),
        "heldout_brier": mean([query["brier"] for query in queries]),
        "calibration_error": mean([record["calibration_error"] for record in records]),
        "map_schema_recovery": mean([record["map_schema_recovered"] for record in records]),
        "mean_target_program_posterior": mean([record["target_program_posterior"] for record in records]),
        "probability_parameter_mae": mean([record["probability_mae"] for record in records]),
        "matched_full_mean_tv": mean([query["full_tv"] for query in queries]),
        "matched_full_log_loss": mean([query["full_log_loss"] for query in queries]),
        "partial_minus_full_mean_tv": mean([query["partial_tv"] - query["full_tv"] for query in queries]),
        "partial_minus_full_log_loss": mean([
            query["partial_log_loss"] - query["full_log_loss"] for query in queries
        ]),
        "map_latent_collapse_log_loss_disadvantage": mean([
            query["map_latent_log_loss"] - query["partial_log_loss"] for query in queries
        ]),
        "observation_history_ablation_log_loss_disadvantage": mean([
            query["history_ablated_log_loss"] - query["partial_log_loss"] for query in queries
        ]),
        "uniformized_log_loss_disadvantage": mean([
            query["uniformized_log_loss"] - query["partial_log_loss"] for query in queries
        ]),
        "literal_masked_trace_lookup_coverage": mean([query["literal_lookup"] for query in queries]),
        "every_family_mean_tv": grouped_mean(records, "partial_tv", "family"),
        "every_probability_mean_tv": grouped_mean(records, "partial_tv", "probability"),
        "every_timing_mean_tv": grouped_mean(records, "partial_tv", "timing"),
        "every_sequence_length_mean_tv": grouped_mean(records, "partial_tv", "sequence_length"),
        "every_visible_fraction_mean_tv": grouped_mean(records, "partial_tv", "visible_fraction"),
        "every_query_prefix_length_mean_tv": grouped_mean(records, "partial_tv", "query_prefix_length"),
        "belief_diagnostics": {
            field: mean([query["diagnostics"][field] for query in queries])
            for field in (
                "filtered_configuration_entropy", "predictive_entropy", "expected_within_program_entropy",
                "program_suffix_mutual_information", "effective_program_count",
                "target_program_posterior_after_prefix", "oracle_program_predictive_entropy",
            )
        },
        "mechanic_cluster_bootstrap_mean_tv": bootstrap(mechanic_tvs, 4943),
        "mechanic_cluster_bootstrap_partial_minus_full_log_loss": bootstrap(mechanic_log_deltas, 4949),
    }
    return result


def qualification(metrics, gates):
    checks = {
        "likelihood_normalization": metrics["likelihood_normalization"] >= gates["minimumLikelihoodNormalization"],
        "belief_normalization": metrics["belief_normalization"] >= gates["minimumBeliefNormalization"],
        "predictive_normalization": metrics["predictive_normalization"] >= gates["minimumPredictiveNormalization"],
        "finite_log_loss": metrics["finite_log_loss_rate"] >= gates["minimumFiniteLogLossRate"],
        "target_latent_retention": metrics["target_latent_continuation_retention"] >= gates["minimumTargetLatentContinuationRetention"],
        "oracle_filter": metrics["oracle_program_partial_mean_tv"] <= gates["maximumOracleProgramPartialMeanTv"],
        "primary_tv": metrics["mean_conditional_latent_suffix_tv"] <= gates["maximumPrimaryMeanConditionalLatentTv"],
        "every_family_tv": max(metrics["every_family_mean_tv"].values()) <= gates["maximumEveryFamilyMeanConditionalLatentTv"],
        "calibration": metrics["calibration_error"] <= gates["maximumCalibrationError"],
        "map_schema": metrics["map_schema_recovery"] >= gates["minimumMapSchemaRecovery"],
        "target_program": metrics["mean_target_program_posterior"] >= gates["minimumMeanTargetProgramPosterior"],
        "probability_mae": metrics["probability_parameter_mae"] <= gates["maximumProbabilityParameterMeanAbsoluteError"],
        "partial_full_tv": metrics["partial_minus_full_mean_tv"] <= gates["maximumPartialMinusFullMeanTv"],
        "partial_full_log_loss": metrics["partial_minus_full_log_loss"] <= gates["maximumPartialMinusFullLogLoss"],
        "map_latent_collapse_inadequate": metrics["map_latent_collapse_log_loss_disadvantage"] >= gates["minimumMapLatentCollapseLogLossDisadvantageNats"],
        "history_ablation_inadequate": metrics["observation_history_ablation_log_loss_disadvantage"] >= gates["minimumObservationHistoryAblationLogLossDisadvantageNats"],
        "uniformized_inadequate": metrics["uniformized_log_loss_disadvantage"] >= gates["minimumUniformizedLogLossDisadvantageNats"],
        "literal_lookup_inadequate": metrics["literal_masked_trace_lookup_coverage"] <= gates["maximumLiteralMaskedTraceLookupCoverage"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v49-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v49-passive-partial-observation/development")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "development-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V49 development already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V49 implementation changed: {path}")
    records = []
    for artifact in seal["corpora"].values():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError("V49 sealed corpus changed")
        records.extend(read(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 49,
        "status": "started",
        "development_run": 1,
        "corpus_seal_sha256": file_sha256(seal_path),
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    registry = mechanic_registry()
    details = [evaluate_record(record, registry) for record in sorted(records, key=lambda row: row["id"])]
    all_metrics = aggregate(details)
    evaluation_metrics = aggregate([row for row in details if row["split"] == "development_evaluation"])
    q = qualification(all_metrics, implementation["config_payload"]["gates"])
    if q["passed"]:
        decision = "passive_partial_observation_pass_preregister_language_composition"
    elif not q["checks"]["oracle_filter"]:
        decision = "repair_latent_world_or_delayed_queue_filter_semantics"
    elif not q["checks"]["map_latent_collapse_inadequate"] or not q["checks"]["history_ablation_inadequate"]:
        decision = "prediction_may_pass_without_demonstrated_persistent_belief_use"
    else:
        decision = "revisit_identifiability_or_passive_evidence_coverage_under_masking"
    output.mkdir(parents=True)
    mechanic_path = output / "mechanic-metrics.jsonl"
    mechanic_path.write_text("".join(canonical_json(row) + "\n" for row in details))
    result = {
        "schema_version": 49,
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
