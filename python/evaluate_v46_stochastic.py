#!/usr/bin/env python3
"""Run the single sealed V46 oracle stochastic-transition development."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from fractions import Fraction
from typing import Sequence

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json
from v42_stateful import compatible_worlds
from v46_stochastic import (
    distribution_key, execute_distribution, map_determinized, mechanic_registry,
    total_variation, uniformized,
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate(values: Sequence[bool]):
    return sum(values) / len(values) if values else 0.0


def initial_world(case):
    worlds = compatible_worlds(case["initial_state"])
    if len(worlds) != 1:
        raise ValueError("V46 requires complete initial states")
    return worlds[0]


def predict(program, case, control="weighted"):
    trajectory = execute_distribution(program, case["entities"], initial_world(case), case["actions"])
    if control == "uniformized":
        return uniformized(trajectory)
    if control == "map":
        return map_determinized(trajectory)
    return trajectory


def normalized(trajectory):
    for step in trajectory:
        mass = sum(
            (Fraction(row["mass"]["numerator"], row["mass"]["denominator"]) for row in step),
            Fraction(0),
        )
        if mass != 1:
            return False
    return True


def filter_version_space(record, registry, control="weighted"):
    survivors = list(registry)
    prefixes = []
    target_key = record["target"]["program_key"]
    for support in record["agent_input"]["support_sequences"]:
        observed = distribution_key(support["observed_step_distributions"])
        survivors = [
            mechanic for mechanic in survivors
            if distribution_key(predict(mechanic["program"], support, control)) == observed
        ]
        prefixes.append({
            "prefix": len(prefixes) + 1,
            "version_space": len(survivors),
            "target_retained": any(mechanic["key"] == target_key for mechanic in survivors),
        })
    return survivors, prefixes


def version_space_prediction(survivors, query, control):
    predictions = [predict(mechanic["program"], query, control) for mechanic in survivors]
    keys = {distribution_key(trajectory) for trajectory in predictions}
    return predictions[0] if len(keys) == 1 else None


def trajectory_tv(predicted, target):
    if predicted is None or len(predicted) != len(target):
        return Fraction(1)
    return sum((total_variation(left, right) for left, right in zip(predicted, target, strict=True)), Fraction(0)) / len(target)


def evaluate_record(record, registry):
    by_key = {mechanic["key"]: mechanic for mechanic in registry}
    target_key = record["target"]["program_key"]
    target_program = by_key[target_key]["program"]
    supports = record["agent_input"]["support_sequences"]
    queries = record["agent_input"]["queries"]
    targets = {row["id"]: row["target"] for row in record["oracle_queries"]}
    validation = [distribution_key(predict(target_program, support)) == distribution_key(support["observed_step_distributions"]) for support in supports]
    validation.extend(distribution_key(predict(target_program, query)) == distribution_key(targets[query["id"]]) for query in queries)
    mass_checks = [normalized(support["observed_step_distributions"]) for support in supports]
    mass_checks.extend(normalized(targets[query["id"]]) for query in queries)
    weighted, prefixes = filter_version_space(record, registry, "weighted")
    uniform_survivors, _ = filter_version_space(record, registry, "uniformized")
    map_survivors, _ = filter_version_space(record, registry, "map")
    lookup = {support["structural_key"]: support["observed_step_distributions"] for support in supports}
    query_rows = []
    for query in queries:
        target = targets[query["id"]]
        primary = version_space_prediction(weighted, query, "weighted")
        uniform = version_space_prediction(uniform_survivors, query, "uniformized")
        modal = version_space_prediction(map_survivors, query, "map")
        literal = lookup.get(query["structural_key"])
        exact = primary is not None and distribution_key(primary) == distribution_key(target)
        query_rows.append({
            "id": query["id"],
            "family": record["construction_family"],
            "split": record["split"],
            "sequence_length": query["sequence_length"],
            "probability": record["oracle_metadata"]["probability"],
            "timing": record["oracle_metadata"]["timing"],
            "probability_sensitive": query["probability_sensitive"],
            "timing_sensitive": query["timing_sensitive"],
            "mass_normalized": primary is not None and normalized(primary),
            "exact": exact,
            "tv_numerator": trajectory_tv(primary, target).numerator,
            "tv_denominator": trajectory_tv(primary, target).denominator,
            "uniform_exact": uniform is not None and distribution_key(uniform) == distribution_key(target),
            "map_exact": modal is not None and distribution_key(modal) == distribution_key(target),
            "lookup_exact": literal is not None and distribution_key(literal) == distribution_key(target),
        })
    target_retained = any(mechanic["key"] == target_key for mechanic in weighted)
    schema = len(weighted) == 1 and weighted[0]["key"] == target_key
    return {
        "program_validation": all(validation),
        "mass_normalization": all(mass_checks) and all(row["mass_normalized"] for row in query_rows),
        "target_retained": target_retained,
        "schema_recovered": schema,
        "empty_version_space": not weighted,
        "version_space": len(weighted),
        "uniform_version_space": len(uniform_survivors),
        "map_version_space": len(map_survivors),
        "queries": query_rows,
        "prefixes": prefixes,
    }, {
        "id": record["id"],
        "family": record["construction_family"],
        "split": record["split"],
        "probability": record["oracle_metadata"]["probability"],
        "timing": record["oracle_metadata"]["timing"],
        "target_retained": target_retained,
        "schema_recovered": schema,
        "version_space": len(weighted),
        "uniform_version_space": len(uniform_survivors),
        "map_version_space": len(map_survivors),
        "exact_trajectory_rate": rate([row["exact"] for row in query_rows]),
        "uniform_exact_rate": rate([row["uniform_exact"] for row in query_rows]),
        "map_exact_rate": rate([row["map_exact"] for row in query_rows]),
    }


def aggregate(records):
    queries = [query for record in records for query in record["queries"]]

    def grouped(field, metric):
        return {
            str(value): rate([row[metric] for row in queries if row[field] == value])
            for value in sorted({row[field] for row in queries}, key=str)
        }

    tv_values = [Fraction(row["tv_numerator"], row["tv_denominator"]) for row in queries]
    mean_tv = sum(tv_values, Fraction(0)) / len(tv_values)
    return {
        "mechanics": len(records),
        "queries": len(queries),
        "oracle_program_validation": rate([record["program_validation"] for record in records]),
        "mass_normalization": rate([record["mass_normalization"] for record in records]),
        "weighted_target_retention": rate([record["target_retained"] for record in records]),
        "weighted_schema_recovery": rate([record["schema_recovered"] for record in records]),
        "weighted_empty_version_space": rate([record["empty_version_space"] for record in records]),
        "exact_trajectory_distribution_match": rate([row["exact"] for row in queries]),
        "mean_trajectory_total_variation": float(mean_tv),
        "mean_trajectory_total_variation_exact": f"{mean_tv.numerator}/{mean_tv.denominator}",
        "by_family_exact": grouped("family", "exact"),
        "by_sequence_length_exact": grouped("sequence_length", "exact"),
        "by_probability_exact": grouped("probability", "exact"),
        "by_timing_exact": grouped("timing", "exact"),
        "by_split_exact": grouped("split", "exact"),
        "probability_sensitive_exact": grouped("probability_sensitive", "exact"),
        "timing_sensitive_exact": grouped("timing_sensitive", "exact"),
        "uniformized_exact_distribution_match": rate([row["uniform_exact"] for row in queries]),
        "map_exact_distribution_match": rate([row["map_exact"] for row in queries]),
        "literal_lookup_exact_distribution_match": rate([row["lookup_exact"] for row in queries]),
        "median_weighted_version_space": float(np.median([record["version_space"] for record in records])),
        "median_uniformized_version_space": float(np.median([record["uniform_version_space"] for record in records])),
        "median_map_version_space": float(np.median([record["map_version_space"] for record in records])),
    }


def qualification(metrics, gates):
    checks = {
        "oracle_program_validation": metrics["oracle_program_validation"] >= gates["minimumOracleProgramValidation"],
        "mass_normalization": metrics["mass_normalization"] >= gates["minimumMassNormalization"],
        "weighted_target_retention": metrics["weighted_target_retention"] >= gates["minimumWeightedTargetRetention"],
        "weighted_schema_recovery": metrics["weighted_schema_recovery"] >= gates["minimumWeightedSchemaRecovery"],
        "weighted_empty_version_space": metrics["weighted_empty_version_space"] <= gates["maximumWeightedEmptyVersionSpace"],
        "exact_trajectory_distribution_match": metrics["exact_trajectory_distribution_match"] >= gates["minimumExactTrajectoryDistributionMatch"],
        "mean_trajectory_total_variation": metrics["mean_trajectory_total_variation"] <= gates["maximumMeanTrajectoryTotalVariation"],
        "every_family_exact": min(metrics["by_family_exact"].values()) >= gates["minimumEveryFamilyExactDistribution"],
        "every_sequence_length_exact": min(metrics["by_sequence_length_exact"].values()) >= gates["minimumEverySequenceLengthExactDistribution"],
        "every_probability_exact": min(metrics["by_probability_exact"].values()) >= gates["minimumEveryProbabilityValueExactDistribution"],
        "uniformized_inadequate": metrics["uniformized_exact_distribution_match"] <= gates["maximumUniformizedExactDistributionMatch"],
        "map_inadequate": metrics["map_exact_distribution_match"] <= gates["maximumMapExactDistributionMatch"],
        "literal_lookup_inadequate": metrics["literal_lookup_exact_distribution_match"] <= gates["maximumLiteralLookupExactDistributionMatch"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v46-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v46-oracle-stochastic-transitions/development")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "development-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V46 development already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V46 implementation changed: {path}")
    records = []
    for artifact in seal["corpora"].values():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError("V46 sealed corpus changed")
        records.extend(read(path))
    records.sort(key=lambda row: row["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 46, "status": "started", "oracle_development_run": 1,
        "corpus_seal_sha256": file_sha256(seal_path),
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    registry = mechanic_registry()
    details, predictions = [], []
    for record in records:
        detail, prediction = evaluate_record(record, registry)
        details.append(detail)
        predictions.append(prediction)
    metrics = aggregate(details)
    qualified = qualification(metrics, implementation["config_payload"]["gates"])
    checks = qualified["checks"]
    if qualified["passed"]:
        decision = "oracle_probability_foundation_pass_preregister_sampled_transition_estimation"
    elif not checks["mass_normalization"] or not checks["exact_trajectory_distribution_match"]:
        decision = "repair_exact_probability_executor"
    elif not checks["weighted_schema_recovery"]:
        decision = "revise_stochastic_dsl_or_support_identifiability"
    else:
        decision = "redesign_probability_or_structural_controls"
    output.mkdir(parents=True, exist_ok=False)
    predictions_path = output / "mechanic-predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions))
    result = {
        "schema_version": 46,
        "experiment": implementation["config_payload"]["experiment"],
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "oracle_development_run_number": 1,
        "metrics": metrics,
        "qualification": qualified,
        "decision": decision,
        "predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "predictions_sha256": file_sha256(predictions_path),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "oracle_development_runs": 1, "mechanics_scored": len(records), "sampled_realizations": 0,
            "selection_on_development_evaluation": 0, "model_forward_passes": 0, "adapter_training_runs": 0,
        },
        "authorization": {
            "preregister_sampled_transition_estimation": qualified["passed"],
            "construct_sampled_transition_population": False,
            "language_grounding": False, "active_intervention_selection": False,
            "open_ontology": False, "final_evaluation": False, "model_access": False,
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
