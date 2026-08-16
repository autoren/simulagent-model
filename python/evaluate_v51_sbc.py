#!/usr/bin/env python3
"""Run the single sealed V51 simulation-based-calibration evaluation."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import Counter
from decimal import Decimal, localcontext

from scipy.stats import chi2

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v51_sbc import (
    batch_inference,
    categorical_sample,
    distribution_tv,
    independent_inference,
    mechanic_registry,
    randomized_rank,
    sequence_tv,
)


QUANTITIES = (
    "program_ordinal",
    "probability_ordinal",
    "configuration_ordinal",
    "program_posterior_probability",
    "configuration_posterior_probability",
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def joint_key_for_truth(inference, target_program_index, true_configuration_key):
    matches = [
        key for key, metadata in inference["metadata"].items()
        if metadata["program_index"] == target_program_index
        and metadata["configuration_key"] == true_configuration_key
    ]
    if len(matches) != 1:
        raise RuntimeError("V51 simulated joint latent is not uniquely retained")
    return matches[0]


def rank_seed(config, replication, condition, quantity):
    return int(sha256_text(
        f"v51-rank|{config['simulation']['tieBreakSeed']}|{replication}|{condition}|{quantity}"
    ), 16)


def draw_seed(config, replication, condition, draw):
    return int(sha256_text(
        f"v51-draw|{config['simulation']['posteriorDrawSeed']}|{replication}|{condition}|{draw}"
    ), 16)


def ranks_for_inference(inference, record, config, condition):
    target_index = record["target_program_index"]
    true_configuration = record["query"]["true_configuration_key"]
    true_joint = joint_key_for_truth(inference, target_index, true_configuration)
    draws = [
        categorical_sample(inference["joint"], draw_seed(config, record["replication"], condition, draw))
        for draw in range(config["simulation"]["posteriorDrawsPerReplication"])
    ]
    true_metadata = inference["metadata"][true_joint]
    true_values = {
        "program_ordinal": true_metadata["program_ordinal"],
        "probability_ordinal": true_metadata["probability_ordinal"],
        "configuration_ordinal": true_metadata["configuration_key"],
        "program_posterior_probability": inference["query_program"][target_index],
        "configuration_posterior_probability": inference["configuration"].get(
            true_configuration, Decimal(0)
        ),
    }
    draw_values = {quantity: [] for quantity in QUANTITIES}
    for key in draws:
        metadata = inference["metadata"][key]
        draw_values["program_ordinal"].append(metadata["program_ordinal"])
        draw_values["probability_ordinal"].append(metadata["probability_ordinal"])
        draw_values["configuration_ordinal"].append(metadata["configuration_key"])
        draw_values["program_posterior_probability"].append(
            inference["query_program"][metadata["program_index"]]
        )
        draw_values["configuration_posterior_probability"].append(
            inference["configuration"][metadata["configuration_key"]]
        )
    return {
        quantity: randomized_rank(
            true_values[quantity], draw_values[quantity],
            rank_seed(config, record["replication"], condition, quantity),
        )
        for quantity in QUANTITIES
    }


def map_inference(inference):
    selected = min(inference["joint"], key=lambda key: (-inference["joint"][key], key))
    metadata = inference["metadata"][selected]
    program = [Decimal(0) for _ in inference["query_program"]]
    program[metadata["program_index"]] = Decimal(1)
    return {
        **inference,
        "joint": {selected: Decimal(1)},
        "query_program": program,
        "configuration": {metadata["configuration_key"]: Decimal(1)},
    }


def evaluate_replication(record, registry, config):
    batch = batch_inference(registry, record["supports"], record["query"])
    independent = independent_inference(registry, record["supports"], record["query"])
    exact_agreement = {
        "support_program_tv": sequence_tv(batch["support_program"], independent["support_program"]),
        "query_program_tv": sequence_tv(batch["query_program"], independent["query_program"]),
        "joint_configuration_tv": distribution_tv(batch["joint"], independent["joint"]),
        "suffix_predictive_tv": distribution_tv(batch["suffix"], independent["suffix"]),
    }
    latest_query = copy.deepcopy(record["query"])
    latest_query["observations"] = [
        *([[]] * (latest_query["prefix_length"] - 1)),
        latest_query["observations"][-1],
    ]
    controls = {
        "tempered_likelihood": independent_inference(
            registry, record["supports"], record["query"], likelihood_power=2
        ),
        "latest_only_query": independent_inference(registry, record["supports"], latest_query),
        "map_posterior": map_inference(independent),
    }
    with localcontext() as context:
        context.prec = 100
        normalization = all([
            abs(sum(batch["support_program"], Decimal(0)) - 1) < Decimal("1e-80"),
            abs(sum(batch["query_program"], Decimal(0)) - 1) < Decimal("1e-80"),
            abs(sum(batch["joint"].values(), Decimal(0)) - 1) < Decimal("1e-80"),
            abs(sum(batch["suffix"].values(), Decimal(0)) - 1) < Decimal("1e-80"),
            abs(sum(independent["support_program"], Decimal(0)) - 1) < Decimal("1e-80"),
            abs(sum(independent["joint"].values(), Decimal(0)) - 1) < Decimal("1e-80"),
            abs(sum(independent["suffix"].values(), Decimal(0)) - 1) < Decimal("1e-80"),
        ])
    return {
        "id": record["id"],
        "replication": record["replication"],
        "family": record["family"],
        "probability": record["probability"],
        "timing": record["timing"],
        "sequence_length": record["query"]["sequence_length"],
        "prefix_length": record["query"]["prefix_length"],
        "normalization": normalization,
        "exact_agreement": exact_agreement,
        "ranks": ranks_for_inference(independent, record, config, "primary"),
        "control_ranks": {
            name: ranks_for_inference(value, record, config, name)
            for name, value in controls.items()
        },
    }


def rank_diagnostics(details, field, config):
    replications = config["simulation"]["replications"]
    bins = config["simulation"]["rankBins"]
    support = config["simulation"]["rankSupportSize"]
    expected = replications / bins
    bin_probability = 1 / bins
    bin_sd = math.sqrt(replications * bin_probability * (1 - bin_probability))
    coverage = {}
    histograms, p_values, max_z = {}, {}, {}
    for quantity in QUANTITIES:
        ranks = [row[field][quantity] for row in details]
        counts = [0 for _ in range(bins)]
        for rank in ranks:
            counts[min(bins - 1, rank * bins // support)] += 1
        statistic = sum((count - expected) ** 2 / expected for count in counts)
        histograms[quantity] = counts
        p_values[quantity] = float(chi2.sf(statistic, bins - 1))
        max_z[quantity] = max(abs(count - expected) / bin_sd for count in counts)
        coverage[quantity] = {}
        for level in config["sbc"]["coverageLevels"]:
            included_ranks = round(level * support)
            lower = (support - included_ranks) // 2
            upper = lower + included_ranks
            expected_coverage = included_ranks / support
            observed = sum(lower <= rank < upper for rank in ranks) / len(ranks)
            sd = math.sqrt(expected_coverage * (1 - expected_coverage) / len(ranks))
            coverage[quantity][str(level)] = {
                "observed": observed,
                "expected": expected_coverage,
                "z": (observed - expected_coverage) / sd,
            }
    return {
        "histograms": histograms,
        "chi_square_p_values": p_values,
        "maximum_absolute_rank_bin_z_by_quantity": max_z,
        "minimum_chi_square_p_value": min(p_values.values()),
        "maximum_absolute_rank_bin_z": max(max_z.values()),
        "coverage": coverage,
        "maximum_absolute_coverage_z": max(
            abs(cell["z"])
            for quantity in coverage.values() for cell in quantity.values()
        ),
    }


def control_diagnostics(details, name, config):
    shaped = [{"control": row["control_ranks"][name]} for row in details]
    diagnostics = rank_diagnostics(shaped, "control", config)
    gates = config["gates"]
    rejected = (
        diagnostics["minimum_chi_square_p_value"] < gates["minimumRankChiSquarePValue"]
        or diagnostics["maximum_absolute_rank_bin_z"] > gates["maximumAbsoluteRankBinZ"]
        or diagnostics["maximum_absolute_coverage_z"] > gates["maximumAbsoluteCoverageZ"]
    )
    return {**diagnostics, "rejected": rejected}


def aggregate(details, config):
    exact_fields = (
        "support_program_tv", "query_program_tv", "joint_configuration_tv", "suffix_predictive_tv"
    )
    primary = rank_diagnostics(details, "ranks", config)
    controls = {
        name: control_diagnostics(details, name, config)
        for name in ("tempered_likelihood", "latest_only_query", "map_posterior")
    }
    return {
        "replications": len(details),
        "completed_replication_fraction": len(details) / config["simulation"]["replications"],
        "normalization_rate": sum(row["normalization"] for row in details) / len(details),
        "maximum_exact_path_tv": max(
            row["exact_agreement"][field] for row in details for field in exact_fields
        ),
        "maximum_exact_path_tv_by_target": {
            field: max(row["exact_agreement"][field] for row in details) for field in exact_fields
        },
        "primary_sbc": primary,
        "bug_controls": controls,
        "bug_controls_rejected": sum(value["rejected"] for value in controls.values()),
    }


def qualification(metrics, gates):
    checks = {
        "completed_replications": metrics["completed_replication_fraction"] >= gates["minimumCompletedReplicationFraction"],
        "normalization": metrics["normalization_rate"] >= gates["minimumNormalizationRate"],
        "exact_path_agreement": metrics["maximum_exact_path_tv"] <= gates["maximumExactPathTv"],
        "rank_chi_square": metrics["primary_sbc"]["minimum_chi_square_p_value"] >= gates["minimumRankChiSquarePValue"],
        "rank_bin_envelope": metrics["primary_sbc"]["maximum_absolute_rank_bin_z"] <= gates["maximumAbsoluteRankBinZ"],
        "coverage": metrics["primary_sbc"]["maximum_absolute_coverage_z"] <= gates["maximumAbsoluteCoverageZ"],
        "bug_sensitivity": metrics["bug_controls_rejected"] >= gates["minimumBugControlsRejected"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v51-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v51-simulation-based-calibration/calibration")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "calibration-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V51 calibration already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V51 implementation changed: {path}")
    corpus_path = PROJECT_ROOT / seal["corpus"]["path"]
    if file_sha256(corpus_path) != seal["corpus"]["sha256"]:
        raise RuntimeError("V51 sealed calibration corpus changed")
    records = read(corpus_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 51,
        "status": "started",
        "calibration_run": 1,
        "corpus_seal_sha256": file_sha256(seal_path),
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    registry = mechanic_registry()
    config = implementation["config_payload"]
    details = [evaluate_replication(row, registry, config) for row in records]
    metrics = aggregate(details, config)
    q = qualification(metrics, config["gates"])
    if q["passed"]:
        decision = "exact_inference_calibrated_authorize_scalable_particle_inference_preregistration"
    elif not q["checks"]["exact_path_agreement"]:
        decision = "repair_exact_inference_semantics_before_approximation"
    elif not q["checks"]["bug_sensitivity"]:
        decision = "increase_sbc_test_quantity_sensitivity_before_interpreting_calibration"
    else:
        decision = "repair_exact_inference_or_sbc_implementation"
    output.mkdir(parents=True)
    details_path = output / "replication-metrics.jsonl"
    details_path.write_text("".join(canonical_json(row) + "\n" for row in details))
    result = {
        "schema_version": 51,
        "experiment": config["experiment"],
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "calibration_run_number": 1,
        "metrics": metrics,
        "qualification": q,
        "decision": decision,
        "replication_metrics": str(details_path.relative_to(PROJECT_ROOT)),
        "replication_metrics_sha256": file_sha256(details_path),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "calibration_runs": 1,
            "replications_scored": len(details),
            "selection_on_calibration_replications": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
        "authorization": {
            "preregister_scalable_particle_inference": q["passed"],
            "construct_particle_population": False,
            "active_intervention_selection": False,
            "reward_or_planning": False,
            "language_grounding": False,
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
