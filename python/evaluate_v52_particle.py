#!/usr/bin/env python3
"""Run the single sealed V52 particle evaluation."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from decimal import Decimal, localcontext

from scipy.stats import chi2

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v51_sbc import (
    categorical_sample,
    distribution_tv,
    independent_inference,
    randomized_rank,
    sequence_tv,
    sequential_filter,
)
from v52_particle import mechanic_registry, particle_inference, probability_marginal, stream_id


QUANTITIES = (
    "program_ordinal",
    "probability_ordinal",
    "configuration_ordinal",
    "program_posterior_probability",
    "configuration_posterior_probability",
)
CORE_TV = (
    "support_program_tv",
    "query_program_tv",
    "probability_marginal_tv",
    "joint_belief_tv",
    "suffix_predictive_tv",
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def q95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def safe_log_loss(probability):
    return -math.log(max(float(probability), 1e-300))


def entropy(values):
    return -sum(float(value) * math.log(float(value)) for value in values if value > 0)


def normalized(*distributions):
    with localcontext() as context:
        context.prec = 100
        return all(
            abs(sum(values, Decimal(0)) - 1) < Decimal("1e-80")
            for values in distributions
        )


def exact_record_log_evidence(registry, supports, query):
    support_logs = []
    with localcontext() as context:
        context.prec = 100
        for mechanic in registry:
            total = Decimal(0)
            possible = True
            for support in supports:
                world = {row["atom"]: row["allowed_values"][0] for row in support["initial_state"]}
                likelihood, _ = sequential_filter(
                    mechanic["program"], support["entities"], world,
                    support["actions"], support["observations"],
                )
                if not likelihood:
                    possible = False
                    break
                total += Decimal(likelihood.numerator).ln() - Decimal(likelihood.denominator).ln()
            support_logs.append(total if possible else None)
        valid = [value for value in support_logs if value is not None]
        maximum = max(valid)
        support_normalizer = maximum + sum((value - maximum).exp() for value in valid).ln()
        support_evidence = support_normalizer - Decimal(len(registry)).ln()
        support_weights = [
            Decimal(0) if value is None else (value - support_normalizer).exp()
            for value in support_logs
        ]
        query_logs = []
        prefix = query["prefix_length"]
        for mechanic, prior in zip(registry, support_weights, strict=True):
            if not prior:
                query_logs.append(None)
                continue
            world = {row["atom"]: row["allowed_values"][0] for row in query["initial_state"]}
            likelihood, _ = sequential_filter(
                mechanic["program"], query["entities"], world,
                query["actions"][:prefix], query["observations"],
            )
            query_logs.append(
                None if not likelihood else prior.ln()
                + Decimal(likelihood.numerator).ln() - Decimal(likelihood.denominator).ln()
            )
        valid_query = [value for value in query_logs if value is not None]
        query_maximum = max(valid_query)
        query_evidence = query_maximum + sum(
            (value - query_maximum).exp() for value in valid_query
        ).ln()
        return support_evidence + query_evidence


def diagnostics_rows(inference):
    for program in inference["support_diagnostics"]:
        yield from program
    yield from inference["query_diagnostics"]


def inference_diagnostic_summary(inference, ancestor_budget):
    diagnostics = list(diagnostics_rows(inference))
    ticks = [tick for row in diagnostics for tick in row.get("ticks", [])]
    streams = [identifier for row in diagnostics for identifier in row.get("resampling_stream_ids", [])]
    fingerprints = [value for row in diagnostics for value in row.get("resampling_fingerprints", [])]
    return {
        "extinct_filters": sum(bool(row.get("extinct")) for row in diagnostics),
        "resampling_events": len(streams),
        "stream_ids": streams,
        "fingerprints": fingerprints,
        "mean_ess_fraction": (
            sum(row["ess_fraction"] for row in ticks) / len(ticks) if ticks else 1.0
        ),
        "minimum_ess_fraction": min((row["ess_fraction"] for row in ticks), default=1.0),
        "mean_distinct_configurations": (
            sum(row["distinct_configurations"] for row in ticks) / len(ticks) if ticks else 1.0
        ),
        "minimum_distinct_ancestor_fraction": min(
            (
                row["distinct_ancestors"] / ancestor_budget
                for row in ticks if row.get("distinct_ancestors") is not None
            ),
            default=1.0,
        ),
        "transition_groups": sum(row.get("transition_groups", 0) for row in diagnostics),
    }


def exact_metrics(record, exact, particle, exact_log_evidence, registry):
    target = record["target_program_index"]
    true_configuration = record["query"]["true_configuration_key"]
    exact_probability = probability_marginal(registry, exact["query_program"])
    particle_program_map = min(
        range(len(particle["query_program"])),
        key=lambda index: (-particle["query_program"][index], index),
    )
    particle_configuration_map = min(
        particle["configuration"],
        key=lambda key: (-particle["configuration"][key], key),
    )
    values = {
        "support_program_tv": sequence_tv(exact["support_program"], particle["support_program"]),
        "query_program_tv": sequence_tv(exact["query_program"], particle["query_program"]),
        "probability_marginal_tv": distribution_tv(exact_probability, particle["probability"]),
        "joint_belief_tv": distribution_tv(exact["joint"], particle["joint"]),
        "suffix_predictive_tv": distribution_tv(exact["suffix"], particle["suffix"]),
        "absolute_log_evidence_error": abs(
            float(exact_log_evidence - particle["record_log_evidence"])
        ),
        "target_program_log_loss_excess": (
            safe_log_loss(particle["query_program"][target])
            - safe_log_loss(exact["query_program"][target])
        ),
        "target_configuration_log_loss_excess": (
            safe_log_loss(particle["configuration"].get(true_configuration, Decimal(0)))
            - safe_log_loss(exact["configuration"].get(true_configuration, Decimal(0)))
        ),
        "target_program_extinct": particle["query_program"][target] == 0,
        "particle_target_program_log_loss": safe_log_loss(
            particle["query_program"][target]
        ),
        "particle_target_configuration_log_loss": safe_log_loss(
            particle["configuration"].get(true_configuration, Decimal(0))
        ),
        "target_is_particle_program_map": target == particle_program_map,
        "target_is_particle_configuration_map": true_configuration == particle_configuration_map,
    }
    ambiguous = max(exact["query_program"]) <= Decimal("0.60")
    values["ambiguous"] = ambiguous
    values["false_static_collapse"] = (
        ambiguous and max(particle["query_program"]) >= Decimal("0.95")
    )
    exact_entropy = entropy(exact["query_program"])
    values["ambiguous_entropy_ratio"] = (
        entropy(particle["query_program"]) / exact_entropy if ambiguous and exact_entropy else None
    )
    material = [key for key, mass in exact["configuration"].items() if mass >= Decimal("0.05")]
    values["material_configurations"] = len(material)
    values["lost_material_configurations"] = sum(
        particle["configuration"].get(key, Decimal(0)) == 0 for key in material
    )
    return values


def evaluate_exact_population(records, registry, config):
    rows, controls = [], {"likelihood_squared": []}
    stream_collisions = 0
    fingerprint_pairs = 0
    fingerprint_collisions = 0
    budgets = config["particleBudgets"]["budgets"]
    repeats = config["particleBudgets"]["independentRepeatsOnExactBenchmark"]
    for record in records:
        record_fingerprints = {}
        exact = independent_inference(registry, record["supports"], record["query"])
        exact_log_evidence = exact_record_log_evidence(
            registry, record["supports"], record["query"]
        )
        for budget in budgets:
            for repeat in range(repeats):
                particle = particle_inference(
                    registry, record["supports"], record["query"], budget,
                    config["population"]["particleSeed"], "exact", record["id"], repeat,
                    config["algorithm"]["resamplingEssThresholdFraction"],
                )
                metrics = exact_metrics(record, exact, particle, exact_log_evidence, registry)
                diagnostic = inference_diagnostic_summary(particle, budget)
                stream_collisions += (
                    len(diagnostic["stream_ids"]) - len(set(diagnostic["stream_ids"]))
                )
                record_fingerprints[(budget, repeat)] = diagnostic["fingerprints"]
                rows.append({
                    "id": record["id"], "budget": budget, "repeat": repeat,
                    **metrics,
                    "resampling_events": diagnostic["resampling_events"],
                    "mean_ess_fraction": diagnostic["mean_ess_fraction"],
                    "minimum_ess_fraction": diagnostic["minimum_ess_fraction"],
                    "mean_distinct_configurations": diagnostic["mean_distinct_configurations"],
                })
        squared = particle_inference(
            registry, record["supports"], record["query"],
            config["particleBudgets"]["primaryBudget"],
            config["population"]["particleSeed"], "exact-control", record["id"], 0,
            config["algorithm"]["resamplingEssThresholdFraction"], likelihood_power=2,
        )
        controls["likelihood_squared"].append(
            sequence_tv(exact["query_program"], squared["query_program"])
        )
        for budget in budgets:
            values = [record_fingerprints[(budget, repeat)] for repeat in range(repeats)]
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    if values[left] or values[right]:
                        fingerprint_pairs += 1
                        fingerprint_collisions += values[left] == values[right]
    return (
        rows, controls, stream_collisions, fingerprint_collisions, fingerprint_pairs
    )


def rank_seed(config, replication, quantity):
    return int(sha256_text(
        f"v52-rank|{config['population']['tieBreakSeed']}|{replication}|{quantity}"
    ), 16)


def draw_seed(config, replication, draw):
    return int(sha256_text(
        f"v52-draw|{config['population']['posteriorDrawSeed']}|{replication}|{draw}"
    ), 16)


def sbc_ranks(inference, record, config):
    target = record["target_program_index"]
    true_configuration = record["query"]["true_configuration_key"]
    draws = [
        categorical_sample(inference["joint"], draw_seed(config, record["replication"], draw))
        for draw in range(config["sbc"]["posteriorDrawsPerReplication"])
    ]
    true_values = {
        "program_ordinal": record["target_program_ordinal"],
        "probability_ordinal": record["target_probability_ordinal"],
        "configuration_ordinal": true_configuration,
        "program_posterior_probability": inference["query_program"][target],
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
            rank_seed(config, record["replication"], quantity),
        )
        for quantity in QUANTITIES
    }


def rank_diagnostics(rows, config):
    specification = config["sbc"]
    replications, bins, support = (
        specification["replications"], specification["rankBins"],
        specification["rankSupportSize"],
    )
    expected = replications / bins
    bin_sd = math.sqrt(replications * (1 / bins) * (1 - 1 / bins))
    histograms, p_values, max_z, coverage = {}, {}, {}, {}
    for quantity in QUANTITIES:
        counts = [0 for _ in range(bins)]
        ranks = [row["ranks"][quantity] for row in rows]
        for rank in ranks:
            counts[min(bins - 1, rank * bins // support)] += 1
        statistic = sum((count - expected) ** 2 / expected for count in counts)
        histograms[quantity] = counts
        p_values[quantity] = float(chi2.sf(statistic, bins - 1))
        max_z[quantity] = max(abs(count - expected) / bin_sd for count in counts)
        coverage[quantity] = {}
        for level in specification["coverageLevels"]:
            included = round(level * support)
            lower = (support - included) // 2
            upper = lower + included
            expected_coverage = included / support
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
        "minimum_chi_square_p_value": min(p_values.values()),
        "maximum_absolute_rank_bin_z_by_quantity": max_z,
        "maximum_absolute_rank_bin_z": max(max_z.values()),
        "coverage": coverage,
        "maximum_absolute_coverage_z": max(
            abs(cell["z"]) for quantity in coverage.values() for cell in quantity.values()
        ),
    }


def evaluate_sbc_population(records, registry, config):
    rows, stream_collisions = [], 0
    budget = config["particleBudgets"]["primaryBudget"]
    repeat = config["particleBudgets"]["primaryRepeatOnSbc"]
    for record in records:
        inference = particle_inference(
            registry, record["supports"], record["query"], budget,
            config["population"]["particleSeed"], "sbc", record["id"], repeat,
            config["algorithm"]["resamplingEssThresholdFraction"],
        )
        diagnostic = inference_diagnostic_summary(inference, budget)
        stream_collisions += (
            len(diagnostic["stream_ids"]) - len(set(diagnostic["stream_ids"]))
        )
        rows.append({
            "id": record["id"],
            "ranks": sbc_ranks(inference, record, config),
            "normalization": normalized(
                inference["support_program"], inference["query_program"],
                inference["joint"].values(),
            ),
            "target_program_extinct": (
                inference["query_program"][record["target_program_index"]] == 0
            ),
            "resampling_events": diagnostic["resampling_events"],
        })
    return rows, stream_collisions


def scale_query(episode):
    return {
        **episode,
        "prefix_length": episode["sequence_length"],
        "true_configuration_key": "not_scored",
    }


def evaluate_scale_population(records, registry, config):
    rows, stream_collisions = [], 0
    budget = config["scaleStress"]["particleBudget"]
    for record in records:
        inference = particle_inference(
            registry, record["episodes"][:-1], scale_query(record["episodes"][-1]), budget,
            config["population"]["particleSeed"], "scale", record["id"], 0,
            config["algorithm"]["resamplingEssThresholdFraction"], track_ancestry=True,
        )
        diagnostic = inference_diagnostic_summary(inference, budget)
        stream_collisions += (
            len(diagnostic["stream_ids"]) - len(set(diagnostic["stream_ids"]))
        )
        rows.append({
            "id": record["id"],
            "normalization": normalized(
                inference["query_program"], inference["joint"].values()
            ),
            "target_program_extinct": (
                inference["query_program"][record["target_program_index"]] == 0
            ),
            "resampling_events": diagnostic["resampling_events"],
            "mean_ess_fraction": diagnostic["mean_ess_fraction"],
            "minimum_ess_fraction": diagnostic["minimum_ess_fraction"],
            "mean_distinct_configurations": diagnostic["mean_distinct_configurations"],
            "minimum_distinct_ancestor_fraction": diagnostic["minimum_distinct_ancestor_fraction"],
            "transition_groups": diagnostic["transition_groups"],
        })
    return rows, stream_collisions


def aggregate_exact(
    rows,
    controls,
    stream_collisions,
    fingerprint_collisions,
    repeated_fingerprint_pairs,
    config,
):
    by_budget = {}
    for budget in config["particleBudgets"]["budgets"]:
        selected = [row for row in rows if row["budget"] == budget]
        by_budget[str(budget)] = {
            **{
                f"mean_{field}": sum(row[field] for row in selected) / len(selected)
                for field in CORE_TV
            },
            **{
                f"q95_{field}": q95([row[field] for row in selected])
                for field in CORE_TV
            },
            "mean_absolute_log_evidence_error": sum(
                row["absolute_log_evidence_error"] for row in selected
            ) / len(selected),
            "target_program_log_loss_excess": sum(
                row["target_program_log_loss_excess"] for row in selected
            ) / len(selected),
            "target_configuration_log_loss_excess": sum(
                row["target_configuration_log_loss_excess"] for row in selected
            ) / len(selected),
            "target_program_extinction_rate": sum(
                row["target_program_extinct"] for row in selected
            ) / len(selected),
            "false_static_collapse_rate": (
                sum(row["false_static_collapse"] for row in selected)
                / max(1, sum(row["ambiguous"] for row in selected))
            ),
            "ambiguous_program_entropy_ratio": (
                sum(
                    row["ambiguous_entropy_ratio"] for row in selected
                    if row["ambiguous_entropy_ratio"] is not None
                )
                / max(1, sum(row["ambiguous_entropy_ratio"] is not None for row in selected))
            ),
            "false_configuration_loss_rate": (
                sum(row["lost_material_configurations"] for row in selected)
                / max(1, sum(row["material_configurations"] for row in selected))
            ),
        }
        by_budget[str(budget)]["mean_core_tv"] = sum(
            by_budget[str(budget)][f"mean_{field}"] for field in CORE_TV
        ) / len(CORE_TV)
        per_record = []
        for record_id in sorted({row["id"] for row in selected}):
            repeat_values = [
                sum(row[field] for field in CORE_TV) / len(CORE_TV)
                for row in selected if row["id"] == record_id
            ]
            per_record.append(statistics.pstdev(repeat_values))
        by_budget[str(budget)]["mean_repeat_core_tv_dispersion"] = (
            sum(per_record) / len(per_record)
        )
    ordered_budgets = config["particleBudgets"]["budgets"]
    primary = str(config["particleBudgets"]["primaryBudget"])
    medium, low = str(ordered_budgets[-2]), str(ordered_budgets[-3])
    primary_rows = [row for row in rows if str(row["budget"]) == primary]
    particle_program_loss = sum(
        row["particle_target_program_log_loss"] for row in primary_rows
    ) / len(primary_rows)
    map_program_loss = sum(
        0.0 if row["target_is_particle_program_map"] else safe_log_loss(Decimal(0))
        for row in primary_rows
    ) / len(primary_rows)
    particle_configuration_loss = sum(
        row["particle_target_configuration_log_loss"] for row in primary_rows
    ) / len(primary_rows)
    map_configuration_loss = sum(
        0.0 if row["target_is_particle_configuration_map"] else safe_log_loss(Decimal(0))
        for row in primary_rows
    ) / len(primary_rows)
    squared_mean = sum(controls["likelihood_squared"]) / len(controls["likelihood_squared"])
    primary_query_tv = by_budget[primary]["mean_query_program_tv"]
    control_checks = {
        "map_program": map_program_loss > particle_program_loss + 0.01,
        "map_configuration": map_configuration_loss > particle_configuration_loss + 0.01,
        "likelihood_squared": squared_mean > primary_query_tv + 0.01,
        "stream_collision": len({
            stream_id(1, "intentional-collision"),
            stream_id(1, "intentional-collision"),
        }) == 1,
    }
    return {
        "primary_budget": config["particleBudgets"]["primaryBudget"],
        "completed_fraction": len(rows) / (
            config["exactBenchmark"]["records"]
            * len(config["particleBudgets"]["budgets"])
            * config["particleBudgets"]["independentRepeatsOnExactBenchmark"]
        ),
        "by_budget": by_budget,
        "primary_minus_medium_mean_tv": by_budget[primary]["mean_core_tv"] - by_budget[medium]["mean_core_tv"],
        "medium_minus_low_mean_tv": by_budget[medium]["mean_core_tv"] - by_budget[low]["mean_core_tv"],
        "unintended_stream_collision_count": stream_collisions,
        "stochastic_fingerprint_collision_rate": (
            fingerprint_collisions / repeated_fingerprint_pairs
            if repeated_fingerprint_pairs else 0.0
        ),
        "controls": {
            "checks": control_checks,
            "detected_or_dominated": sum(control_checks.values()),
            "likelihood_squared_mean_query_program_tv": squared_mean,
        },
    }


def aggregate_scale(rows):
    return {
        "completion_fraction": 1.0,
        "normalization_rate": sum(row["normalization"] for row in rows) / len(rows),
        "target_program_extinction_rate": sum(
            row["target_program_extinct"] for row in rows
        ) / len(rows),
        "mean_resampling_events": sum(row["resampling_events"] for row in rows) / len(rows),
        "mean_ess_fraction": sum(row["mean_ess_fraction"] for row in rows) / len(rows),
        "minimum_ess_fraction": min(row["minimum_ess_fraction"] for row in rows),
        "mean_distinct_configurations": sum(
            row["mean_distinct_configurations"] for row in rows
        ) / len(rows),
        "minimum_distinct_ancestor_fraction": min(
            row["minimum_distinct_ancestor_fraction"] for row in rows
        ),
        "transition_groups": sum(row["transition_groups"] for row in rows),
    }


def qualification(metrics, gates):
    primary = metrics["exact"]["by_budget"][str(metrics["exact"]["primary_budget"])]
    checks = {
        "exact_completion": metrics["exact"]["completed_fraction"] >= gates["minimumCompletedExactBenchmarkFraction"],
        "normalization": metrics["sbc_normalization_rate"] >= gates["minimumNormalizationRate"],
        "support_program_mean_tv": primary["mean_support_program_tv"] <= gates["maximumPrimaryMeanSupportProgramTv"],
        "support_program_q95_tv": primary["q95_support_program_tv"] <= gates["maximumPrimaryQ95SupportProgramTv"],
        "query_program_mean_tv": primary["mean_query_program_tv"] <= gates["maximumPrimaryMeanQueryProgramTv"],
        "query_program_q95_tv": primary["q95_query_program_tv"] <= gates["maximumPrimaryQ95QueryProgramTv"],
        "probability_mean_tv": primary["mean_probability_marginal_tv"] <= gates["maximumPrimaryMeanProbabilityMarginalTv"],
        "probability_q95_tv": primary["q95_probability_marginal_tv"] <= gates["maximumPrimaryQ95ProbabilityMarginalTv"],
        "joint_mean_tv": primary["mean_joint_belief_tv"] <= gates["maximumPrimaryMeanJointBeliefTv"],
        "joint_q95_tv": primary["q95_joint_belief_tv"] <= gates["maximumPrimaryQ95JointBeliefTv"],
        "suffix_mean_tv": primary["mean_suffix_predictive_tv"] <= gates["maximumPrimaryMeanSuffixPredictiveTv"],
        "suffix_q95_tv": primary["q95_suffix_predictive_tv"] <= gates["maximumPrimaryQ95SuffixPredictiveTv"],
        "log_evidence": primary["mean_absolute_log_evidence_error"] <= gates["maximumMeanAbsoluteLogEvidenceError"],
        "target_program_log_loss": primary["target_program_log_loss_excess"] <= gates["maximumTargetProgramLogLossExcess"],
        "target_configuration_log_loss": primary["target_configuration_log_loss_excess"] <= gates["maximumTargetConfigurationLogLossExcess"],
        "primary_vs_medium": metrics["exact"]["primary_minus_medium_mean_tv"] <= gates["maximumPrimaryMinusMediumMeanTv"],
        "medium_vs_low": metrics["exact"]["medium_minus_low_mean_tv"] <= gates["maximumMediumMinusLowMeanTv"],
        "target_extinction": primary["target_program_extinction_rate"] <= gates["maximumTargetProgramExtinctionRate"],
        "false_static_collapse": primary["false_static_collapse_rate"] <= gates["maximumFalseStaticCollapseRate"],
        "false_configuration_loss": primary["false_configuration_loss_rate"] <= gates["maximumFalseConfigurationLossRate"],
        "ambiguity_entropy": primary["ambiguous_program_entropy_ratio"] >= gates["minimumAmbiguousProgramEntropyRatio"],
        "stream_collisions": metrics["exact"]["unintended_stream_collision_count"] <= gates["maximumUnintendedStreamCollisions"],
        "fingerprint_collisions": metrics["exact"]["stochastic_fingerprint_collision_rate"] <= gates["maximumStochasticFingerprintCollisionRate"],
        "rank_chi_square": metrics["sbc"]["minimum_chi_square_p_value"] >= gates["minimumRankChiSquarePValue"],
        "rank_bin_envelope": metrics["sbc"]["maximum_absolute_rank_bin_z"] <= gates["maximumAbsoluteRankBinZ"],
        "coverage": metrics["sbc"]["maximum_absolute_coverage_z"] <= gates["maximumAbsoluteCoverageZ"],
        "control_sensitivity": metrics["exact"]["controls"]["detected_or_dominated"] >= gates["minimumControlsDetectedOrDominated"],
        "scale_completion": metrics["scale"]["completion_fraction"] >= gates["minimumScaleStressCompletionFraction"],
        "scale_normalization": metrics["scale"]["normalization_rate"] >= gates["minimumScaleStressNormalizationRate"],
        "scale_target_extinction": metrics["scale"]["target_program_extinction_rate"] <= gates["maximumScaleStressTargetProgramExtinctionRate"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-seal", default="configs/v52-population-seal.json")
    parser.add_argument(
        "--output-dir", default="outputs/v52-rao-blackwellized-particle-filtering/evaluation"
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.population_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "evaluation-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V52 particle evaluation already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V52 implementation changed: {path}")
    records = {}
    for name, artifact in seal["populations"].items():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"V52 sealed {name} population changed")
        records[name] = read(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 52,
        "status": "started",
        "evaluation_run": 1,
        "population_seal_sha256": file_sha256(seal_path),
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    registry = mechanic_registry()
    config = implementation["config_payload"]
    (
        exact_rows,
        controls,
        exact_stream_collisions,
        fingerprint_collisions,
        fingerprint_pairs,
    ) = evaluate_exact_population(records["exact"], registry, config)
    sbc_rows, sbc_stream_collisions = evaluate_sbc_population(
        records["sbc"], registry, config
    )
    scale_rows, scale_stream_collisions = evaluate_scale_population(
        records["scale"], registry, config
    )
    exact_metrics_value = aggregate_exact(
        exact_rows,
        controls,
        exact_stream_collisions,
        fingerprint_collisions,
        fingerprint_pairs,
        config,
    )
    exact_metrics_value["unintended_stream_collision_count"] += (
        sbc_stream_collisions + scale_stream_collisions
    )
    metrics = {
        "exact": exact_metrics_value,
        "sbc": rank_diagnostics(sbc_rows, config),
        "sbc_normalization_rate": sum(row["normalization"] for row in sbc_rows) / len(sbc_rows),
        "sbc_target_program_extinction_rate": sum(
            row["target_program_extinct"] for row in sbc_rows
        ) / len(sbc_rows),
        "scale": aggregate_scale(scale_rows),
    }
    q = qualification(metrics, config["gates"])
    decision = (
        "authorize_continuous_parameter_smc_squared_preregistration_with_pmcmc_reference"
        if q["passed"]
        else "repair_particle_accuracy_calibration_degeneracy_or_stream_integrity"
    )
    output.mkdir(parents=True)
    detail_paths = {}
    for name, rows in (("exact", exact_rows), ("sbc", sbc_rows), ("scale", scale_rows)):
        path = output / f"{name}-metrics.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        detail_paths[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)
        }
    audit_support_path = output / "audit-support.json"
    audit_support_path.write_text(json.dumps({
        "controls": controls,
        "exact_stream_collisions": exact_stream_collisions,
        "sbc_stream_collisions": sbc_stream_collisions,
        "scale_stream_collisions": scale_stream_collisions,
        "fingerprint_collisions": fingerprint_collisions,
        "fingerprint_pairs": fingerprint_pairs,
    }, indent=2, sort_keys=True) + "\n")
    result = {
        "schema_version": 52,
        "experiment": config["experiment"],
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_run_number": 1,
        "metrics": metrics,
        "qualification": q,
        "decision": decision,
        "detail_metrics": detail_paths,
        "audit_support": {
            "path": str(audit_support_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(audit_support_path),
        },
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "particle_evaluation_runs": 1,
            "selection_on_sealed_results": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
        "authorization": {
            "preregister_continuous_parameter_smc_squared": q["passed"],
            "construct_smc_squared_population": False,
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
