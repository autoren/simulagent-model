#!/usr/bin/env python3
"""Run the single sealed V53r1 continuous-parameter evaluation."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import Counter

import numpy as np
from scipy.stats import chi2, norm, rankdata, wasserstein_distance

from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import (
    exact_conditional_theta,
    exact_inference,
    mechanic_registry,
    pmmh_conditional_chains,
    pool_smc2_repeats,
    smc2_inference,
)


QUANTITIES = (
    "program_ordinal",
    "continuous_theta",
    "configuration_ordinal",
    "target_program_posterior_probability",
    "target_configuration_posterior_probability",
)
ERROR_FIELDS = (
    "program_tv",
    "theta_wasserstein",
    "binned_program_theta_tv",
    "configuration_tv",
    "suffix_predictive_tv",
    "absolute_log_evidence_error",
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def q95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def sequence_tv(left, right):
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))


def map_tv(left, right):
    return 0.5 * sum(
        abs(left.get(key, 0.0) - right.get(key, 0.0))
        for key in set(left) | set(right)
    )


def theta_wasserstein(left, right):
    return float(wasserstein_distance(
        left["theta_values"], right["theta_values"],
        u_weights=left["theta_weights"], v_weights=right["theta_weights"],
    ))


def weighted_quantile(values, weights, probability):
    ordered = sorted(zip(values, weights, strict=True))
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= probability:
            return value
    return ordered[-1][0]


def entropy(values):
    return -sum(value * math.log(value) for value in values if value > 0)


def normalized(result, tolerance=1e-10):
    checks = (
        sum(result["program"]),
        sum(result["theta_weights"]),
        sum(result["joint_bins"].values()),
        sum(result["configuration"].values()),
        sum(result["suffix"].values()),
        sum(row["weight"] for row in result["atoms"]),
    )
    return all(abs(value - 1.0) <= tolerance for value in checks)


def comparison(exact, estimate):
    return {
        "program_tv": sequence_tv(exact["program"], estimate["program"]),
        "theta_wasserstein": theta_wasserstein(exact, estimate),
        "binned_program_theta_tv": map_tv(
            exact["joint_bins"], estimate["joint_bins"]
        ),
        "configuration_tv": map_tv(
            exact["configuration"], estimate["configuration"]
        ),
        "suffix_predictive_tv": map_tv(exact["suffix"], estimate["suffix"]),
        "absolute_log_evidence_error": abs(
            exact["log_evidence"] - estimate["log_evidence"]
        ),
    }


def core_error(row):
    return sum(row[field] for field in ERROR_FIELDS) / len(ERROR_FIELDS)


def single_diagnostics(result, outer_budget):
    outer_streams, outer_fingerprints = [], []
    inner_streams, inner_fingerprints, inner_ess = [], [], []
    final_outer_ess, ancestor_fractions = [], []
    moves_attempted = moves_accepted = 0
    for program in result["program_results"]:
        if program is None:
            continue
        diagnostic = program["diagnostics"]
        outer_streams.extend(diagnostic["outer_resampling_stream_ids"])
        outer_fingerprints.extend(diagnostic["outer_resampling_fingerprints"])
        inner_streams.extend(diagnostic["inner_resampling_stream_ids"])
        inner_fingerprints.extend(diagnostic["inner_resampling_fingerprints"])
        inner_ess.extend(diagnostic["inner_ess_fractions"])
        moves_attempted += diagnostic["move_attempts"]
        moves_accepted += diagnostic["move_accepts"]
        weights = [row["weight"] for row in program["particles"]]
        final_outer_ess.append(1 / sum(value * value for value in weights) / outer_budget)
        ancestor_fractions.append(
            len({row["ancestor"] for row in program["particles"]}) / outer_budget
        )
    return {
        "outer_streams": outer_streams,
        "outer_fingerprints": outer_fingerprints,
        "inner_streams": inner_streams,
        "inner_fingerprints": inner_fingerprints,
        "mean_inner_ess_fraction": statistics.fmean(inner_ess) if inner_ess else 1.0,
        "minimum_inner_ess_fraction": min(inner_ess, default=1.0),
        "mean_final_outer_ess_fraction": (
            statistics.fmean(final_outer_ess) if final_outer_ess else 0.0
        ),
        "minimum_final_outer_ess_fraction": min(final_outer_ess, default=0.0),
        "mean_distinct_theta_ancestor_fraction": (
            statistics.fmean(ancestor_fractions) if ancestor_fractions else 0.0
        ),
        "move_attempts": moves_attempted,
        "move_acceptance_rate": (
            moves_accepted / moves_attempted if moves_attempted else 0.0
        ),
    }


def merge_diagnostics(results, outer_budget):
    rows = [single_diagnostics(result, outer_budget) for result in results]
    return {
        "outer_streams": [value for row in rows for value in row["outer_streams"]],
        "outer_fingerprints": [
            value for row in rows for value in row["outer_fingerprints"]
        ],
        "inner_streams": [value for row in rows for value in row["inner_streams"]],
        "inner_fingerprints": [
            value for row in rows for value in row["inner_fingerprints"]
        ],
        "mean_inner_ess_fraction": statistics.fmean(
            row["mean_inner_ess_fraction"] for row in rows
        ),
        "minimum_inner_ess_fraction": min(
            row["minimum_inner_ess_fraction"] for row in rows
        ),
        "mean_final_outer_ess_fraction": statistics.fmean(
            row["mean_final_outer_ess_fraction"] for row in rows
        ),
        "minimum_final_outer_ess_fraction": min(
            row["minimum_final_outer_ess_fraction"] for row in rows
        ),
        "mean_distinct_theta_ancestor_fraction": statistics.fmean(
            row["mean_distinct_theta_ancestor_fraction"] for row in rows
        ),
        "move_attempts": sum(row["move_attempts"] for row in rows),
        "move_acceptance_rate": (
            sum(row["move_acceptance_rate"] * row["move_attempts"] for row in rows)
            / max(1, sum(row["move_attempts"] for row in rows))
        ),
    }


def duplicate_count(values):
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def altered_stream_collision_control(config):
    seed = config["population"]["outerParticleSeed"]
    left = sha256_text(f"v53-stream|{seed}|intentional|collision")
    right = sha256_text(f"v53-stream|{seed}|intentional|collision")
    independent = sha256_text(f"v53-stream|{seed}|intentional|independent")
    return left == right and left != independent


def control_records(records, config):
    stride = config["exactBenchmark"]["recordsPerTemplate"]
    ordinals = set(config["controls"]["controlReplicateOrdinalsPerTemplate"])
    return [
        record for index, record in enumerate(records) if index % stride in ordinals
    ]


def evaluate_exact(records, registry, config):
    budgets = config["smcSquared"]["outerThetaParticleBudgets"]
    repeats = config["smcSquared"]["independentRepeatsOnExactBenchmark"]
    primary = config["smcSquared"]["primaryOuterThetaParticleBudget"]
    selected_controls = {row["id"] for row in control_records(records, config)}
    rows, pmcmc_rows, control_rows = [], [], []
    all_streams = {"outer": [], "inner": []}
    all_fingerprints = {"outer": [], "inner": []}
    for ordinal, record in enumerate(records):
        started = time.perf_counter()
        exact = exact_inference(registry, record, config)
        estimates = {}
        for budget in budgets:
            independent = [
                smc2_inference(
                    registry, record, config, budget, repeat, "exact",
                    track_ancestry=True,
                )
                for repeat in range(repeats)
            ]
            pooled = pool_smc2_repeats(independent)
            estimates[budget] = pooled
            diagnostic = merge_diagnostics(independent, budget)
            all_streams["outer"].extend(diagnostic["outer_streams"])
            all_streams["inner"].extend(diagnostic["inner_streams"])
            all_fingerprints["outer"].extend(diagnostic["outer_fingerprints"])
            all_fingerprints["inner"].extend(diagnostic["inner_fingerprints"])
            metrics = comparison(exact, pooled)
            exact_width = weighted_quantile(
                exact["theta_values"], exact["theta_weights"], 0.9
            ) - weighted_quantile(
                exact["theta_values"], exact["theta_weights"], 0.1
            )
            unique_fraction = len(set(pooled["theta_values"])) / len(
                pooled["theta_values"]
            )
            ambiguous = max(exact["program"]) <= 0.60
            exact_entropy = entropy(exact["program"])
            rows.append({
                "id": record["id"],
                "budget": budget,
                **metrics,
                "normalization": normalized(pooled),
                "target_program_extinct": (
                    pooled["program"][record["target_program_index"]] == 0
                ),
                "ambiguous": ambiguous,
                "false_static_collapse": ambiguous and max(pooled["program"]) >= 0.95,
                "ambiguous_entropy_ratio": (
                    entropy(pooled["program"]) / exact_entropy
                    if ambiguous and exact_entropy else None
                ),
                "exact_theta_central_80_width": exact_width,
                "unique_theta_support_fraction": unique_fraction,
                "false_theta_collapse": exact_width >= 0.15 and unique_fraction < 0.02,
                "mean_final_outer_ess_fraction": diagnostic[
                    "mean_final_outer_ess_fraction"
                ],
                "minimum_final_outer_ess_fraction": diagnostic[
                    "minimum_final_outer_ess_fraction"
                ],
                "mean_inner_ess_fraction": diagnostic["mean_inner_ess_fraction"],
                "mean_distinct_theta_ancestor_fraction": diagnostic[
                    "mean_distinct_theta_ancestor_fraction"
                ],
                "move_acceptance_rate": diagnostic["move_acceptance_rate"],
            })

        if record["id"] in selected_controls:
            base = comparison(exact, estimates[primary])
            mean_theta = sum(
                value * weight
                for value, weight in zip(
                    exact["theta_values"], exact["theta_weights"], strict=True
                )
            )
            point_wasserstein = float(wasserstein_distance(
                exact["theta_values"], [mean_theta],
                u_weights=exact["theta_weights"], v_weights=[1.0],
            ))
            map_index = min(
                range(len(exact["program"])),
                key=lambda index: (-exact["program"][index], index),
            )
            map_values = [float(index == map_index) for index in range(len(registry))]
            squared = smc2_inference(
                registry, record, config, primary, 0, "control-squared",
                likelihood_power=2,
            )
            disabled = smc2_inference(
                registry, record, config, primary, 0, "control-disabled",
                disable_outer_resampling=True,
            )
            control_rows.append({
                "id": record["id"],
                "base_core_error": core_error(base),
                "base_theta_wasserstein": base["theta_wasserstein"],
                "theta_point_mass_wasserstein": point_wasserstein,
                "base_program_tv": base["program_tv"],
                "map_program_tv": sequence_tv(exact["program"], map_values),
                "likelihood_squared_core_error": core_error(comparison(exact, squared)),
                "outer_resampling_disabled_core_error": core_error(
                    comparison(exact, disabled)
                ),
            })

        if record.get("pmcmc_reference"):
            program_index = record["target_program_index"]
            exact_values, exact_weights = exact_conditional_theta(exact, program_index)
            chains = pmmh_conditional_chains(
                registry[program_index], record, config
            )
            diagnostics = chain_diagnostics([row["draws"] for row in chains])
            draws = [value for row in chains for value in row["draws"]]
            pmcmc_rows.append({
                "id": record["id"],
                "acceptance_rate": statistics.fmean(
                    row["acceptance_rate"] for row in chains
                ),
                "split_rhat": diagnostics["split_rhat"],
                "bulk_ess": diagnostics["bulk_ess"],
                "theta_wasserstein": float(wasserstein_distance(
                    exact_values, draws, u_weights=exact_weights
                )),
                "conditional_theta_mean_error": abs(
                    statistics.fmean(draws)
                    - sum(value * weight for value, weight in zip(
                        exact_values, exact_weights, strict=True
                    ))
                ),
            })
        print(json.dumps({
            "stage": "exact", "record": ordinal + 1, "total": len(records),
            "id": record["id"], "seconds": time.perf_counter() - started,
        }), flush=True)
    return rows, pmcmc_rows, control_rows, all_streams, all_fingerprints


def categorical_atom(atoms, seed):
    draw = random.Random(seed).random()
    cumulative = 0.0
    for atom in atoms:
        cumulative += atom["weight"]
        if draw < cumulative:
            return atom
    return atoms[-1]


def randomized_rank(true_value, draws, seed):
    lower = sum(value < true_value for value in draws)
    ties = sum(value == true_value for value in draws)
    return lower + random.Random(seed).randrange(ties + 1)


def rank_seed(config, record_id, quantity):
    return int(sha256_text(
        f"v53-rank|{config['population']['tieBreakSeed']}|{record_id}|{quantity}"
    ), 16)


def draw_seed(config, record_id, draw):
    return int(sha256_text(
        f"v53-draw|{config['population']['posteriorDrawSeed']}|{record_id}|{draw}"
    ), 16)


def sbc_ranks(inference, record, config):
    atoms = [
        categorical_atom(inference["atoms"], draw_seed(config, record["id"], draw))
        for draw in range(config["sbc"]["posteriorDrawsPerReplication"])
    ]
    target_program = record["target_program_index"]
    target_configuration = record["query"]["true_configuration_key"]
    configuration_keys = sorted({
        target_configuration,
        *(atom["configuration_key"] for atom in atoms),
    })
    configuration_ordinal = {
        value: index for index, value in enumerate(configuration_keys)
    }
    true_values = {
        "program_ordinal": record["target_program_ordinal"],
        "continuous_theta": record["target_theta"],
        "configuration_ordinal": configuration_ordinal[target_configuration],
        "target_program_posterior_probability": inference["program"][target_program],
        "target_configuration_posterior_probability": inference["configuration"].get(
            target_configuration, 0.0
        ),
    }
    draw_values = {quantity: [] for quantity in QUANTITIES}
    for atom in atoms:
        draw_values["program_ordinal"].append(
            atom["program_index"]
        )
        draw_values["continuous_theta"].append(atom["theta"])
        draw_values["configuration_ordinal"].append(
            configuration_ordinal[atom["configuration_key"]]
        )
        draw_values["target_program_posterior_probability"].append(
            inference["program"][atom["program_index"]]
        )
        draw_values["target_configuration_posterior_probability"].append(
            inference["configuration"][atom["configuration_key"]]
        )
    return {
        quantity: randomized_rank(
            true_values[quantity], draw_values[quantity],
            rank_seed(config, record["id"], quantity),
        )
        for quantity in QUANTITIES
    }


def evaluate_sbc(records, registry, config):
    primary = config["smcSquared"]["primaryOuterThetaParticleBudget"]
    rows, streams, fingerprints = [], {"outer": [], "inner": []}, {"outer": [], "inner": []}
    for ordinal, record in enumerate(records):
        result = smc2_inference(
            registry, record, config, primary, 0, "sbc", track_ancestry=True
        )
        diagnostic = single_diagnostics(result, primary)
        for key in streams:
            streams[key].extend(diagnostic[f"{key}_streams"])
            fingerprints[key].extend(diagnostic[f"{key}_fingerprints"])
        rows.append({
            "id": record["id"],
            "ranks": sbc_ranks(result, record, config),
            "normalization": normalized(result),
            "target_program_extinct": (
                result["program"][record["target_program_index"]] == 0
            ),
            "mean_final_outer_ess_fraction": diagnostic[
                "mean_final_outer_ess_fraction"
            ],
            "mean_distinct_theta_ancestor_fraction": diagnostic[
                "mean_distinct_theta_ancestor_fraction"
            ],
        })
        if (ordinal + 1) % 8 == 0 or ordinal + 1 == len(records):
            print(json.dumps({
                "stage": "sbc", "record": ordinal + 1, "total": len(records)
            }), flush=True)
    return rows, streams, fingerprints


def evaluate_scale(records, registry, config):
    budget = config["scaleStress"]["outerThetaParticleBudget"]
    rows, streams, fingerprints = [], {"outer": [], "inner": []}, {"outer": [], "inner": []}
    for ordinal, record in enumerate(records):
        started = time.perf_counter()
        result = smc2_inference(
            registry, record, config, budget, 0, "scale", track_ancestry=True
        )
        diagnostic = single_diagnostics(result, budget)
        for key in streams:
            streams[key].extend(diagnostic[f"{key}_streams"])
            fingerprints[key].extend(diagnostic[f"{key}_fingerprints"])
        rows.append({
            "id": record["id"],
            "normalization": normalized(result),
            "target_program_extinct": (
                result["program"][record["target_program_index"]] == 0
            ),
            "mean_final_outer_ess_fraction": diagnostic[
                "mean_final_outer_ess_fraction"
            ],
            "minimum_final_outer_ess_fraction": diagnostic[
                "minimum_final_outer_ess_fraction"
            ],
            "mean_inner_ess_fraction": diagnostic["mean_inner_ess_fraction"],
            "mean_distinct_theta_ancestor_fraction": diagnostic[
                "mean_distinct_theta_ancestor_fraction"
            ],
            "outer_resampling_count": len(diagnostic["outer_streams"]),
            "inner_resampling_count": len(diagnostic["inner_streams"]),
            "move_acceptance_rate": diagnostic["move_acceptance_rate"],
            "distinct_configuration_fraction": (
                len(result["configuration"]) / max(1, budget)
            ),
            "runtime_seconds_non_gating": time.perf_counter() - started,
        })
        print(json.dumps({
            "stage": "scale", "record": ordinal + 1, "total": len(records),
            "id": record["id"], "seconds": rows[-1]["runtime_seconds_non_gating"],
        }), flush=True)
    return rows, streams, fingerprints


def _rank_normalize(chains):
    array = np.asarray(chains, dtype=float)
    ranks = rankdata(array.ravel(), method="average")
    transformed = norm.ppf((ranks - 0.375) / (len(ranks) + 0.25))
    return transformed.reshape(array.shape)


def _split_chains(chains):
    array = np.asarray(chains, dtype=float)
    half = array.shape[1] // 2
    return np.concatenate((array[:, :half], array[:, -half:]), axis=0)


def _basic_rhat(chains):
    array = np.asarray(chains, dtype=float)
    chains_count, draws = array.shape
    within = float(np.mean(np.var(array, axis=1, ddof=1)))
    between = draws * float(np.var(np.mean(array, axis=1), ddof=1))
    variance = (draws - 1) / draws * within + between / draws
    if within == 0:
        return 1.0 if between == 0 else math.inf
    return math.sqrt(variance / within)


def _autocovariance(values):
    centered = values - np.mean(values)
    size = len(values)
    result = np.correlate(centered, centered, mode="full")[size - 1:]
    return result / np.arange(size, 0, -1)


def _bulk_ess(chains):
    array = _rank_normalize(_split_chains(chains))
    chains_count, draws = array.shape
    autocov = np.asarray([_autocovariance(row) for row in array])
    within = float(np.mean(autocov[:, 0]))
    between = draws * float(np.var(np.mean(array, axis=1), ddof=1))
    variance = (draws - 1) / draws * within + between / draws
    if variance <= 0:
        return float(chains_count * draws)
    rho = [1.0]
    for lag in range(1, draws):
        rho.append(1 - (within - float(np.mean(autocov[:, lag]))) / variance)
    paired = []
    for index in range(1, len(rho) - 1, 2):
        value = rho[index] + rho[index + 1]
        if value < 0:
            break
        paired.append(value)
    tau = max(1.0, -1 + 2 * (1 + sum(paired)))
    return min(float(chains_count * draws), chains_count * draws / tau)


def chain_diagnostics(chains):
    split = _split_chains(chains)
    ranked = _rank_normalize(split)
    folded = np.abs(ranked - np.median(ranked))
    return {
        "split_rhat": max(_basic_rhat(ranked), _basic_rhat(folded)),
        "bulk_ess": _bulk_ess(chains),
    }


def rank_diagnostics(rows, config):
    specification = config["sbc"]
    replications = len(rows)
    bins, support = specification["rankBins"], specification["rankSupportSize"]
    expected = replications / bins
    bin_sd = math.sqrt(replications * (1 / bins) * (1 - 1 / bins))
    histograms, p_values, max_z, coverage = {}, {}, {}, {}
    for quantity in QUANTITIES:
        ranks = [row["ranks"][quantity] for row in rows]
        counts = [0 for _ in range(bins)]
        for value in ranks:
            counts[min(bins - 1, value * bins // support)] += 1
        statistic = sum((value - expected) ** 2 / expected for value in counts)
        histograms[quantity] = counts
        p_values[quantity] = float(chi2.sf(statistic, bins - 1))
        max_z[quantity] = max(abs(value - expected) / bin_sd for value in counts)
        coverage[quantity] = {}
        for level in specification["coverageLevels"]:
            included = round(level * support)
            lower = (support - included) // 2
            upper = lower + included
            expected_coverage = included / support
            observed = sum(lower <= rank < upper for rank in ranks) / len(ranks)
            sd = math.sqrt(
                expected_coverage * (1 - expected_coverage) / len(ranks)
            )
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
            abs(cell["z"])
            for values in coverage.values() for cell in values.values()
        ),
    }


def aggregate_exact(rows, pmcmc_rows, controls, streams, fingerprints, config):
    by_budget = {}
    for budget in config["smcSquared"]["outerThetaParticleBudgets"]:
        selected = [row for row in rows if row["budget"] == budget]
        values = {
            f"mean_{field}": statistics.fmean(row[field] for row in selected)
            for field in ERROR_FIELDS
        }
        values.update({
            f"q95_{field}": q95([row[field] for row in selected])
            for field in ERROR_FIELDS[:-1]
        })
        values.update({
            "mean_core_error": statistics.fmean(core_error(row) for row in selected),
            "normalization_rate": statistics.fmean(row["normalization"] for row in selected),
            "target_program_extinction_rate": statistics.fmean(
                row["target_program_extinct"] for row in selected
            ),
            "false_static_collapse_rate": (
                sum(row["false_static_collapse"] for row in selected)
                / max(1, sum(row["ambiguous"] for row in selected))
            ),
            "false_theta_collapse_rate": statistics.fmean(
                row["false_theta_collapse"] for row in selected
            ),
            "ambiguous_program_entropy_ratio": statistics.fmean(
                row["ambiguous_entropy_ratio"] for row in selected
                if row["ambiguous_entropy_ratio"] is not None
            ) if any(row["ambiguous_entropy_ratio"] is not None for row in selected) else 1.0,
            "mean_final_outer_ess_fraction": statistics.fmean(
                row["mean_final_outer_ess_fraction"] for row in selected
            ),
            "minimum_final_outer_ess_fraction": min(
                row["minimum_final_outer_ess_fraction"] for row in selected
            ),
            "mean_inner_ess_fraction": statistics.fmean(
                row["mean_inner_ess_fraction"] for row in selected
            ),
            "mean_distinct_theta_ancestor_fraction": statistics.fmean(
                row["mean_distinct_theta_ancestor_fraction"] for row in selected
            ),
        })
        by_budget[str(budget)] = values
    low, medium, primary = config["smcSquared"]["outerThetaParticleBudgets"]
    base_core = statistics.fmean(row["base_core_error"] for row in controls)
    control_checks = {
        "theta_point_mass": statistics.fmean(
            row["theta_point_mass_wasserstein"] - row["base_theta_wasserstein"]
            for row in controls
        ) > 0.001,
        "map_program": statistics.fmean(
            row["map_program_tv"] - row["base_program_tv"] for row in controls
        ) > 0.001,
        "likelihood_squared": statistics.fmean(
            row["likelihood_squared_core_error"] for row in controls
        ) > base_core + 0.001,
        "outer_resampling_disabled": statistics.fmean(
            row["outer_resampling_disabled_core_error"] for row in controls
        ) >= base_core,
        "stream_collision": altered_stream_collision_control(config),
    }
    pmcmc = {
        "records": len(pmcmc_rows),
        "mean_acceptance_rate": statistics.fmean(
            row["acceptance_rate"] for row in pmcmc_rows
        ),
        "maximum_split_rhat": max(row["split_rhat"] for row in pmcmc_rows),
        "minimum_bulk_ess": min(row["bulk_ess"] for row in pmcmc_rows),
        "maximum_theta_wasserstein": max(
            row["theta_wasserstein"] for row in pmcmc_rows
        ),
        "mean_theta_wasserstein": statistics.fmean(
            row["theta_wasserstein"] for row in pmcmc_rows
        ),
    }
    return {
        "completed_fraction": len(rows) / (
            config["exactBenchmark"]["records"]
            * len(config["smcSquared"]["outerThetaParticleBudgets"])
        ),
        "primary_budget": primary,
        "repeat_aggregation": config["exactBenchmark"]["repeatAggregation"],
        "theta_bins": config["exactBenchmark"]["thetaBins"],
        "by_budget": by_budget,
        "primary_minus_medium_mean_error": (
            by_budget[str(primary)]["mean_core_error"]
            - by_budget[str(medium)]["mean_core_error"]
        ),
        "medium_minus_low_mean_error": (
            by_budget[str(medium)]["mean_core_error"]
            - by_budget[str(low)]["mean_core_error"]
        ),
        "unintended_outer_stream_collision_count": duplicate_count(streams["outer"]),
        "unintended_inner_stream_collision_count": duplicate_count(streams["inner"]),
        "outer_fingerprint_collision_rate": (
            duplicate_count(fingerprints["outer"]) / max(1, len(fingerprints["outer"]))
        ),
        "inner_fingerprint_collision_rate": (
            duplicate_count(fingerprints["inner"]) / max(1, len(fingerprints["inner"]))
        ),
        "controls": {
            "checks": control_checks,
            "detected_or_dominated": sum(control_checks.values()),
            "records": len(controls),
        },
        "pmcmc": pmcmc,
    }


def aggregate_scale(rows):
    return {
        "completion_fraction": 1.0,
        "normalization_rate": statistics.fmean(row["normalization"] for row in rows),
        "target_program_extinction_rate": statistics.fmean(
            row["target_program_extinct"] for row in rows
        ),
        "mean_final_outer_ess_fraction": statistics.fmean(
            row["mean_final_outer_ess_fraction"] for row in rows
        ),
        "minimum_final_outer_ess_fraction": min(
            row["minimum_final_outer_ess_fraction"] for row in rows
        ),
        "mean_inner_ess_fraction": statistics.fmean(
            row["mean_inner_ess_fraction"] for row in rows
        ),
        "mean_distinct_theta_ancestor_fraction": statistics.fmean(
            row["mean_distinct_theta_ancestor_fraction"] for row in rows
        ),
        "outer_resampling_count": sum(row["outer_resampling_count"] for row in rows),
        "inner_resampling_count": sum(row["inner_resampling_count"] for row in rows),
        "pmmh_move_acceptance_rate": statistics.fmean(
            row["move_acceptance_rate"] for row in rows
        ),
        "mean_distinct_configuration_fraction": statistics.fmean(
            row["distinct_configuration_fraction"] for row in rows
        ),
        "runtime_seconds_non_gating": sum(
            row["runtime_seconds_non_gating"] for row in rows
        ),
    }


def qualification(metrics, gates):
    exact = metrics["exact"]
    primary = exact["by_budget"][str(exact["primary_budget"])]
    pmcmc = exact["pmcmc"]
    stream_collisions = (
        exact["unintended_outer_stream_collision_count"]
        + exact["unintended_inner_stream_collision_count"]
        + metrics["other_unintended_stream_collision_count"]
    )
    checks = {
        "exact_completion": exact["completed_fraction"] >= gates["minimumCompletedExactBenchmarkFraction"],
        "exact_normalization": primary["normalization_rate"] >= gates["minimumNormalizationRate"],
        "program_mean_tv": primary["mean_program_tv"] <= gates["maximumPrimaryMeanProgramTv"],
        "program_q95_tv": primary["q95_program_tv"] <= gates["maximumPrimaryQ95ProgramTv"],
        "theta_mean_wasserstein": primary["mean_theta_wasserstein"] <= gates["maximumPrimaryMeanThetaWasserstein"],
        "theta_q95_wasserstein": primary["q95_theta_wasserstein"] <= gates["maximumPrimaryQ95ThetaWasserstein"],
        "joint_mean_tv": primary["mean_binned_program_theta_tv"] <= gates["maximumPrimaryMeanBinnedProgramThetaTv"],
        "joint_q95_tv": primary["q95_binned_program_theta_tv"] <= gates["maximumPrimaryQ95BinnedProgramThetaTv"],
        "configuration_mean_tv": primary["mean_configuration_tv"] <= gates["maximumPrimaryMeanConfigurationTv"],
        "configuration_q95_tv": primary["q95_configuration_tv"] <= gates["maximumPrimaryQ95ConfigurationTv"],
        "suffix_mean_tv": primary["mean_suffix_predictive_tv"] <= gates["maximumPrimaryMeanSuffixPredictiveTv"],
        "suffix_q95_tv": primary["q95_suffix_predictive_tv"] <= gates["maximumPrimaryQ95SuffixPredictiveTv"],
        "log_evidence": primary["mean_absolute_log_evidence_error"] <= gates["maximumMeanAbsoluteLogEvidenceError"],
        "primary_vs_medium": exact["primary_minus_medium_mean_error"] <= gates["maximumPrimaryMinusMediumMeanError"],
        "medium_vs_low": exact["medium_minus_low_mean_error"] <= gates["maximumMediumMinusLowMeanError"],
        "target_extinction": primary["target_program_extinction_rate"] <= gates["maximumTargetProgramExtinctionRate"],
        "false_static_collapse": primary["false_static_collapse_rate"] <= gates["maximumFalseStaticCollapseRate"],
        "false_theta_collapse": primary["false_theta_collapse_rate"] <= gates["maximumFalseThetaCollapseRate"],
        "ambiguity_entropy": primary["ambiguous_program_entropy_ratio"] >= gates["minimumAmbiguousProgramEntropyRatio"],
        "outer_ess": primary["minimum_final_outer_ess_fraction"] >= gates["minimumFinalOuterEssFraction"],
        "theta_ancestry": primary["mean_distinct_theta_ancestor_fraction"] >= gates["minimumDistinctThetaAncestorFraction"],
        "stream_collisions": stream_collisions <= gates["maximumUnintendedStreamCollisions"],
        "outer_fingerprint_collisions": exact["outer_fingerprint_collision_rate"] <= gates["maximumOuterFingerprintCollisionRate"],
        "inner_fingerprint_collisions": exact["inner_fingerprint_collision_rate"] <= gates["maximumInnerFingerprintCollisionRate"],
        "rank_chi_square": metrics["sbc"]["minimum_chi_square_p_value"] >= gates["minimumRankChiSquarePValue"],
        "rank_bin_envelope": metrics["sbc"]["maximum_absolute_rank_bin_z"] <= gates["maximumAbsoluteRankBinZ"],
        "coverage": metrics["sbc"]["maximum_absolute_coverage_z"] <= gates["maximumAbsoluteCoverageZ"],
        "pmcmc_acceptance_low": pmcmc["mean_acceptance_rate"] >= gates["minimumPmcmcAcceptanceRate"],
        "pmcmc_acceptance_high": pmcmc["mean_acceptance_rate"] <= gates["maximumPmcmcAcceptanceRate"],
        "pmcmc_rhat": pmcmc["maximum_split_rhat"] <= gates["maximumPmcmcSplitRhat"],
        "pmcmc_bulk_ess": pmcmc["minimum_bulk_ess"] >= gates["minimumPmcmcBulkEss"],
        "pmcmc_wasserstein": pmcmc["maximum_theta_wasserstein"] <= gates["maximumPmcmcThetaWasserstein"],
        "controls": exact["controls"]["detected_or_dominated"] >= gates["minimumControlsDetectedOrDominated"],
        "scale_completion": metrics["scale"]["completion_fraction"] >= gates["minimumScaleStressCompletionFraction"],
        "scale_normalization": metrics["scale"]["normalization_rate"] >= gates["minimumScaleStressNormalizationRate"],
        "scale_extinction": metrics["scale"]["target_program_extinction_rate"] <= gates["maximumScaleStressTargetProgramExtinctionRate"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-seal", default="configs/v53r2-population-seal.json")
    parser.add_argument(
        "--output-dir", default="outputs/v53r2-continuous-parameter-smc2/evaluation"
    )
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.population_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "evaluation-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V53r2 evaluation already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V53r1 implementation changed: {path}")
    records = {}
    for name, artifact in seal["populations"].items():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"V53r1 sealed {name} population changed")
        records[name] = read_jsonl(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 53,
        "revision": "r2",
        "status": "started",
        "evaluation_run": 1,
        "population_seal_sha256": file_sha256(seal_path),
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    registry = mechanic_registry(
        implementation["config_payload"]["population"]["templateSeed"]
    )
    config = implementation["config_payload"]
    exact_rows, pmcmc_rows, controls, exact_streams, exact_fingerprints = evaluate_exact(
        records["exact"], registry, config
    )
    sbc_rows, sbc_streams, sbc_fingerprints = evaluate_sbc(
        records["sbc"], registry, config
    )
    scale_rows, scale_streams, scale_fingerprints = evaluate_scale(
        records["scale"], registry, config
    )
    other_stream_collisions = sum(
        duplicate_count(sbc_streams[key]) + duplicate_count(scale_streams[key])
        for key in ("outer", "inner")
    )
    exact_metrics = aggregate_exact(
        exact_rows, pmcmc_rows, controls, exact_streams, exact_fingerprints, config
    )
    # Fingerprint checks span all non-PMCMC SMC-squared populations.
    for key, gate_key in (("outer", "outer_fingerprint_collision_rate"),
                          ("inner", "inner_fingerprint_collision_rate")):
        values = [
            *exact_fingerprints[key], *sbc_fingerprints[key], *scale_fingerprints[key]
        ]
        exact_metrics[gate_key] = duplicate_count(values) / max(1, len(values))
    metrics = {
        "exact": exact_metrics,
        "sbc": rank_diagnostics(sbc_rows, config),
        "sbc_normalization_rate": statistics.fmean(
            row["normalization"] for row in sbc_rows
        ),
        "sbc_target_program_extinction_rate": statistics.fmean(
            row["target_program_extinct"] for row in sbc_rows
        ),
        "scale": aggregate_scale(scale_rows),
        "other_unintended_stream_collision_count": other_stream_collisions,
    }
    result_qualification = qualification(metrics, config["gates"])
    decision = (
        "authorize_exact_one_step_expected_information_gain_preregistration_only"
        if result_qualification["passed"]
        else "repair_v53r1_inference_calibration_pmcmc_degeneracy_or_stream_integrity"
    )
    output.mkdir(parents=True)
    detail_paths = {}
    for name, values in (
        ("exact", exact_rows), ("pmcmc", pmcmc_rows), ("controls", controls),
        ("sbc", sbc_rows), ("scale", scale_rows),
    ):
        path = output / f"{name}-metrics.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in values))
        detail_paths[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path)
        }
    result = {
        "schema_version": 53,
        "revision": "r2",
        "experiment": config["experiment"],
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_run_number": 1,
        "metrics": metrics,
        "qualification": result_qualification,
        "decision": decision,
        "detail_metrics": detail_paths,
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "smc_squared_evaluation_runs": 1,
            "pmcmc_reference_runs": 1,
            "selection_on_sealed_results": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
        "authorization": {
            "preregister_exact_one_step_expected_information_gain": result_qualification["passed"],
            "construct_active_population": False,
            "reward_or_planning": False,
            "verification": False,
            "language_grounding": False,
            "model_access": False,
        },
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    state = json.loads(attempt.read_text())
    state.update({"status": "completed", "result_sha256": file_sha256(result_path)})
    attempt.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
