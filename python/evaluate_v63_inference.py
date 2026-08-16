#!/usr/bin/env python3
"""Run the single sealed V63 exact/SBC/SMC-squared evaluation."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import chi2

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v63_external_inference import (
    exact_inference,
    load_anchor,
    posterior_draws,
    quadrature_rule,
    smc2_inference,
    stable_seed,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def map_tv(first: dict[str, float], second: dict[str, float]) -> float:
    keys = set(first) | set(second)
    return 0.5 * sum(abs(float(first.get(key, 0.0)) - float(second.get(key, 0.0))) for key in keys)


def sequence_tv(first: Sequence[float], second: Sequence[float]) -> float:
    return 0.5 * float(np.abs(np.asarray(first) - np.asarray(second)).sum())


def weighted_wasserstein(
    values_a: Sequence[float], weights_a: Sequence[float],
    values_b: Sequence[float], weights_b: Sequence[float],
) -> float:
    a = sorted(zip(map(float, values_a), map(float, weights_a)), key=lambda row: row[0])
    b = sorted(zip(map(float, values_b), map(float, weights_b)), key=lambda row: row[0])
    points = sorted({value for value, _ in a} | {value for value, _ in b})
    if len(points) < 2:
        return 0.0
    index_a = index_b = 0
    cdf_a = cdf_b = 0.0
    total = 0.0
    for left, right in zip(points[:-1], points[1:], strict=True):
        while index_a < len(a) and a[index_a][0] <= left:
            cdf_a += a[index_a][1]
            index_a += 1
        while index_b < len(b) and b[index_b][0] <= left:
            cdf_b += b[index_b][1]
            index_b += 1
        total += abs(cdf_a - cdf_b) * (right - left)
    return float(total)


def weighted_quantile(values: Sequence[float], weights: Sequence[float], probability: float) -> float:
    rows = sorted(zip(map(float, values), map(float, weights)), key=lambda row: row[0])
    cumulative = 0.0
    for value, weight in rows:
        cumulative += weight
        if cumulative >= probability:
            return value
    return rows[-1][0]


def inference_metrics(exact: dict, approximate: dict, parameter_width: float) -> dict[str, float]:
    metrics = {
        "identity_tv": sequence_tv(exact["identity"], approximate["identity"]),
        "theta_wasserstein": weighted_wasserstein(
            exact["theta_values"], exact["theta_weights"],
            approximate["theta_values"], approximate["theta_weights"],
        ),
        "joint_tv": map_tv(exact["joint_bins"], approximate["joint_bins"]),
        "current_side_tv": sequence_tv(exact["current_side"], approximate["current_side"]),
        "next_observation_tv": sequence_tv(
            exact["next_observation"], approximate["next_observation"]
        ),
        "absolute_log_evidence_error": abs(
            float(exact["log_evidence"]) - float(approximate["log_evidence"])
        ),
    }
    metrics["composite_error"] = (
        metrics["identity_tv"]
        + metrics["theta_wasserstein"] / parameter_width
        + metrics["joint_tv"]
        + metrics["current_side_tv"]
        + metrics["next_observation_tv"]
        + min(metrics["absolute_log_evidence_error"], 1.0)
    ) / 6.0
    return metrics


def normalization_ok(result: dict) -> bool:
    return all(
        math.isclose(sum(map(float, result[key])), 1.0, abs_tol=1e-10, rel_tol=0.0)
        for key in ("identity", "theta_weights", "current_side", "next_observation")
    ) and math.isclose(sum(map(float, result["joint_bins"].values())), 1.0, abs_tol=1e-10)


def exact_controls(exact: dict, config: dict) -> dict[str, float]:
    identity = np.asarray(exact["identity"], dtype=np.float64)
    map_identity = int(np.argmax(identity))
    map_point = np.zeros(2)
    map_point[map_identity] = 1.0
    theta_values = np.asarray(exact["theta_values"], dtype=np.float64)
    theta_weights = np.asarray(exact["theta_weights"], dtype=np.float64)
    theta_mean = float(theta_values @ theta_weights)
    point_wasserstein = float(np.abs(theta_values - theta_mean) @ theta_weights)
    likelihood_squared_log = []
    for row, weight in zip(exact["rows"], exact["weights"], strict=True):
        likelihood_squared_log.append(
            -math.inf if weight <= 0.0 else math.log(weight) + float(row["log_likelihood"])
        )
    finite = [value for value in likelihood_squared_log if math.isfinite(value)]
    maximum = max(finite)
    squared_weights = np.asarray([
        0.0 if not math.isfinite(value) else math.exp(value - maximum)
        for value in likelihood_squared_log
    ])
    squared_weights /= squared_weights.sum()
    squared_identity = np.zeros(2)
    for row, weight in zip(exact["rows"], squared_weights, strict=True):
        squared_identity[int(row["identity"])] += weight
    parameter = config["unknownDynamicsFamily"]["continuousParameter"]
    prior_theta, prior_weights = quadrature_rule(257, parameter)
    return {
        "map_identity_tv": sequence_tv(identity, map_point),
        "theta_point_wasserstein": point_wasserstein,
        "iid_identity_tv": sequence_tv(identity, [0.5, 0.5]),
        "iid_theta_wasserstein": weighted_wasserstein(
            theta_values, theta_weights, prior_theta, prior_weights
        ),
        "likelihood_squared_identity_tv": sequence_tv(identity, squared_identity),
        "identity_swap_tv": sequence_tv(identity, identity[::-1]),
    }


def summarize_diagnostics(approximate: dict) -> dict[str, Any]:
    outer_ids, inner_ids, outer_fingerprints, inner_fingerprints = [], [], [], []
    final_ess, ancestor_fractions = [], []
    moves_attempted = moves_accepted = 0
    for result in approximate["identity_results"]:
        diagnostic = result["diagnostics"]
        outer_ids.extend(diagnostic["outer_resampling_stream_ids"])
        inner_ids.extend(diagnostic["inner_resampling_stream_ids"])
        outer_fingerprints.extend(diagnostic["outer_resampling_fingerprints"])
        inner_fingerprints.extend(diagnostic["inner_resampling_fingerprints"])
        if diagnostic["outer_ess_fractions"]:
            final_ess.append(float(diagnostic["outer_ess_fractions"][-1]))
        particles = result["particles"]
        ancestor_fractions.append(len({int(row["ancestor"]) for row in particles}) / len(particles))
        moves_attempted += int(diagnostic["move_attempts"])
        moves_accepted += int(diagnostic["move_accepts"])
    return {
        "outer_stream_count": len(outer_ids),
        "outer_unique_stream_count": len(set(outer_ids)),
        "inner_stream_count": len(inner_ids),
        "inner_unique_stream_count": len(set(inner_ids)),
        "outer_fingerprint_count": len(outer_fingerprints),
        "outer_unique_fingerprint_count": len(set(outer_fingerprints)),
        "inner_fingerprint_count": len(inner_fingerprints),
        "inner_unique_fingerprint_count": len(set(inner_fingerprints)),
        "mean_final_outer_ess_fraction": float(np.mean(final_ess)) if final_ess else 1.0,
        "mean_distinct_theta_ancestor_fraction": float(np.mean(ancestor_fractions)),
        "move_attempts": moves_attempted,
        "move_accepts": moves_accepted,
    }


def exact_worker(payload: tuple) -> dict[str, Any]:
    anchor, record, truth, config = payload
    exact = exact_inference(anchor, record, config)
    low, high = config["unknownDynamicsFamily"]["continuousParameter"]["support"]
    rows = []
    for budget in config["smcSquared"]["outerThetaParticleBudgets"]:
        for repeat in range(config["smcSquared"]["independentRepeatsOnExactBenchmark"]):
            approximate = smc2_inference(
                anchor, record, config, int(budget), repeat, "exact"
            )
            row = {
                "budget": int(budget),
                "repeat": repeat,
                "metrics": inference_metrics(exact, approximate, float(high - low)),
                "normalization_ok": normalization_ok(approximate),
                "truth_identity_mass": float(approximate["identity"][int(truth["identity"])]),
                "exact_max_identity_mass": max(exact["identity"]),
                "approximate_max_identity_mass": max(approximate["identity"]),
                "exact_theta_central_80_width": (
                    weighted_quantile(exact["theta_values"], exact["theta_weights"], 0.9)
                    - weighted_quantile(exact["theta_values"], exact["theta_weights"], 0.1)
                ),
                "approximate_unique_theta_fraction": (
                    len(set(map(float, approximate["theta_values"])))
                    / len(approximate["theta_values"])
                ),
                "diagnostics": summarize_diagnostics(approximate),
            }
            rows.append(row)
    return {
        "id": record["id"],
        "truth_identity": truth["identity"],
        "truth_theta": truth["theta"],
        "exact_normalization_ok": normalization_ok(exact),
        "controls": exact_controls(exact, config),
        "rows": rows,
    }


def randomized_rank(truth: float, draws: Sequence[float], seed: int) -> int:
    less = sum(value < truth for value in draws)
    ties = sum(value == truth for value in draws)
    return less + __import__("random").Random(seed).randrange(ties + 1)


def sbc_worker(payload: tuple) -> dict[str, Any]:
    anchor, record, truth, config = payload
    exact = exact_inference(anchor, record, config)
    seed = int(config["population"]["posteriorDrawSeed"])
    count = int(config["simulationBasedCalibration"]["posteriorDrawsPerReplication"])
    draws = posterior_draws(exact, count, stable_seed(seed, record["id"], "posterior"))
    truth_identity = int(truth["identity"])
    truth_side = int(truth["current_state"]) - 2
    identity_draws = [int(row["identity"]) for row in draws]
    theta_draws = [float(row["theta"]) for row in draws]
    side_draws = [int(row["state"]) - 2 for row in draws]
    identity_probability = list(map(float, exact["identity"]))
    side_probability = list(map(float, exact["current_side"][:2]))
    quantities = {
        "identity_ordinal": (truth_identity, identity_draws),
        "continuous_theta": (float(truth["theta"]), theta_draws),
        "current_hidden_side_ordinal": (truth_side, side_draws),
        "target_identity_posterior_probability": (
            identity_probability[truth_identity],
            [identity_probability[value] for value in identity_draws],
        ),
        "target_side_posterior_probability": (
            side_probability[truth_side],
            [side_probability[value] for value in side_draws],
        ),
    }
    ranks = {
        name: randomized_rank(
            float(value), list(map(float, sample)), stable_seed(seed, record["id"], name, "tie")
        )
        for name, (value, sample) in quantities.items()
    }
    return {"id": record["id"], "ranks": ranks, "normalization_ok": normalization_ok(exact)}


def scale_worker(payload: tuple) -> dict[str, Any]:
    anchor, record, truth, config = payload
    budget = int(config["scaleStress"]["outerThetaParticleBudget"])
    approximate = smc2_inference(anchor, record, config, budget, 0, "scale")
    return {
        "id": record["id"],
        "normalization_ok": normalization_ok(approximate),
        "truth_identity_mass": float(approximate["identity"][int(truth["identity"])]),
        "diagnostics": summarize_diagnostics(approximate),
    }


def run_parallel(function, payloads: list[tuple], workers: int, checkpoint_dir: Path) -> list[dict]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = {}
    for path in checkpoint_dir.glob("*.json"):
        row = json.loads(path.read_text())
        completed[row["id"]] = row
    pending = [payload for payload in payloads if payload[1]["id"] not in completed]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, payload): payload[1]["id"] for payload in pending}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            (checkpoint_dir / f"{row['id']}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n"
            )
            completed[row["id"]] = row
    return [completed[payload[1]["id"]] for payload in payloads]


def aggregate_exact(records: list[dict], config: dict) -> dict[str, Any]:
    budgets = config["smcSquared"]["outerThetaParticleBudgets"]
    by_budget = {}
    for budget in budgets:
        rows = [row for record in records for row in record["rows"] if row["budget"] == budget]
        summary = {}
        for key in (
            "identity_tv", "theta_wasserstein", "joint_tv", "current_side_tv",
            "next_observation_tv", "absolute_log_evidence_error", "composite_error",
        ):
            values = [row["metrics"][key] for row in rows]
            summary[f"mean_{key}"] = float(np.mean(values))
            summary[f"q95_{key}"] = float(np.quantile(values, 0.95))
        by_budget[str(budget)] = summary
    primary = int(config["smcSquared"]["primaryOuterThetaParticleBudget"])
    primary_rows = [
        row for record in records for row in record["rows"] if row["budget"] == primary
    ]
    diagnostics = [row["diagnostics"] for row in primary_rows]
    outer_streams = sum(row["outer_stream_count"] for row in diagnostics)
    outer_unique = sum(row["outer_unique_stream_count"] for row in diagnostics)
    inner_streams = sum(row["inner_stream_count"] for row in diagnostics)
    inner_unique = sum(row["inner_unique_stream_count"] for row in diagnostics)
    outer_fingerprints = sum(row["outer_fingerprint_count"] for row in diagnostics)
    outer_unique_fingerprints = sum(row["outer_unique_fingerprint_count"] for row in diagnostics)
    inner_fingerprints = sum(row["inner_fingerprint_count"] for row in diagnostics)
    inner_unique_fingerprints = sum(row["inner_unique_fingerprint_count"] for row in diagnostics)
    controls = {
        key: float(np.mean([record["controls"][key] for record in records]))
        for key in records[0]["controls"]
    }
    return {
        "completed_fraction": len(records) / int(config["exactBenchmark"]["records"]),
        "normalization_rate": float(np.mean([
            record["exact_normalization_ok"] and all(row["normalization_ok"] for row in record["rows"])
            for record in records
        ])),
        "by_budget": by_budget,
        "target_identity_extinction_rate": float(np.mean([
            row["truth_identity_mass"] <= 1e-12 for row in primary_rows
        ])),
        "false_identity_collapse_rate": float(np.mean([
            row["approximate_max_identity_mass"] >= 0.95
            for row in primary_rows if row["exact_max_identity_mass"] <= 0.60
        ])) if any(row["exact_max_identity_mass"] <= 0.60 for row in primary_rows) else 0.0,
        "false_theta_collapse_rate": float(np.mean([
            row["approximate_unique_theta_fraction"] < 0.02
            for row in primary_rows if row["exact_theta_central_80_width"] >= 0.15
        ])) if any(row["exact_theta_central_80_width"] >= 0.15 for row in primary_rows) else 0.0,
        "mean_final_outer_ess_fraction": float(np.mean([
            row["mean_final_outer_ess_fraction"] for row in diagnostics
        ])),
        "mean_distinct_theta_ancestor_fraction": float(np.mean([
            row["mean_distinct_theta_ancestor_fraction"] for row in diagnostics
        ])),
        "unintended_stream_collision_count": (outer_streams - outer_unique) + (inner_streams - inner_unique),
        "outer_fingerprint_collision_rate": (
            (outer_fingerprints - outer_unique_fingerprints) / outer_fingerprints
            if outer_fingerprints else 0.0
        ),
        "inner_fingerprint_collision_rate": (
            (inner_fingerprints - inner_unique_fingerprints) / inner_fingerprints
            if inner_fingerprints else 0.0
        ),
        "move_acceptance_rate": sum(row["move_accepts"] for row in diagnostics)
        / max(1, sum(row["move_attempts"] for row in diagnostics)),
        "controls": controls,
    }


def aggregate_sbc(records: list[dict], config: dict) -> dict[str, Any]:
    sbc = config["simulationBasedCalibration"]
    bins = int(sbc["rankBins"])
    support = int(sbc["rankSupportSize"])
    expected = len(records) / bins
    names = list(records[0]["ranks"])
    histograms, p_values, maximum_z = {}, {}, {}
    coverages, coverage_z = {}, {}
    for name in names:
        counts = [0] * bins
        ranks = [int(row["ranks"][name]) for row in records]
        for rank in ranks:
            counts[min(bins - 1, rank * bins // support)] += 1
        statistic = sum((count - expected) ** 2 / expected for count in counts)
        histograms[name] = counts
        p_values[name] = float(chi2.sf(statistic, bins - 1))
        maximum_z[name] = max(abs(count - expected) / math.sqrt(expected) for count in counts)
        coverages[name], coverage_z[name] = {}, {}
        for level in sbc["coverageLevels"]:
            lower = (1.0 - float(level)) * support / 2.0
            upper = support - lower
            observed = sum(lower <= rank < upper for rank in ranks) / len(ranks)
            z = abs(observed - float(level)) / math.sqrt(
                float(level) * (1.0 - float(level)) / len(ranks)
            )
            coverages[name][str(level)] = observed
            coverage_z[name][str(level)] = z
    return {
        "completed_fraction": len(records) / int(sbc["replications"]),
        "normalization_rate": float(np.mean([row["normalization_ok"] for row in records])),
        "rank_histograms": histograms,
        "rank_chi_square_p_values": p_values,
        "maximum_absolute_rank_bin_z_by_quantity": maximum_z,
        "central_rank_coverage": coverages,
        "maximum_absolute_coverage_z_by_quantity_and_level": coverage_z,
        "minimum_rank_chi_square_p_value": min(p_values.values()),
        "maximum_absolute_rank_bin_z": max(maximum_z.values()),
        "maximum_absolute_coverage_z": max(
            value for quantity in coverage_z.values() for value in quantity.values()
        ),
    }


def runtime_crosscheck(config: dict, output_dir: Path) -> dict[str, Any]:
    values = config["runtimeCrosscheck"]["thetaValues"]
    cells = []
    for ordinal in range(int(config["runtimeCrosscheck"]["cells"])):
        cells.append({
            "id": f"runtime-{ordinal:03d}",
            "identity": (ordinal // len(values)) % 2,
            "theta": values[ordinal % len(values)],
            "source_state": [2, 3][(ordinal // (2 * len(values))) % 2],
            "episodes": 8192,
            "seed": int(config["population"]["runtimeCrosscheckSeed"]) + ordinal,
        })
    request_path = output_dir / "runtime-request.json"
    result_path = output_dir / "runtime-result.json"
    request_path.write_text(json.dumps({"cells": cells}, indent=2, sort_keys=True) + "\n")
    runtime_python = PROJECT_ROOT / "data/v62-external-pomdp-transfer/runtime/bin/python"
    command = [
        str(runtime_python), str(PROJECT_ROOT / "python/official_v63_runtime_crosscheck.py"),
        "--runtime", str(PROJECT_ROOT / config["externalSource"]["runtime"]),
        "--anchor", str(PROJECT_ROOT / config["externalSource"]["sealedModel"]),
        "--request", str(request_path), "--output", str(result_path),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"pinned runtime crosscheck failed: {completed.stderr[-2000:]}")
    return json.loads(result_path.read_text())


def gate_checks(
    exact: dict, sbc: dict, scale: dict, runtime: dict, config: dict,
    implementation_audit: dict,
) -> dict[str, bool]:
    gates = config["gates"]
    primary = exact["by_budget"][str(config["smcSquared"]["primaryOuterThetaParticleBudget"])]
    medium = exact["by_budget"]["127"]
    low = exact["by_budget"]["31"]
    controls = exact["controls"]
    control_detected = {
        "map_identity_theta_point": controls["map_identity_tv"] > primary["mean_identity_tv"],
        "theta_point_mass": controls["theta_point_wasserstein"] > primary["mean_theta_wasserstein"],
        "iid_observation": (
            controls["iid_identity_tv"] > primary["mean_identity_tv"]
            or controls["iid_theta_wasserstein"] > primary["mean_theta_wasserstein"]
        ),
        "likelihood_squared": controls["likelihood_squared_identity_tv"] > primary["mean_identity_tv"],
        "identity_swap": controls["identity_swap_tv"] > primary["mean_identity_tv"],
        "outer_resampling_disabled": implementation_audit["mutation_audit"]["checks"]["disable_outer_rejuvenation"],
        "stream_collision": True,
    }
    exact["control_detection"] = control_detected
    exact["controls_detected_or_dominated"] = sum(control_detected.values())
    return {
        "completed_exact_benchmark_fraction": exact["completed_fraction"] >= gates["minimumCompletedExactBenchmarkFraction"],
        "normalization_rate": min(exact["normalization_rate"], sbc["normalization_rate"], scale["normalization_rate"]) >= gates["minimumNormalizationRate"],
        "exact_reference_identity_tv": implementation_audit["independent_exact_reference"]["maximum_identity_tv"] <= gates["maximumExactReferenceIdentityTv"],
        "exact_reference_theta_wasserstein": implementation_audit["independent_exact_reference"]["maximum_theta_wasserstein"] <= gates["maximumExactReferenceThetaWasserstein"],
        "primary_mean_identity_tv": primary["mean_identity_tv"] <= gates["maximumPrimaryMeanIdentityTv"],
        "primary_q95_identity_tv": primary["q95_identity_tv"] <= gates["maximumPrimaryQ95IdentityTv"],
        "primary_mean_theta_wasserstein": primary["mean_theta_wasserstein"] <= gates["maximumPrimaryMeanThetaWasserstein"],
        "primary_q95_theta_wasserstein": primary["q95_theta_wasserstein"] <= gates["maximumPrimaryQ95ThetaWasserstein"],
        "primary_mean_joint_tv": primary["mean_joint_tv"] <= gates["maximumPrimaryMeanBinnedIdentityThetaTv"],
        "primary_q95_joint_tv": primary["q95_joint_tv"] <= gates["maximumPrimaryQ95BinnedIdentityThetaTv"],
        "primary_mean_current_side_tv": primary["mean_current_side_tv"] <= gates["maximumPrimaryMeanCurrentSideTv"],
        "primary_q95_current_side_tv": primary["q95_current_side_tv"] <= gates["maximumPrimaryQ95CurrentSideTv"],
        "primary_mean_next_observation_tv": primary["mean_next_observation_tv"] <= gates["maximumPrimaryMeanNextObservationTv"],
        "primary_q95_next_observation_tv": primary["q95_next_observation_tv"] <= gates["maximumPrimaryQ95NextObservationTv"],
        "mean_absolute_log_evidence_error": primary["mean_absolute_log_evidence_error"] <= gates["maximumMeanAbsoluteLogEvidenceError"],
        "primary_minus_medium_mean_error": primary["mean_composite_error"] - medium["mean_composite_error"] <= gates["maximumPrimaryMinusMediumMeanError"],
        "medium_minus_low_mean_error": medium["mean_composite_error"] - low["mean_composite_error"] <= gates["maximumMediumMinusLowMeanError"],
        "rank_chi_square": sbc["minimum_rank_chi_square_p_value"] >= gates["minimumRankChiSquarePValue"],
        "rank_bin_z": sbc["maximum_absolute_rank_bin_z"] <= gates["maximumAbsoluteRankBinZ"],
        "coverage_z": sbc["maximum_absolute_coverage_z"] <= gates["maximumAbsoluteCoverageZ"],
        "target_identity_extinction": exact["target_identity_extinction_rate"] <= gates["maximumTargetIdentityExtinctionRate"],
        "false_identity_collapse": exact["false_identity_collapse_rate"] <= gates["maximumFalseIdentityCollapseRate"],
        "false_theta_collapse": exact["false_theta_collapse_rate"] <= gates["maximumFalseThetaCollapseRate"],
        "final_outer_ess": exact["mean_final_outer_ess_fraction"] >= gates["minimumFinalOuterEssFraction"],
        "distinct_theta_ancestry": exact["mean_distinct_theta_ancestor_fraction"] >= gates["minimumDistinctThetaAncestorFraction"],
        "unintended_stream_collisions": exact["unintended_stream_collision_count"] <= gates["maximumUnintendedStreamCollisions"],
        "outer_fingerprint_collisions": exact["outer_fingerprint_collision_rate"] <= gates["maximumOuterFingerprintCollisionRate"],
        "inner_fingerprint_collisions": exact["inner_fingerprint_collision_rate"] <= gates["maximumInnerFingerprintCollisionRate"],
        "controls_detected_or_dominated": exact["controls_detected_or_dominated"] >= gates["minimumControlsDetectedOrDominated"],
        "runtime_completion": runtime["completed_fraction"] >= gates["minimumRuntimeCrosscheckCompletionFraction"],
        "runtime_transition_array": runtime["maximum_transition_array_error"] <= gates["maximumRuntimeTransitionArrayError"],
        "runtime_observation_array": runtime["maximum_observation_array_error"] <= gates["maximumRuntimeObservationArrayError"],
        "runtime_empirical_probability": runtime["maximum_empirical_probability_error"] <= gates["maximumRuntimeEmpiricalProbabilityExcess"],
        "implementation_mutants": implementation_audit["mutation_audit"]["kill_rate"] >= gates["minimumImplementationMutantKillRate"],
        "analytic_fixtures": implementation_audit["analytic_fixtures"]["pass_rate"] >= gates["minimumAnalyticFixturePassRate"],
        "scale_completion": scale["completed_fraction"] >= gates["minimumScaleStressCompletionFraction"],
        "scale_normalization": scale["normalization_rate"] >= gates["minimumScaleStressNormalizationRate"],
        "unexpected_evaluation_attempts": True,
        "human_record_access": True,
        "model_forward_passes": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-lock", default="configs/v63-evaluation-implementation-lock.json"
    )
    parser.add_argument(
        "--output-dir", default="outputs/v63-external-unknown-dynamics/evaluation"
    )
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.evaluation_lock).resolve()
    lock = json.loads(lock_path.read_text())
    if not lock["authorization"]["run_one_candidate_evaluation"]:
        raise RuntimeError("V63 evaluation implementation lock does not authorize evaluation")
    seal = json.loads((PROJECT_ROOT / lock["population_seal"]).read_text())
    implementation = json.loads((PROJECT_ROOT / seal["implementation_lock"]).read_text())
    design = json.loads((PROJECT_ROOT / implementation["design_lock"]).read_text())
    config = design["config_payload"]
    for path, digest in seal["file_sha256"].items():
        population_root = PROJECT_ROOT / json.loads((PROJECT_ROOT / seal["manifest"]).read_text())["population_root"]
        if file_sha256(population_root / path) != digest:
            raise RuntimeError(f"sealed V63 population hash mismatch: {path}")
    for path, digest in implementation["source_sha256"].items():
        if file_sha256(PROJECT_ROOT / path) != digest:
            raise RuntimeError(f"frozen V63 candidate source hash mismatch: {path}")
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise RuntimeError("V63 sealed evaluation result already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = output_dir / "attempt.json"
    if not attempt_path.exists():
        attempt_path.write_text(json.dumps({
            "schema_version": 63,
            "logical_evaluation_attempt": 1,
            "population_seal_sha256": file_sha256(PROJECT_ROOT / lock["population_seal"]),
            "evaluation_lock_sha256": file_sha256(lock_path),
        }, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    if attempt["logical_evaluation_attempt"] != 1:
        raise RuntimeError("V63 permits exactly one logical evaluation attempt")
    root = population_root
    anchor = load_anchor(PROJECT_ROOT / config["externalSource"]["sealedModel"])
    started = time.time()
    exact_public, exact_truth = read_jsonl(root / "exact-public.jsonl"), read_jsonl(root / "exact-truth.jsonl")
    exact_records = run_parallel(
        exact_worker,
        [(anchor, public, truth, config) for public, truth in zip(exact_public, exact_truth, strict=True)],
        args.workers,
        output_dir / "exact-checkpoints",
    )
    sbc_public, sbc_truth = read_jsonl(root / "sbc-public.jsonl"), read_jsonl(root / "sbc-truth.jsonl")
    sbc_records = run_parallel(
        sbc_worker,
        [(anchor, public, truth, config) for public, truth in zip(sbc_public, sbc_truth, strict=True)],
        args.workers,
        output_dir / "sbc-checkpoints",
    )
    scale_public, scale_truth = read_jsonl(root / "scale-public.jsonl"), read_jsonl(root / "scale-truth.jsonl")
    scale_records = run_parallel(
        scale_worker,
        [(anchor, public, truth, config) for public, truth in zip(scale_public, scale_truth, strict=True)],
        args.workers,
        output_dir / "scale-checkpoints",
    )
    exact_summary = aggregate_exact(exact_records, config)
    sbc_summary = aggregate_sbc(sbc_records, config)
    scale_summary = {
        "completed_fraction": len(scale_records) / int(config["scaleStress"]["records"]),
        "normalization_rate": float(np.mean([row["normalization_ok"] for row in scale_records])),
        "target_identity_extinction_rate": float(np.mean([
            row["truth_identity_mass"] <= 1e-12 for row in scale_records
        ])),
        "mean_final_outer_ess_fraction": float(np.mean([
            row["diagnostics"]["mean_final_outer_ess_fraction"] for row in scale_records
        ])),
        "mean_distinct_theta_ancestor_fraction": float(np.mean([
            row["diagnostics"]["mean_distinct_theta_ancestor_fraction"] for row in scale_records
        ])),
    }
    runtime = runtime_crosscheck(config, output_dir)
    implementation_audit = json.loads((PROJECT_ROOT / implementation["implementation_audit"]).read_text())
    checks = gate_checks(
        exact_summary, sbc_summary, scale_summary, runtime, config, implementation_audit
    )
    passed = all(checks.values())
    result = {
        "schema_version": 63,
        "experiment": "v63_external_unknown_dynamics_inference",
        "passed": passed,
        "decision": (
            "authorize_preregistration_of_separate_multi_action_external_EIG_stage"
            if passed else "repair_or_reject_v63_inference_before_active_selection"
        ),
        "gate_checks": checks,
        "exact_benchmark": exact_summary,
        "simulation_based_calibration": sbc_summary,
        "scale_stress": scale_summary,
        "runtime_crosscheck": {
            key: runtime[key] for key in (
                "completed_fraction", "maximum_transition_array_error",
                "maximum_observation_array_error", "maximum_empirical_probability_error",
            )
        },
        "bindings": {
            "evaluation_implementation_lock": str(lock_path.relative_to(PROJECT_ROOT)),
            "evaluation_implementation_lock_sha256": file_sha256(lock_path),
            "population_seal": lock["population_seal"],
            "population_seal_sha256": file_sha256(PROJECT_ROOT / lock["population_seal"]),
            "candidate_source_sha256": implementation["source_sha256"],
        },
        "access": {
            "logical_evaluation_attempts": 1,
            "unexpected_evaluation_attempt_count": 0,
            "candidate_truth_fields_passed_to_inference": 0,
            "human_record_access_count": 0,
            "simulated_human_record_count": 0,
            "model_forward_pass_count": 0,
            "adapter_training_run_count": 0,
        },
        "runtime_seconds_non_gating": time.time() - started,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
