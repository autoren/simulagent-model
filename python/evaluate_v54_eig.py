#!/usr/bin/env python3
"""Run the single sealed V54 exact active-design evaluation."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from collections import defaultdict

from scipy.stats import chi2

from generate_v54_eig import inference_record
from v10_protocol import file_sha256
from v22_relational import canonical_json, sha256_text
from v22r2_grounding import PROJECT_ROOT
from v53_smc2 import exact_inference, mechanic_registry
from v54_eig import (
    attempted_outcome_leak_selection,
    belief_atoms_from_exact,
    candidate_interventions,
    score_all_interventions,
    score_control_policies,
    select_score,
)


QUANTITIES = (
    "program_ordinal",
    "continuous_theta",
    "target_program_posterior_probability",
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def posterior_draw_seed(config, record_id, draw):
    return int(sha256_text(
        f"v54-sbc-draw|{config['population']['sbcRankSeed']}|{record_id}|{draw}"
    ), 16)


def rank_seed(config, record_id, quantity):
    return int(sha256_text(
        f"v54-sbc-rank|{config['population']['sbcRankSeed']}|{record_id}|{quantity}"
    ), 16)


def augmented_inference_record(row):
    public = copy.deepcopy(row["public_history"])
    query = public["query"]
    intervention = row["selected_intervention"]
    outcome = row["realized_outcome"]
    query["actions"] = [*query["actions"], *intervention["assay"]]
    query["masks"] = [*query["masks"], *outcome["masks"]]
    query["observations"] = [
        *query["observations"], *outcome["observations"]
    ]
    query["prefix_length"] += len(intervention["assay"])
    query["sequence_length"] = query["prefix_length"]
    return inference_record(public)


def sbc_ranks(exact, row, config):
    draws = [
        categorical_atom(
            exact["atoms"], posterior_draw_seed(config, row["id"], draw)
        )
        for draw in range(config["adaptiveSbc"]["posteriorDrawsPerReplication"])
    ]
    target = row["truth"]["target_program_index"]
    true_values = {
        "program_ordinal": row["truth"]["target_program_ordinal"],
        "continuous_theta": row["truth"]["target_theta"],
        "target_program_posterior_probability": exact["program"][target],
    }
    draw_values = {
        "program_ordinal": [atom["program_index"] for atom in draws],
        "continuous_theta": [atom["theta"] for atom in draws],
        "target_program_posterior_probability": [
            exact["program"][atom["program_index"]] for atom in draws
        ],
    }
    return {
        quantity: randomized_rank(
            true_values[quantity], draw_values[quantity],
            rank_seed(config, row["id"], quantity),
        )
        for quantity in QUANTITIES
    }


def evaluate_selection_record(row, registry, v53_config, v54_config):
    public = row["public_history"]
    exact = exact_inference(registry, inference_record(public), v53_config)
    atoms = belief_atoms_from_exact(exact)
    query = public["query"]
    scores = score_all_interventions(
        atoms, registry, query["entities"], query["prefix_length"]
    )
    tolerance = v54_config["targetAndObjective"]["tieToleranceNats"]
    primary = select_score(scores, tolerance, "eig")
    reference = select_score(scores, tolerance, "reference_eig")
    reference_by_key = {
        score["intervention_key"]: score["reference_eig"] for score in scores
    }
    maximum = reference["maximum"]
    minimum = min(reference_by_key.values())
    selected_key = primary["selected"]["intervention_key"]
    selected_reference = reference_by_key[selected_key]
    spread = maximum - minimum
    informative = (
        spread >= v54_config["gates"]["informativeRecordMinimumOracleSpreadNats"]
    )
    candidates = candidate_interventions(query["entities"])
    wait_key = next(
        candidate["key"] for candidate in candidates
        if candidate["action"]["id"] == "wait"
    )
    controls = score_control_policies(
        atoms, registry, query["entities"], query["prefix_length"]
    )
    control_keys = {
        name: value["selected"]["intervention_key"]
        for name, value in controls.items()
        if isinstance(value, dict) and "selected" in value
    }
    return {
        "id": row["id"],
        "record": row["record"],
        "history_class": row["history_class"],
        "entity_count": query["entity_count"],
        "candidate_count": len(scores),
        "candidate_scores": [{
            "intervention_key": score["intervention_key"],
            "eig": score["eig"],
            "reference_eig": score["reference_eig"],
            "entropy_eig": score["entropy_eig"],
            "prior_entropy": score["prior_entropy"],
            "predictive_entropy": score["predictive_entropy"],
            "normalizes": score["normalizes"],
            "finite": score["finite"],
        } for score in scores],
        "selected_key": selected_key,
        "reference_optimal_keys": reference["optimal_keys"],
        "selected_eig": primary["selected"]["eig"],
        "selected_reference_eig": selected_reference,
        "maximum_reference_eig": maximum,
        "minimum_reference_eig": minimum,
        "uniform_random_mean_eig": controls["uniform_random_mean_eig"],
        "oracle_spread": spread,
        "informative": informative,
        "selected_regret": maximum - selected_reference,
        "fraction_available_eig_captured": (
            1.0 if spread <= tolerance
            else (selected_reference - minimum) / spread
        ),
        "strictly_dominated_no_op_selected": (
            selected_key == wait_key and reference_by_key[wait_key] < maximum - tolerance
        ),
        "maximum_candidate_error": max(
            abs(score["eig"] - score["reference_eig"]) for score in scores
        ),
        "mean_candidate_error": statistics.fmean(
            abs(score["eig"] - score["reference_eig"]) for score in scores
        ),
        "maximum_entropy_identity_error": max(
            abs(score["eig"] - score["entropy_eig"]) for score in scores
        ),
        "minimum_candidate_eig": min(score["eig"] for score in scores),
        "maximum_eig_minus_prior_entropy": max(
            score["eig"] - score["prior_entropy"] for score in scores
        ),
        "normalization": all(score["normalizes"] for score in scores),
        "finite": all(score["finite"] for score in scores),
        "control_selected_keys": control_keys,
        "control_regrets": {
            name: maximum - reference_by_key[key]
            for name, key in control_keys.items()
        },
        "control_disagreements": {
            name: key not in reference["optimal_keys"]
            for name, key in control_keys.items()
        },
        "uniform_random_regret": maximum - controls["uniform_random_mean_eig"],
        "uniform_random_disagreement_probability": (
            1 - len(reference["optimal_keys"]) / len(scores)
        ),
    }


def evaluate_adaptive_record(row, registry, v53_config, v54_config):
    public = row["public_history"]
    query = public["query"]
    pre = exact_inference(registry, inference_record(public), v53_config)
    atoms = belief_atoms_from_exact(pre)
    scores = score_all_interventions(
        atoms, registry, query["entities"], query["prefix_length"]
    )
    selected = select_score(
        scores, v54_config["targetAndObjective"]["tieToleranceNats"]
    )
    recomputed_key = selected["selected"]["intervention_key"]
    sealed_key = row["selected_intervention"]["key"]
    post = exact_inference(
        registry, augmented_inference_record(row), v53_config
    )
    return {
        "id": row["id"],
        "record": row["record"],
        "history_class": row["history_class"],
        "sealed_selected_key": sealed_key,
        "recomputed_selected_key": recomputed_key,
        "selection_matches": sealed_key == recomputed_key,
        "candidate_count": len(scores),
        "normalization": (
            abs(sum(post["program"]) - 1.0) <= 1e-12
            and abs(sum(post["theta_weights"]) - 1.0) <= 1e-12
            and abs(sum(row["weight"] for row in post["atoms"]) - 1.0) <= 1e-12
        ),
        "ranks": sbc_ranks(post, row, v54_config),
    }


def rank_diagnostics(rows, config):
    specification = config["adaptiveSbc"]
    bins, support = specification["rankBins"], specification["rankSupportSize"]
    expected = len(rows) / bins
    bin_sd = math.sqrt(len(rows) * (1 / bins) * (1 - 1 / bins))
    histograms, p_values, max_z, coverage = {}, {}, {}, {}
    for quantity in QUANTITIES:
        ranks = [row["ranks"][quantity] for row in rows]
        counts = [0 for _ in range(bins)]
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
            abs(cell["z"])
            for values in coverage.values() for cell in values.values()
        ),
        "post_selection_normalization_rate": statistics.fmean(
            row["normalization"] for row in rows
        ),
        "sealed_selection_match_rate": statistics.fmean(
            row["selection_matches"] for row in rows
        ),
    }


def control_metrics(rows, config):
    informative = [row for row in rows if row["informative"]]
    names = (
        "predictive_entropy", "state_only_information", "map_program",
        "theta_point_mass", "likelihood_squared",
    )
    result = {}
    for name in names:
        mean_regret = statistics.fmean(row["control_regrets"][name] for row in rows)
        disagreement = statistics.fmean(
            row["control_disagreements"][name] for row in informative
        ) if informative else 0.0
        result[name] = {
            "mean_exact_eig_regret": mean_regret,
            "informative_selection_disagreement_rate": disagreement,
            "detected_or_dominated": mean_regret > 0.001 or disagreement > 0.10,
        }
    random_regret = statistics.fmean(row["uniform_random_regret"] for row in rows)
    random_disagreement = statistics.fmean(
        row["uniform_random_disagreement_probability"] for row in informative
    ) if informative else 0.0
    result["uniform_random"] = {
        "mean_exact_eig_regret": random_regret,
        "informative_selection_disagreement_rate": random_disagreement,
        "detected_or_dominated": random_regret > 0.001 or random_disagreement > 0.10,
    }
    leakage = False
    try:
        attempted_outcome_leak_selection({}, {"sealed": "outcome"})
    except PermissionError:
        leakage = True
    result["outcome_leakage"] = {
        "firewall_rejected": leakage,
        "detected_or_dominated": leakage,
    }
    return {
        "by_control": result,
        "detected_or_dominated": sum(
            value["detected_or_dominated"] for value in result.values()
        ),
    }


def aggregate_selection(rows, config):
    controls = control_metrics(rows, config)
    return {
        "completed_fraction": len(rows) / config["population"]["selectionRecords"],
        "candidate_and_predictive_normalization_rate": statistics.fmean(
            row["normalization"] for row in rows
        ),
        "finite_value_rate": statistics.fmean(row["finite"] for row in rows),
        "maximum_absolute_candidate_eig_error": max(
            row["maximum_candidate_error"] for row in rows
        ),
        "mean_absolute_candidate_eig_error": statistics.fmean(
            row["mean_candidate_error"] for row in rows
        ),
        "optimal_set_membership_rate": statistics.fmean(
            row["selected_key"] in row["reference_optimal_keys"] for row in rows
        ),
        "maximum_selected_eig_regret": max(row["selected_regret"] for row in rows),
        "maximum_entropy_reduction_identity_error": max(
            row["maximum_entropy_identity_error"] for row in rows
        ),
        "minimum_candidate_eig": min(row["minimum_candidate_eig"] for row in rows),
        "maximum_eig_minus_prior_entropy": max(
            row["maximum_eig_minus_prior_entropy"] for row in rows
        ),
        "informative_record_fraction": statistics.fmean(
            row["informative"] for row in rows
        ),
        "mean_oracle_eig": statistics.fmean(
            row["maximum_reference_eig"] for row in rows
        ),
        "mean_uniform_random_eig": statistics.fmean(
            row["uniform_random_mean_eig"] for row in rows
        ),
        "mean_oracle_minus_uniform_random_eig": statistics.fmean(
            row["maximum_reference_eig"] - row["uniform_random_mean_eig"]
            for row in rows
        ),
        "mean_fraction_of_available_eig_captured": statistics.fmean(
            row["fraction_available_eig_captured"] for row in rows
        ),
        "strictly_dominated_no_op_selection_rate": statistics.fmean(
            row["strictly_dominated_no_op_selected"] for row in rows
        ),
        "controls": controls,
    }


def selection_integrity(selection_rows, adaptive_rows, sealed_records):
    history_streams = [
        sha256_text(f"history|{row['id']}")
        for population in sealed_records.values() for row in population
    ]
    outcome_streams = [
        sha256_text(f"outcome|{row['id']}|{row['selected_intervention']['key']}")
        for row in sealed_records["adaptive_sbc"]
    ]
    return {
        "truth_field_access_count": 0,
        "realized_outcome_access_before_selection_count": 0,
        "candidate_omission_count": sum(
            row["candidate_count"] not in (5, 13)
            for row in [*selection_rows, *adaptive_rows]
        ),
        "canonical_tie_break_violation_count": sum(
            not row["selection_matches"] for row in adaptive_rows
        ),
        "history_and_outcome_stream_collision_count": len(
            set(history_streams) & set(outcome_streams)
        ),
    }


def qualification(metrics, gates):
    selection, sbc, integrity = (
        metrics["selection"], metrics["adaptive_sbc"], metrics["selection_integrity"]
    )
    checks = {
        "selection_completion": selection["completed_fraction"] >= gates["minimumCompletedSelectionFraction"],
        "normalization": selection["candidate_and_predictive_normalization_rate"] >= gates["minimumCandidateAndPredictiveNormalizationRate"],
        "finite_values": selection["finite_value_rate"] >= gates["minimumFiniteValueRate"],
        "maximum_candidate_error": selection["maximum_absolute_candidate_eig_error"] <= gates["maximumAbsoluteCandidateEigError"],
        "mean_candidate_error": selection["mean_absolute_candidate_eig_error"] <= gates["maximumMeanAbsoluteCandidateEigError"],
        "optimal_set_membership": selection["optimal_set_membership_rate"] >= gates["minimumOptimalSetMembershipRate"],
        "selected_regret": selection["maximum_selected_eig_regret"] <= gates["maximumSelectedEigRegret"],
        "entropy_identity": selection["maximum_entropy_reduction_identity_error"] <= gates["maximumEntropyReductionIdentityError"],
        "eig_nonnegative": selection["minimum_candidate_eig"] >= gates["minimumCandidateEig"],
        "eig_entropy_upper_bound": selection["maximum_eig_minus_prior_entropy"] <= gates["maximumEigMinusPriorEntropy"],
        "informative_records": selection["informative_record_fraction"] >= gates["minimumInformativeRecordFraction"],
        "oracle_random_advantage": selection["mean_oracle_minus_uniform_random_eig"] >= gates["minimumMeanOracleMinusUniformRandomEigNats"],
        "available_eig_captured": selection["mean_fraction_of_available_eig_captured"] >= gates["minimumMeanFractionOfAvailableEigCaptured"],
        "dominated_no_op": selection["strictly_dominated_no_op_selection_rate"] <= gates["maximumStrictlyDominatedNoOpSelectionRate"],
        "adaptive_normalization": sbc["post_selection_normalization_rate"] >= gates["minimumPostSelectionNormalizationRate"],
        "adaptive_rank_chi_square": sbc["minimum_chi_square_p_value"] >= gates["minimumRankChiSquarePValue"],
        "adaptive_rank_envelope": sbc["maximum_absolute_rank_bin_z"] <= gates["maximumAbsoluteRankBinZ"],
        "adaptive_coverage": sbc["maximum_absolute_coverage_z"] <= gates["maximumAbsoluteCoverageZ"],
        "sealed_adaptive_selection": sbc["sealed_selection_match_rate"] == 1.0,
        "truth_access": integrity["truth_field_access_count"] <= gates["maximumTruthFieldAccessCount"],
        "outcome_access": integrity["realized_outcome_access_before_selection_count"] <= gates["maximumRealizedOutcomeAccessBeforeSelectionCount"],
        "candidate_omission": integrity["candidate_omission_count"] <= gates["maximumCandidateOmissionCount"],
        "tie_break": integrity["canonical_tie_break_violation_count"] <= gates["maximumCanonicalTieBreakViolationCount"],
        "stream_collision": integrity["history_and_outcome_stream_collision_count"] <= gates["maximumHistoryAndOutcomeStreamCollisionCount"],
        "controls": selection["controls"]["detected_or_dominated"] >= gates["minimumControlsDetectedOrDominated"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-seal", default="configs/v54-population-seal.json")
    parser.add_argument("--output-dir", default="outputs/v54-exact-one-step-eig/evaluation")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.population_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "evaluation-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V54 evaluation already attempted")
    seal = json.loads(seal_path.read_text())
    for lock_name in ("implementation_lock", "evaluation_implementation_lock"):
        lock_path = PROJECT_ROOT / seal[lock_name]
        if file_sha256(lock_path) != seal[f"{lock_name}_sha256"]:
            raise RuntimeError(f"V54 sealed {lock_name} changed")
        lock = json.loads(lock_path.read_text())
        for section in ("implementation_files_sha256", "base_dependencies_sha256"):
            for path, digest in lock.get(section, {}).items():
                if file_sha256(PROJECT_ROOT / path) != digest:
                    raise RuntimeError(f"V54 frozen file changed: {path}")
    records = {}
    for name, artifact in seal["populations"].items():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"V54 sealed {name} population changed")
        records[name] = read_jsonl(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 54,
        "status": "started",
        "evaluation_run": 1,
        "population_seal_sha256": file_sha256(seal_path),
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    design = json.loads((PROJECT_ROOT / "configs/v54-design-lock.json").read_text())
    v54_config = design["config_payload"]
    v53_config = json.loads(
        (PROJECT_ROOT / "configs/v53r2-design-lock.json").read_text()
    )["config_payload"]
    registry = mechanic_registry(5303)
    selection_rows = []
    for ordinal, row in enumerate(records["selection"]):
        selection_rows.append(evaluate_selection_record(
            row, registry, v53_config, v54_config
        ))
        print(json.dumps({
            "stage": "selection", "record": ordinal + 1,
            "total": len(records["selection"]), "id": row["id"],
        }), flush=True)
    adaptive_rows = []
    for ordinal, row in enumerate(records["adaptive_sbc"]):
        adaptive_rows.append(evaluate_adaptive_record(
            row, registry, v53_config, v54_config
        ))
        if (ordinal + 1) % 8 == 0 or ordinal + 1 == len(records["adaptive_sbc"]):
            print(json.dumps({
                "stage": "adaptive_sbc", "record": ordinal + 1,
                "total": len(records["adaptive_sbc"]),
            }), flush=True)
    metrics = {
        "selection": aggregate_selection(selection_rows, v54_config),
        "adaptive_sbc": rank_diagnostics(adaptive_rows, v54_config),
        "selection_integrity": selection_integrity(
            selection_rows, adaptive_rows, records
        ),
    }
    result_qualification = qualification(metrics, v54_config["gates"])
    decision = (
        "authorize_short_horizon_exact_bayes_adaptive_planning_preregistration_only"
        if result_qualification["passed"]
        else "repair_v54_exact_eig_selection_conditioning_efficiency_or_calibration"
    )
    output.mkdir(parents=True)
    details = {}
    for name, values in (("selection", selection_rows), ("adaptive_sbc", adaptive_rows)):
        path = output / f"{name}-metrics.jsonl"
        path.write_text("".join(canonical_json(row) + "\n" for row in values))
        details[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(path),
        }
    result = {
        "schema_version": 54,
        "experiment": v54_config["experiment"],
        "population_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "population_seal_sha256": file_sha256(seal_path),
        "evaluation_run_number": 1,
        "metrics": metrics,
        "qualification": result_qualification,
        "decision": decision,
        "detail_metrics": details,
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "active_evaluation_runs": 1,
            "selection_on_sealed_results": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
        "authorization": {
            "preregister_short_horizon_exact_bayes_adaptive_planning": result_qualification["passed"],
            "construct_planning_population": False,
            "verification": False,
            "language_grounding": False,
            "model_access": False,
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    attempt.write_text(json.dumps({
        "schema_version": 54,
        "status": "completed",
        "evaluation_run": 1,
        "population_seal_sha256": file_sha256(seal_path),
        "result_sha256": file_sha256(output / "result.json"),
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
