#!/usr/bin/env python3
"""Run the single locked V23 exposed-data probabilistic support replay."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from audit_v22r2_grounding import read_jsonl_directory
from evaluate_v22r2_relational_grounding import load_npz, pair_features
from run_v22_oracle_baselines import outcome_vocabulary
from v10_protocol import file_sha256
from v22_relational import enumerate_program_hypotheses, execute_partial, rows_to_epistemic
from v22r2_grounding import PROJECT_ROOT
from v23_probabilistic_relational import (
    credible_indices,
    k_best_assignments,
    k_best_independent,
    normalized_top_graphs,
    stable_log,
)


TRUTH_VALUES = {"false": [False], "true": [True], "unknown": [False, True]}


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def feature_maps(arrays: dict[str, np.ndarray]):
    return (
        {str(key): arrays["candidate_features"][i] for i, key in enumerate(arrays["candidate_ids"].tolist())},
        {str(key): arrays["evidence_features"][i] for i, key in enumerate(arrays["evidence_ids"].tolist())},
    )


def enumerate_scene_graphs(
    scene: dict[str, Any], candidate_features: dict[str, np.ndarray],
    evidence_features: dict[str, np.ndarray], heads: dict[str, np.ndarray],
    maximum_budget: int, maximum_unknown: int,
) -> list[dict[str, Any]]:
    candidate_ids = [row["id"] for row in scene["agent_input"]["atom_candidates"]]
    evidence_ids = [row["id"] for row in scene["agent_input"]["evidence"]]
    candidate_x = np.stack([candidate_features[value] for value in candidate_ids])
    evidence_x = np.stack([evidence_features[value] for value in evidence_ids])
    atom_coef = heads["atom_coef"][0]
    atom_intercept = float(heads["atom_intercept"][0])
    atom_log_scores = np.empty((len(evidence_ids), len(candidate_ids)), dtype=np.float64)
    for index, vector in enumerate(evidence_x):
        probabilities = sigmoid(
            pair_features(vector[None, :], candidate_x) @ atom_coef + atom_intercept
        )
        atom_log_scores[index] = np.log(np.maximum(probabilities, 1e-12))
    assignments = k_best_assignments(atom_log_scores, maximum_budget)

    truth_classes = heads["truth_classes"].tolist()
    truth_logits = evidence_x @ heads["truth_coef"].T + heads["truth_intercept"]
    truth_probabilities = sigmoid(truth_logits)
    truth_probabilities /= truth_probabilities.sum(axis=1, keepdims=True)
    truth_choices = [
        [(truth_classes[index], float(probability)) for index, probability in enumerate(row)]
        for row in truth_probabilities
    ]
    truth_vectors = k_best_independent(truth_choices, maximum_budget)
    graphs = normalized_top_graphs(
        assignments, truth_vectors, maximum_budget, maximum_unknown
    )
    atom_by_candidate = {
        row["candidate_id"]: row["atom"] for row in scene["target"]["atom_groundings"]
    }
    for graph in graphs:
        rows = []
        for evidence_index, candidate_index in enumerate(graph["assignment"]):
            rows.append({
                "atom": atom_by_candidate[candidate_ids[candidate_index]],
                "allowed_values": TRUTH_VALUES[graph["truth"][evidence_index]],
            })
        graph["epistemic_state"] = sorted(rows, key=lambda row: row["atom"])
    return graphs


def renormalized_prefix(graphs: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    selected = list(graphs[:budget])
    if not selected:
        return []
    maximum = max(row["log_score"] for row in selected)
    weights = np.asarray([math.exp(row["log_score"] - maximum) for row in selected])
    weights /= weights.sum()
    return [{**row, "probability": float(weight)} for row, weight in zip(selected, weights, strict=True)]


def compatibility_matrix(
    graphs: Sequence[dict[str, Any]], hypotheses: Sequence[Any], support: dict[str, Any],
    observed: str, v22_config: dict[str, Any], maximum_unknown: int,
) -> np.ndarray:
    result = np.zeros((len(graphs), len(hypotheses)), dtype=bool)
    for graph_index, graph in enumerate(graphs):
        state = rows_to_epistemic(graph["epistemic_state"])
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            possible = execute_partial(
                [hypothesis.program], v22_config, support["entities"], state,
                support["action_binding"], maximum_unknown,
            )["possible_transition_codes"]
            result[graph_index, hypothesis_index] = observed in possible
    return result


def program_posterior(
    hypotheses: Sequence[Any], traces: Sequence[tuple[Sequence[dict[str, Any]], np.ndarray]],
) -> np.ndarray:
    log_weights = np.zeros(len(hypotheses), dtype=np.float64)
    for graphs, compatibility in traces:
        probabilities = np.asarray([row["probability"] for row in graphs], dtype=np.float64)
        likelihood = probabilities @ compatibility
        positive = likelihood > 0
        log_weights[~positive] = -np.inf
        active = positive & np.isfinite(log_weights)
        log_weights[active] += np.log(likelihood[active])
    finite = np.isfinite(log_weights)
    if not finite.any():
        return np.zeros(len(hypotheses), dtype=np.float64)
    maximum = np.max(log_weights[finite])
    posterior = np.zeros(len(hypotheses), dtype=np.float64)
    posterior[finite] = np.exp(log_weights[finite] - maximum)
    posterior /= posterior.sum()
    return posterior


def mean(values: Sequence[float | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def evaluate_cell(
    records: Sequence[dict[str, Any]], scenes: dict[str, dict[str, Any]],
    graph_cache: dict[str, list[dict[str, Any]]], compatibility_cache: dict[str, np.ndarray],
    budget: int, credible_mass: float, v22_config: dict[str, Any],
) -> dict[str, Any]:
    query_rows = []
    episode_rows = []
    proposal_rows = []
    started = time.perf_counter()
    for record in records:
        bits = record["agent_input"]["dsl_contract"]["outcome_bits"]
        hypotheses = list(enumerate_program_hypotheses(bits))
        keys = [row.key for row in hypotheses]
        target_index = keys.index(record["target"]["program_key"])
        traces = []
        for support in record["oracle_grounding"]["support"]:
            graphs = renormalized_prefix(graph_cache[support["id"]], budget)
            compatibility = compatibility_cache[support["id"]][:len(graphs)]
            traces.append((graphs, compatibility))
            if graphs:
                probabilities = np.asarray([row["probability"] for row in graphs])
                proposal_rows.append({
                    "top1_probability": float(np.max(probabilities)),
                    "effective_graphs": float(1.0 / np.sum(probabilities ** 2)),
                    "graphs": len(graphs),
                })
        posterior = program_posterior(hypotheses, traces)
        selected = credible_indices(posterior, keys, credible_mass)
        nonzero_target = posterior[target_index] > 0
        credible_target = target_index in selected
        exact_values = []
        for query in record["oracle_grounding"]["queries"]:
            if not selected:
                possible = outcome_vocabulary(bits)
            else:
                state = rows_to_epistemic(query["epistemic_state"])
                possible = execute_partial(
                    [hypotheses[index].program for index in selected],
                    v22_config, query["entities"], state, query["action_binding"],
                    v22_config["limits"]["maximumUnknownAtomsPerQuery"],
                )["possible_transition_codes"]
            target = set(query["possible_transition_codes"])
            predicted = set(possible)
            exact = predicted == target
            exact_values.append(exact)
            query_rows.append({
                "exact": exact, "predicted_size": len(predicted), "target_size": len(target),
                "excess": len(predicted - target), "missing": bool(target - predicted),
                "axis": query["query_axis"],
            })
        episode_rows.append({
            "exact": all(exact_values), "query_accuracy": mean(exact_values),
            "target_nonzero": nonzero_target, "target_credible": credible_target,
            "empty": posterior.sum() == 0, "nonzero_programs": int(np.sum(posterior > 0)),
            "credible_programs": len(selected),
            "target_posterior": float(posterior[target_index]),
        })
    return {
        "branch_budget": budget, "credible_program_mass": credible_mass,
        "episodes": len(episode_rows), "queries": len(query_rows),
        "transition_set_exact_match": mean([row["exact"] for row in query_rows]),
        "episode_macro_transition_set_exact_match": mean([row["query_accuracy"] for row in episode_rows]),
        "complete_episodes": sum(row["exact"] for row in episode_rows),
        "target_nonzero_posterior_retention": mean([row["target_nonzero"] for row in episode_rows]),
        "target_credible_set_retention": mean([row["target_credible"] for row in episode_rows]),
        "empty_posterior_rate": mean([row["empty"] for row in episode_rows]),
        "mean_target_posterior": mean([row["target_posterior"] for row in episode_rows]),
        "median_nonzero_programs": float(np.median([row["nonzero_programs"] for row in episode_rows])),
        "median_credible_programs": float(np.median([row["credible_programs"] for row in episode_rows])),
        "mean_predicted_outcomes": mean([row["predicted_size"] for row in query_rows]),
        "mean_target_outcomes": mean([row["target_size"] for row in query_rows]),
        "mean_excess_outcomes": mean([row["excess"] for row in query_rows]),
        "missing_target_outcome_rate": mean([row["missing"] for row in query_rows]),
        "mean_top1_proposal_probability": mean([row["top1_probability"] for row in proposal_rows]),
        "mean_effective_graphs": mean([row["effective_graphs"] for row in proposal_rows]),
        "runtime_seconds": time.perf_counter() - started,
        "by_axis": {
            axis: {
                "queries": len(rows),
                "transition_set_exact_match": mean([row["exact"] for row in rows]),
                "mean_excess_outcomes": mean([row["excess"] for row in rows]),
            }
            for axis in sorted({row["axis"] for row in query_rows})
            for rows in [[row for row in query_rows if row["axis"] == axis]]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v23-probabilistic-support-lock.json")
    parser.add_argument("--output-dir", default="outputs/v23-probabilistic-support")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "v23-probabilistic-support-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V23 probabilistic replay was already attempted")
    lock = json.loads(lock_path.read_text())
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V23 locked implementation changed: {path}")
    if lock["limits"]["probabilisticReplayRuns"] != 1:
        raise RuntimeError("V23 lock does not authorize exactly one replay")
    metadata = json.loads((PROJECT_ROOT / lock["source"]["feature_metadata"]).read_text())
    arrays = load_npz(PROJECT_ROOT / metadata["feature_artifact"])
    heads = load_npz(PROJECT_ROOT / lock["source"]["heads"])
    original_lock = json.loads((PROJECT_ROOT / lock["source"]["v22r2_original_lock"]).read_text())
    scenes_list = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    scenes = {row["id"]: row for row in scenes_list}
    records = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "records")
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())
    config = lock["config_payload"]
    maximum_budget = max(config["branchBudgets"])
    maximum_unknown = config["interface"]["maximumUnknownAtomsPerProposal"]
    candidate_features, evidence_features = feature_maps(arrays)

    attempt_path.write_text(json.dumps({
        "schema_version": 23, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_probabilistic_replay",
    }, indent=2, sort_keys=True) + "\n")
    graph_cache = {}
    compatibility_cache = {}
    proposal_started = time.perf_counter()
    observed_by_item = {
        row["id"]: row["observed_transition_code"]
        for record in records for row in record["agent_input"]["support_traces"]
    }
    oracle_support = {
        row["id"]: (record, row)
        for record in records for row in record["oracle_grounding"]["support"]
    }
    for scene in scenes_list:
        if scene["role"] != "support":
            continue
        graphs = enumerate_scene_graphs(
            scene, candidate_features, evidence_features, heads,
            maximum_budget, maximum_unknown,
        )
        graph_cache[scene["id"]] = graphs
        record, support = oracle_support[scene["id"]]
        hypotheses = enumerate_program_hypotheses(
            record["agent_input"]["dsl_contract"]["outcome_bits"]
        )
        compatibility_cache[scene["id"]] = compatibility_matrix(
            graphs, hypotheses, support, observed_by_item[scene["id"]],
            v22_config, maximum_unknown,
        )
    proposal_seconds = time.perf_counter() - proposal_started

    curves = {}
    for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation"):
        selected_records = [row for row in records if row["split"] == split]
        curves[split] = {}
        for budget in config["branchBudgets"]:
            curves[split][str(budget)] = {}
            for mass in config["credibleProgramMasses"]:
                curves[split][str(budget)][str(mass)] = evaluate_cell(
                    selected_records, scenes, graph_cache, compatibility_cache,
                    budget, mass, v22_config,
                )
    reference_spec = config["registeredReference"]
    reference = curves["grounding_evaluation"][str(reference_spec["branchBudget"])][str(reference_spec["credibleProgramMass"])]
    gates = config["gates"]
    checks = {
        "target_nonzero_retention": reference["target_nonzero_posterior_retention"] >= gates["referenceMinimumTargetNonzeroRetention"],
        "target_credible_retention": reference["target_credible_set_retention"] >= gates["referenceMinimumTargetCredibleRetention"],
        "empty_posterior": reference["empty_posterior_rate"] <= gates["referenceMaximumEmptyPosteriorRate"],
        "transition_set_exact": reference["transition_set_exact_match"] >= gates["referenceMinimumTransitionSetExact"],
        "excess_outcomes": reference["mean_excess_outcomes"] <= gates["referenceMaximumMeanExcessOutcomes"],
        "missing_target_outcomes": reference["missing_target_outcome_rate"] <= gates["referenceMaximumMissingTargetOutcomeRate"],
    }
    passed = all(checks.values())
    decision = (
        "authorize_fresh_relational_benchmark_protocol_with_probabilistic_support"
        if passed else "probabilistic_support_insufficient_revise_language_interface"
    )
    source_result = json.loads((PROJECT_ROOT / lock["source"]["v22r2_result"]).read_text())
    result = {
        "schema_version": 23, "experiment": config["experiment"],
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "curves": curves, "registered_reference": reference,
        "checks": checks, "passed": passed, "decision": decision,
        "hard_v22r2a_external_ceiling": {
            "frozen_support_oracle_query": source_result["integration"]["frozen_support_oracle_query"],
            "oracle_support_frozen_query": source_result["integration"]["oracle_support_frozen_query"],
            "frozen_support_frozen_query": source_result["integration"]["frozen_support_frozen_query"],
        },
        "proposal_precomputation_seconds": proposal_seconds,
        "data_access": {
            "new_model_forward_passes": 0, "new_feature_extractions": 0,
            "new_linear_fits": 0, "hyperparameter_selections": 0,
            "adapter_training_runs": 0, "probabilistic_replay_runs": 1,
            "final_suite_constructions": 0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({
        "status": "completed", "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
    })
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
