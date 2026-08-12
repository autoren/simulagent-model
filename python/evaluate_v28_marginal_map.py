#!/usr/bin/env python3
"""Run the single locked V28 marginal program MAP evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import numpy as np

from audit_v22r2_grounding import read_jsonl_directory
from audit_v24_cross_encoder import read_pairs
from evaluate_v22r2_relational_grounding import (
    condition_modes, grounding_summary, integration_condition, load_npz, mean,
)
from v10_protocol import file_sha256
from v22_relational import enumerate_program_hypotheses
from v22r2_grounding import PROJECT_ROOT, predicted_epistemic_rows, validate_scene_prediction
from v27_support_map import enumerate_scene_graphs
from v28_marginal_map import compatibility_matrix_deduplicated, select_marginal_episode_map


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v28-marginal-map-lock.json")
    parser.add_argument("--output-dir", default="outputs/v28-marginal-map/evaluation")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V28 marginal MAP evaluation was already attempted")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["marginalMapEvaluations"] != 1 or lock["limits"]["newModelForwardPasses"] != 0:
        raise RuntimeError("V28 lock does not authorize the registered zero-forward evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V28 locked implementation changed: {path}")
    source_keys = (
        "sourceV27Lock", "sourceV27Result", "sourceV27PostAudit",
        "sourceV27EdgeMetadata", "sourceV27EdgeScores", "sourceV27Predictions",
        "sourceV27Diagnostics", "sourceV27NativeMatchDiagnostic",
    )
    for key in source_keys:
        if file_sha256(PROJECT_ROOT / lock["source"][key]) != lock["source"][f"{key}_sha256"]:
            raise RuntimeError(f"V28 locked source changed: {key}")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 28, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_marginal_map",
    }, indent=2, sort_keys=True) + "\n")

    config = lock["config_payload"]
    v27_lock = json.loads((PROJECT_ROOT / config["sourceV27Lock"]).read_text())
    locked_v27_sources = {
        "v26_lock": v27_lock["source"]["v26_lock_sha256"],
        "v26_result": v27_lock["source"]["v26_result_sha256"],
        "v26_post_audit": v27_lock["source"]["v26_post_audit_sha256"],
        "v26_residual": v27_lock["source"]["v26_residual_sha256"],
        "v26_scores": v27_lock["source"]["v26_scores_sha256"],
        "v26_predictions": v27_lock["source"]["v26_predictions_sha256"],
        "v24_feature_metadata": v27_lock["source"]["v24_feature_metadata_sha256"],
        "v24_feature_artifact": v27_lock["source"]["v24_feature_artifact_sha256"],
        "v24_heads": v27_lock["source"]["v24_heads_sha256"],
    }
    for name, expected in locked_v27_sources.items():
        if file_sha256(PROJECT_ROOT / v27_lock["source"][name]) != expected:
            raise RuntimeError(f"V28 locked V27 upstream artifact changed: {name}")
    proposal_manifest = PROJECT_ROOT / v27_lock["source"]["v24_proposal_corpus"] / "manifest.json"
    if file_sha256(proposal_manifest) != v27_lock["source"]["v24_proposal_manifest_sha256"]:
        raise RuntimeError("V28 locked V24 proposal corpus changed")
    v26_lock = json.loads((PROJECT_ROOT / v27_lock["source"]["v26_lock"]).read_text())
    v25_lock = json.loads((PROJECT_ROOT / v26_lock["source"]["v25_lock"]).read_text())
    v24_lock = json.loads((PROJECT_ROOT / v25_lock["source"]["v24_lock"]).read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    scene_lookup = {row["id"]: row for row in scenes}
    records = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "records")
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())
    v27_config = v27_lock["config_payload"]
    proposal_pairs = [
        row for row in read_pairs(PROJECT_ROOT / v27_config["sourceV24ProposalCorpus"])
        if row["role"] == "support"
    ]
    proposals_by_scene = defaultdict(list)
    for row in proposal_pairs:
        proposals_by_scene[row["scene_id"]].append(row)
    feature_metadata = json.loads((PROJECT_ROOT / v27_config["sourceV24Features"]).read_text())
    arrays = load_npz(PROJECT_ROOT / feature_metadata["feature_artifact"])
    feature_lookup = {
        str(identifier): arrays["pair_features"][index]
        for index, identifier in enumerate(arrays["pair_ids"].tolist())
    }
    heads = load_npz(PROJECT_ROOT / v27_config["sourceV24Heads"])
    match_coef = heads["match_coef"][0]
    match_intercept = float(heads["match_intercept"][0])
    v27_edge_metadata = json.loads((PROJECT_ROOT / config["sourceV27EdgeMetadata"]).read_text())
    score_rows = [
        *[row for row in jsonl(PROJECT_ROOT / v27_config["sourceV26Scores"]) if row["role"] == "support"],
        *jsonl(PROJECT_ROOT / v27_edge_metadata["score_artifact"]),
    ]
    truth_logits = {
        (row["scene_id"], row["evidence_id"], row["candidate_id"]): row["fp32_direct_logits"]
        for row in score_rows
    }
    expected_edges = {
        (row["scene_id"], row["evidence_id"], row["candidate_id"])
        for row in proposal_pairs
    }
    if set(truth_logits) != expected_edges:
        raise RuntimeError("V28 truth scores do not cover exactly all support proposal edges")

    started = time.perf_counter()
    graph_cache = {}
    support_scenes = [row for row in scenes if row["role"] == "support"]
    for index, scene in enumerate(support_scenes, start=1):
        graph_cache[scene["id"]] = enumerate_scene_graphs(
            scene, proposals_by_scene[scene["id"]], feature_lookup,
            match_coef, match_intercept, truth_logits, v27_config,
        )
        if index % 10 == 0 or index == len(support_scenes):
            print(f"v28 marginal MAP: enumerated {index}/{len(support_scenes)} support scenes", file=sys.stderr, flush=True)

    v27_predictions = jsonl(PROJECT_ROOT / config["sourceV27Predictions"])
    v27_lookup = {row["scene_id"]: row for row in v27_predictions}
    selected_support = {}
    episode_diagnostics = []
    observed = {
        row["id"]: row["observed_transition_code"]
        for record in records for row in record["agent_input"]["support_traces"]
    }
    total_unique_states = 0
    total_graph_rows = 0
    for record_index, record in enumerate(records, start=1):
        hypotheses = list(enumerate_program_hypotheses(
            record["agent_input"]["dsl_contract"]["outcome_bits"]
        ))
        graphs_by_trace = []
        compatibility_by_trace = []
        unique_states = 0
        for support in record["oracle_grounding"]["support"]:
            graphs = graph_cache[support["id"]]
            compatibility, unique = compatibility_matrix_deduplicated(
                graphs, hypotheses, support, observed[support["id"]], v22_config,
                v27_config["jointMap"]["maximumUnknownAtomsPerGraph"],
            )
            graphs_by_trace.append(graphs)
            compatibility_by_trace.append(compatibility)
            unique_states += unique
            total_unique_states += unique
            total_graph_rows += len(graphs)
        selection = select_marginal_episode_map(
            hypotheses, graphs_by_trace, compatibility_by_trace
        )
        fallback = selection is None
        target_index = [row.key for row in hypotheses].index(record["target"]["program_key"])
        if fallback:
            for support in record["oracle_grounding"]["support"]:
                selected_support[support["id"]] = v27_lookup[support["id"]]
            selected_program = None
            selected_target = False
            target_posterior = 0.0
            target_rank = None
            finite_programs = 0
            maximum_posterior = 0.0
        else:
            posterior = selection["posterior"]
            selected_program = selection["program_key"]
            selected_target = selection["program_index"] == target_index
            target_posterior = float(posterior[target_index])
            ordered = sorted(
                range(len(hypotheses)),
                key=lambda position: (-float(posterior[position]), hypotheses[position].key),
            )
            target_rank = ordered.index(target_index) + 1
            finite_programs = selection["finite_programs"]
            maximum_posterior = selection["maximum_posterior"]
            for support, graphs, graph_index in zip(
                record["oracle_grounding"]["support"], graphs_by_trace,
                selection["graph_indices"], strict=True,
            ):
                scene = scene_lookup[support["id"]]
                graph = graphs[graph_index]
                rows = [
                    {**row, "marginal_map_log_score": graph["log_score"]}
                    for row in graph["prediction_rows"]
                ]
                validate_scene_prediction(scene, rows)
                selected_support[support["id"]] = {
                    "scene_id": scene["id"], "episode_id": scene["episode_id"],
                    "split": scene["split"], "role": "support", "rows": rows,
                    "epistemic_state": predicted_epistemic_rows(scene, rows),
                }
        episode_diagnostics.append({
            "episode_id": record["id"], "split": record["split"], "fallback": fallback,
            "selected_program_key": selected_program, "target_program_selected": selected_target,
            "target_program_posterior": target_posterior, "target_program_rank": target_rank,
            "finite_programs": finite_programs, "maximum_program_posterior": maximum_posterior,
            "unique_graph_states": unique_states,
        })
        if record_index % 5 == 0 or record_index == len(records):
            print(f"v28 marginal MAP: solved {record_index}/{len(records)} episodes", file=sys.stderr, flush=True)
    predictions = [
        selected_support[scene["id"]] if scene["role"] == "support" else v27_lookup[scene["id"]]
        for scene in scenes
    ]
    grounding = grounding_summary(scenes, predictions)
    prediction_lookup = {row["scene_id"]: row for row in predictions}
    evaluation_records = [row for row in records if row["split"] == "grounding_evaluation"]
    integration = {}
    for condition in lock["integration_conditions"]:
        support_mode, query_mode = condition_modes(condition)
        integration[condition] = integration_condition(
            evaluation_records, support_mode, query_mode, prediction_lookup,
            v22_config, original_lock["config_payload"],
        )
    gates = lock["gates"]
    evaluation = grounding["by_split"]["grounding_evaluation"]
    oracle = integration["oracle_support_oracle_query"]
    support = integration["frozen_support_oracle_query"]
    query = integration["oracle_support_frozen_query"]
    primary = integration["frozen_support_frozen_query"]
    checks = {
        "oracle_oracle_exact": oracle["transition_set_exact_match"] >= gates["minimumOracleOracleExact"],
        "evaluation_support_exact_graph": evaluation["exact_support_graph"] >= gates["minimumEvaluationSupportExactGraph"],
        "frozen_support_oracle_query_exact": support["transition_set_exact_match"] >= gates["minimumFrozenSupportOracleQueryExact"],
        "oracle_support_frozen_query_exact": query["transition_set_exact_match"] >= gates["minimumOracleSupportFrozenQueryExact"],
        "frozen_frozen_exact": primary["transition_set_exact_match"] >= gates["minimumFrozenFrozenExact"],
        "frozen_support_target_retention": support["target_retention_rate"] >= gates["minimumFrozenSupportTargetRetention"],
        "frozen_support_empty_version_space": support["empty_version_space_rate"] <= gates["maximumFrozenSupportEmptyVersionSpace"],
    }
    passed = all(checks.values())
    reference = lock["source_v27_reference"]
    if passed:
        decision = "authorize_query_exact_graph_repair_no_fresh_benchmark_yet"
    elif support["transition_set_exact_match"] > reference["frozen_support_oracle_query_exact"]:
        decision = "marginal_program_map_improves_support_continue_query_repair_no_lora"
    else:
        decision = "marginal_program_map_insufficient_revisit_support_identifiability_no_lora"

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions
    ))
    diagnostics_path = output_dir / "episode-marginal-diagnostics.jsonl"
    diagnostics_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in episode_diagnostics
    ))
    evaluation_episode_rows = [row for row in episode_diagnostics if row["split"] == "grounding_evaluation"]
    result = {
        "schema_version": 28, "experiment": lock["experiment"],
        "protocol_lock_sha256": file_sha256(lock_path), "evaluation_number": 1,
        "grounding": grounding, "integration": integration,
        "checks": checks, "passed": passed, "decision": decision,
        "marginal_search": {
            "support_scenes": len(support_scenes),
            "graph_rows": total_graph_rows, "unique_graph_states": total_unique_states,
            "compatibility_deduplication_rate": 1.0 - total_unique_states / total_graph_rows,
            "episode_fallback_rate": mean([row["fallback"] for row in episode_diagnostics]),
            "evaluation_target_program_selection_rate": mean([
                row["target_program_selected"] for row in evaluation_episode_rows
            ]),
            "evaluation_median_target_program_rank": float(np.median([
                row["target_program_rank"] for row in evaluation_episode_rows
                if row["target_program_rank"] is not None
            ])),
            "evaluation_mean_target_program_posterior": mean([
                row["target_program_posterior"] for row in evaluation_episode_rows
            ]),
            "runtime_seconds": time.perf_counter() - started,
        },
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "episode_diagnostics": str(diagnostics_path.relative_to(PROJECT_ROOT)),
        "episode_diagnostics_sha256": file_sha256(diagnostics_path),
        "lora_authorized": False, "fresh_benchmark_constructed": False,
        "data_access": {
            "new_model_forward_passes": 0, "marginal_map_evaluations": 1,
            "head_fits": 0, "threshold_fits": 0, "hyperparameter_selections": 0,
            "adapter_training_runs": 0, "fresh_benchmark_records_read": 0,
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "result_sha256": file_sha256(result_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
