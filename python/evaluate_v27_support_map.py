#!/usr/bin/env python3
"""Run the single locked V27 outcome-constrained support MAP evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from audit_v22r2_grounding import read_jsonl_directory
from audit_v24_cross_encoder import read_pairs
from evaluate_v22r2_relational_grounding import (
    condition_modes, grounding_summary, integration_condition, load_npz, mean,
)
from v10_protocol import file_sha256
from v22_relational import canonical_json, enumerate_program_hypotheses
from v22r2_grounding import PROJECT_ROOT, predicted_epistemic_rows, validate_scene_prediction
from v27_support_map import compatibility_matrix, enumerate_scene_graphs, select_episode_map


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v27-support-map-lock.json")
    parser.add_argument("--scores", default="outputs/v27-support-map/edge-scores")
    parser.add_argument("--output-dir", default="outputs/v27-support-map/evaluation")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    score_root = (PROJECT_ROOT / args.scores).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "map-evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V27 MAP evaluation was already attempted")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["jointMapEvaluations"] != 1 or lock["limits"]["headFits"] != 0:
        raise RuntimeError("V27 lock does not authorize the registered MAP evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V27 locked implementation changed: {path}")
    metadata_path = score_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V27 edge scores do not share the MAP lock")
    locked_sources = {
        "v26_lock": lock["source"]["v26_lock_sha256"],
        "v26_result": lock["source"]["v26_result_sha256"],
        "v26_post_audit": lock["source"]["v26_post_audit_sha256"],
        "v26_residual": lock["source"]["v26_residual_sha256"],
        "v26_scores": lock["source"]["v26_scores_sha256"],
        "v26_predictions": lock["source"]["v26_predictions_sha256"],
        "v24_feature_metadata": lock["source"]["v24_feature_metadata_sha256"],
        "v24_feature_artifact": lock["source"]["v24_feature_artifact_sha256"],
        "v24_heads": lock["source"]["v24_heads_sha256"],
    }
    for name, expected in locked_sources.items():
        path = PROJECT_ROOT / lock["source"][name]
        if file_sha256(path) != expected:
            raise RuntimeError(f"V27 locked upstream artifact changed: {name}")
    proposal_manifest = PROJECT_ROOT / lock["source"]["v24_proposal_corpus"] / "manifest.json"
    if file_sha256(proposal_manifest) != lock["source"]["v24_proposal_manifest_sha256"]:
        raise RuntimeError("V27 locked V24 proposal corpus changed")
    score_path = PROJECT_ROOT / metadata["score_artifact"]
    if file_sha256(score_path) != metadata["score_artifact_sha256"]:
        raise RuntimeError("V27 edge-score artifact changed")
    attempt_path.write_text(json.dumps({
        "schema_version": 27, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_joint_map",
    }, indent=2, sort_keys=True) + "\n")

    config = lock["config_payload"]
    v26_lock = json.loads((PROJECT_ROOT / lock["source"]["v26_lock"]).read_text())
    v25_lock = json.loads((PROJECT_ROOT / v26_lock["source"]["v25_lock"]).read_text())
    v24_lock = json.loads((PROJECT_ROOT / v25_lock["source"]["v24_lock"]).read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    scene_lookup = {row["id"]: row for row in scenes}
    records = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "records")
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())
    proposal_pairs = [
        row for row in read_pairs(PROJECT_ROOT / config["sourceV24ProposalCorpus"])
        if row["role"] == "support"
    ]
    proposals_by_scene = defaultdict(list)
    for row in proposal_pairs:
        proposals_by_scene[row["scene_id"]].append(row)
    feature_metadata = json.loads((PROJECT_ROOT / config["sourceV24Features"]).read_text())
    arrays = load_npz(PROJECT_ROOT / feature_metadata["feature_artifact"])
    feature_lookup = {
        str(identifier): arrays["pair_features"][index]
        for index, identifier in enumerate(arrays["pair_ids"].tolist())
    }
    heads = load_npz(PROJECT_ROOT / config["sourceV24Heads"])
    match_coef = heads["match_coef"][0]
    match_intercept = float(heads["match_intercept"][0])
    all_score_rows = [
        *[row for row in jsonl(PROJECT_ROOT / config["sourceV26Scores"]) if row["role"] == "support"],
        *jsonl(score_path),
    ]
    truth_logits = {
        (row["scene_id"], row["evidence_id"], row["candidate_id"]): row["fp32_direct_logits"]
        for row in all_score_rows
    }
    expected_edges = {
        (row["scene_id"], row["evidence_id"], row["candidate_id"])
        for row in proposal_pairs
    }
    if set(truth_logits) != expected_edges:
        raise RuntimeError("V27 truth scores do not cover exactly all support proposal edges")

    graph_cache = {}
    target_graph_ranks = {}
    started = time.perf_counter()
    support_scenes = [row for row in scenes if row["role"] == "support"]
    for index, scene in enumerate(support_scenes, start=1):
        graphs = enumerate_scene_graphs(
            scene, proposals_by_scene[scene["id"]], feature_lookup,
            match_coef, match_intercept, truth_logits, config,
        )
        graph_cache[scene["id"]] = graphs
        target_key = canonical_json(sorted([
            {"atom": row["atom"], "allowed_values": row["allowed_values"]}
            for row in scene["target"]["atom_groundings"]
        ], key=lambda row: row["atom"]))
        ranks = [position + 1 for position, graph in enumerate(graphs) if graph["graph_key"] == target_key]
        target_graph_ranks[scene["id"]] = ranks[0] if ranks else None
        if index % 10 == 0 or index == len(support_scenes):
            print(f"v27 MAP: enumerated {index}/{len(support_scenes)} support scenes", file=sys.stderr, flush=True)

    v26_predictions = jsonl(PROJECT_ROOT / config["sourceV26Predictions"])
    v26_lookup = {row["scene_id"]: row for row in v26_predictions}
    selected_support = {}
    episode_diagnostics = []
    observed = {
        row["id"]: row["observed_transition_code"]
        for record in records for row in record["agent_input"]["support_traces"]
    }
    for record_index, record in enumerate(records, start=1):
        hypotheses = list(enumerate_program_hypotheses(
            record["agent_input"]["dsl_contract"]["outcome_bits"]
        ))
        graphs_by_trace = []
        compatibility_by_trace = []
        for support in record["oracle_grounding"]["support"]:
            graphs = graph_cache[support["id"]]
            graphs_by_trace.append(graphs)
            compatibility_by_trace.append(compatibility_matrix(
                graphs, hypotheses, support, observed[support["id"]], v22_config,
                config["jointMap"]["maximumUnknownAtomsPerGraph"],
            ))
        selection = select_episode_map(hypotheses, graphs_by_trace, compatibility_by_trace)
        fallback = selection is None
        if fallback:
            for support in record["oracle_grounding"]["support"]:
                selected_support[support["id"]] = v26_lookup[support["id"]]
            selected_program = None
            selected_target = False
        else:
            selected_program = selection["program_key"]
            selected_target = selected_program == record["target"]["program_key"]
            for support, graphs, graph_index in zip(
                record["oracle_grounding"]["support"], graphs_by_trace,
                selection["graph_indices"], strict=True,
            ):
                scene = scene_lookup[support["id"]]
                graph = graphs[graph_index]
                rows = [
                    {**row, "joint_map_log_score": graph["log_score"]}
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
            "feasible_programs": 0 if fallback else selection["feasible_programs"],
            "all_target_graphs_in_branches": all(
                target_graph_ranks[support["id"]] is not None
                for support in record["oracle_grounding"]["support"]
            ),
        })
        if record_index % 5 == 0 or record_index == len(records):
            print(
                f"v27 MAP: solved {record_index}/{len(records)} episodes",
                file=sys.stderr, flush=True,
            )
    predictions = [
        selected_support[scene["id"]] if scene["role"] == "support" else v26_lookup[scene["id"]]
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
    gates = lock["gates"]["development"]
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
    if passed:
        decision = "authorize_query_exact_graph_repair_no_fresh_benchmark_yet"
    elif support["transition_set_exact_match"] > lock["source_v26_reference"]["frozen_support_oracle_query_exact"]:
        decision = "support_map_improves_execution_continue_match_repair_no_lora"
    else:
        decision = "outcome_constrained_support_map_insufficient_no_lora"

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions
    ))
    diagnostics_path = output_dir / "episode-map-diagnostics.jsonl"
    diagnostics_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in episode_diagnostics
    ))
    result = {
        "schema_version": 27, "experiment": lock["experiment"],
        "protocol_lock_sha256": file_sha256(lock_path), "evaluation_number": 1,
        "grounding": grounding, "integration": integration,
        "checks": checks, "passed": passed, "decision": decision,
        "graph_search": {
            "support_scenes": len(support_scenes),
            "mean_graph_branches": float(np.mean([len(rows) for rows in graph_cache.values()])),
            "maximum_graph_branches": max(len(rows) for rows in graph_cache.values()),
            "target_graph_in_branch_rate": mean([rank is not None for rank in target_graph_ranks.values()]),
            "median_target_graph_rank_when_present": float(np.median([
                rank for rank in target_graph_ranks.values() if rank is not None
            ])),
            "episode_fallback_rate": mean([row["fallback"] for row in episode_diagnostics]),
            "selected_target_program_rate": mean([
                row["target_program_selected"] for row in episode_diagnostics
            ]),
            "runtime_seconds": time.perf_counter() - started,
        },
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "episode_diagnostics": str(diagnostics_path.relative_to(PROJECT_ROOT)),
        "episode_diagnostics_sha256": file_sha256(diagnostics_path),
        "lora_authorized": False, "fresh_benchmark_constructed": False,
        "data_access": {
            "new_model_forward_passes": metadata["new_model_forward_passes"],
            "joint_map_evaluations": 1, "head_fits": 0, "threshold_fits": 0,
            "hyperparameter_selections": 0, "adapter_training_runs": 0,
            "fresh_benchmark_records_read": 0,
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
