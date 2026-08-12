#!/usr/bin/env python3
"""Run the single locked V29 posterior-marginal graph evaluation."""

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
from v22_relational import canonical_json, enumerate_program_hypotheses
from v22r2_grounding import PROJECT_ROOT, predicted_epistemic_rows, validate_scene_prediction
from v27_support_map import enumerate_scene_graphs
from v28_marginal_map import compatibility_matrix_deduplicated
from v29_posterior_graph import posterior_marginal_decode


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v29-posterior-graph-lock.json")
    parser.add_argument("--output-dir", default="outputs/v29-posterior-graph/evaluation")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V29 posterior graph evaluation was already attempted")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["posteriorGraphEvaluations"] != 1 or lock["limits"]["newModelForwardPasses"] != 0:
        raise RuntimeError("V29 lock does not authorize the registered zero-forward evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V29 locked implementation changed: {path}")
    for key in (
        "sourceV28Lock", "sourceV28Result", "sourceV28PostAudit",
        "sourceV28Predictions", "sourceV28Diagnostics",
    ):
        if file_sha256(PROJECT_ROOT / lock["source"][key]) != lock["source"][f"{key}_sha256"]:
            raise RuntimeError(f"V29 locked source changed: {key}")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 29, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_posterior_graph_evaluation",
    }, indent=2, sort_keys=True) + "\n")

    config = lock["config_payload"]
    v28_lock = json.loads((PROJECT_ROOT / config["sourceV28Lock"]).read_text())
    for key, value in v28_lock["source"].items():
        if not key.endswith("_sha256"):
            expected = v28_lock["source"].get(f"{key}_sha256")
            if expected is not None and file_sha256(PROJECT_ROOT / value) != expected:
                raise RuntimeError(f"V29 locked V28 source changed: {key}")
    v27_lock = json.loads((PROJECT_ROOT / v28_lock["source"]["sourceV27Lock"]).read_text())
    for key, value in v27_lock["source"].items():
        if key.endswith("_sha256") or key in {"corpus", "corpus_sha256", "corpus_file_sha256"}:
            continue
        expected = v27_lock["source"].get(f"{key}_sha256")
        if expected is not None and isinstance(value, str) and file_sha256(PROJECT_ROOT / value) != expected:
            raise RuntimeError(f"V29 locked V27 source changed: {key}")
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
    v27_edge_metadata = json.loads(
        (PROJECT_ROOT / v28_lock["source"]["sourceV27EdgeMetadata"]).read_text()
    )
    score_rows = [
        *[row for row in jsonl(PROJECT_ROOT / v27_config["sourceV26Scores"]) if row["role"] == "support"],
        *jsonl(PROJECT_ROOT / v27_edge_metadata["score_artifact"]),
    ]
    truth_logits = {
        (row["scene_id"], row["evidence_id"], row["candidate_id"]): row["fp32_direct_logits"]
        for row in score_rows
    }
    if set(truth_logits) != {
        (row["scene_id"], row["evidence_id"], row["candidate_id"])
        for row in proposal_pairs
    }:
        raise RuntimeError("V29 truth scores do not cover exactly all support proposal edges")

    started = time.perf_counter()
    graph_cache = {}
    target_graph_keys = {}
    support_scenes = [row for row in scenes if row["role"] == "support"]
    for index, scene in enumerate(support_scenes, start=1):
        graph_cache[scene["id"]] = enumerate_scene_graphs(
            scene, proposals_by_scene[scene["id"]], feature_lookup,
            match_coef, match_intercept, truth_logits, v27_config,
        )
        target_graph_keys[scene["id"]] = canonical_json(sorted([
            {"atom": row["atom"], "allowed_values": row["allowed_values"]}
            for row in scene["target"]["atom_groundings"]
        ], key=lambda row: row["atom"]))
        if index % 10 == 0 or index == len(support_scenes):
            print(f"v29 posterior graph: enumerated {index}/{len(support_scenes)} support scenes", file=sys.stderr, flush=True)

    v28_predictions = jsonl(PROJECT_ROOT / config["sourceV28Predictions"])
    v28_lookup = {row["scene_id"]: row for row in v28_predictions}
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
            compatibility, _ = compatibility_matrix_deduplicated(
                graphs, hypotheses, support, observed[support["id"]], v22_config,
                v27_config["jointMap"]["maximumUnknownAtomsPerGraph"],
            )
            graphs_by_trace.append(graphs)
            compatibility_by_trace.append(compatibility)
        selection = posterior_marginal_decode(
            hypotheses, graphs_by_trace, compatibility_by_trace
        )
        fallback = selection is None
        if fallback:
            for support in record["oracle_grounding"]["support"]:
                selected_support[support["id"]] = v28_lookup[support["id"]]
            selected_target = False
            exact_graphs = 0
            mean_selected_graph_posterior = 0.0
        else:
            selected_target = selection["program_key"] == record["target"]["program_key"]
            exact_graphs = 0
            for support, graphs, graph_index, graph_posterior in zip(
                record["oracle_grounding"]["support"], graphs_by_trace,
                selection["graph_indices"], selection["graph_posteriors"], strict=True,
            ):
                scene = scene_lookup[support["id"]]
                graph = graphs[graph_index]
                exact_graphs += graph["graph_key"] == target_graph_keys[scene["id"]]
                rows = [
                    {**row, "posterior_graph_probability": graph_posterior}
                    for row in graph["prediction_rows"]
                ]
                validate_scene_prediction(scene, rows)
                selected_support[support["id"]] = {
                    "scene_id": scene["id"], "episode_id": scene["episode_id"],
                    "split": scene["split"], "role": "support", "rows": rows,
                    "epistemic_state": predicted_epistemic_rows(scene, rows),
                }
            mean_selected_graph_posterior = mean(selection["graph_posteriors"])
        episode_diagnostics.append({
            "episode_id": record["id"], "split": record["split"], "fallback": fallback,
            "target_program_top1": selected_target,
            "selected_exact_support_graphs": exact_graphs,
            "support_scenes": len(record["oracle_grounding"]["support"]),
            "mean_selected_graph_posterior": mean_selected_graph_posterior,
        })
        if record_index % 5 == 0 or record_index == len(records):
            print(f"v29 posterior graph: solved {record_index}/{len(records)} episodes", file=sys.stderr, flush=True)

    predictions = [
        selected_support[scene["id"]] if scene["role"] == "support" else v28_lookup[scene["id"]]
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
    reference = lock["source_v28_reference"]
    if passed:
        decision = "authorize_query_exact_graph_repair_no_fresh_benchmark_yet"
    elif evaluation["exact_support_graph"] > reference["evaluation_support_exact_graph"]:
        decision = "posterior_graph_decoding_improves_exact_support_continue_no_lora"
    else:
        decision = "posterior_graph_decoding_insufficient_revisit_language_scores_no_lora"
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions
    ))
    diagnostics_path = output_dir / "episode-posterior-graph-diagnostics.jsonl"
    diagnostics_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in episode_diagnostics
    ))
    result = {
        "schema_version": 29, "experiment": lock["experiment"],
        "protocol_lock_sha256": file_sha256(lock_path), "evaluation_number": 1,
        "grounding": grounding, "integration": integration,
        "checks": checks, "passed": passed, "decision": decision,
        "posterior_graph": {
            "support_scenes": len(support_scenes),
            "episode_fallback_rate": mean([row["fallback"] for row in episode_diagnostics]),
            "evaluation_target_program_top1_rate": mean([
                row["target_program_top1"] for row in episode_diagnostics
                if row["split"] == "grounding_evaluation"
            ]),
            "evaluation_mean_selected_graph_posterior": mean([
                row["mean_selected_graph_posterior"] for row in episode_diagnostics
                if row["split"] == "grounding_evaluation"
            ]),
            "runtime_seconds": time.perf_counter() - started,
        },
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "episode_diagnostics": str(diagnostics_path.relative_to(PROJECT_ROOT)),
        "episode_diagnostics_sha256": file_sha256(diagnostics_path),
        "lora_authorized": False, "fresh_benchmark_constructed": False,
        "data_access": {
            "new_model_forward_passes": 0, "posterior_graph_evaluations": 1,
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
