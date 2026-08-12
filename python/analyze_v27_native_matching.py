"""Zero-forward V27 diagnostic for native decoder matching scores."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import numpy as np

from audit_v22r2_grounding import read_jsonl_directory
from audit_v24_cross_encoder import read_pairs
from evaluate_v22r2_relational_grounding import load_npz, mean
from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from v23_probabilistic_relational import k_best_assignments


TOKEN_TRUTH = {"A": [True], "B": [False], "C": [False, True]}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def graph_key(rows):
    return canonical_json(sorted(rows, key=lambda row: row["atom"]))


def assignment_metrics(scene, assignment, logits_by_edge):
    evidence_ids = [row["id"] for row in scene["agent_input"]["evidence"]]
    candidate_ids = [row["id"] for row in scene["agent_input"]["atom_candidates"]]
    target_rows = scene["target"]["atom_groundings"]
    target_by_evidence = {row["evidence_id"]: row for row in target_rows}
    atom_by_candidate = {row["candidate_id"]: row["atom"] for row in target_rows}
    gold_edges = {
        (row["evidence_id"], row["candidate_id"]) for row in target_rows
    }
    predicted_edges = {
        (evidence_ids[index], candidate_ids[position])
        for index, position in enumerate(assignment)
    }
    oracle_truth_rows = []
    native_truth_rows = []
    for evidence_index, candidate_position in enumerate(assignment):
        evidence_id = evidence_ids[evidence_index]
        candidate_id = candidate_ids[candidate_position]
        target = target_by_evidence[evidence_id]
        oracle_truth_rows.append({
            "atom": atom_by_candidate[candidate_id],
            "allowed_values": target["allowed_values"],
        })
        logits = logits_by_edge[(scene["id"], evidence_id, candidate_id)]
        token = min(("A", "B", "C"), key=lambda value: (-logits[value], value))
        native_truth_rows.append({
            "atom": atom_by_candidate[candidate_id],
            "allowed_values": TOKEN_TRUTH[token],
        })
    target_key = graph_key([
        {"atom": row["atom"], "allowed_values": row["allowed_values"]}
        for row in target_rows
    ])
    return {
        "edge_accuracy": len(gold_edges & predicted_edges) / len(gold_edges),
        "exact_assignment": gold_edges == predicted_edges,
        "oracle_truth_exact_graph": graph_key(oracle_truth_rows) == target_key,
        "native_truth_exact_graph": graph_key(native_truth_rows) == target_key,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v27-support-map-lock.json")
    parser.add_argument("--result", default="outputs/v27-support-map/evaluation/result.json")
    parser.add_argument("--output", default="outputs/v27-support-map/native-matching-diagnostic.json")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.result).resolve()
    lock = json.loads(lock_path.read_text())
    result = json.loads(result_path.read_text())
    if result["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V27 result and lock differ")
    config = lock["config_payload"]
    v26_lock = json.loads((PROJECT_ROOT / lock["source"]["v26_lock"]).read_text())
    v25_lock = json.loads((PROJECT_ROOT / v26_lock["source"]["v25_lock"]).read_text())
    v24_lock = json.loads((PROJECT_ROOT / v25_lock["source"]["v24_lock"]).read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = [
        row for row in read_jsonl_directory(
            PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes"
        ) if row["role"] == "support"
    ]
    proposals = [
        row for row in read_pairs(PROJECT_ROOT / config["sourceV24ProposalCorpus"])
        if row["role"] == "support"
    ]
    proposals_by_scene = defaultdict(list)
    for row in proposals:
        proposals_by_scene[row["scene_id"]].append(row)
    feature_metadata = json.loads((PROJECT_ROOT / config["sourceV24Features"]).read_text())
    features = load_npz(PROJECT_ROOT / feature_metadata["feature_artifact"])
    feature_lookup = {
        str(identifier): features["pair_features"][index]
        for index, identifier in enumerate(features["pair_ids"].tolist())
    }
    heads = load_npz(PROJECT_ROOT / config["sourceV24Heads"])
    match_coef = heads["match_coef"][0]
    match_intercept = float(heads["match_intercept"][0])
    score_metadata = json.loads(
        (PROJECT_ROOT / "outputs/v27-support-map/edge-scores/metadata.json").read_text()
    )
    score_rows = [
        *[row for row in jsonl(PROJECT_ROOT / config["sourceV26Scores"]) if row["role"] == "support"],
        *jsonl(PROJECT_ROOT / score_metadata["score_artifact"]),
    ]
    logits_by_edge = {
        (row["scene_id"], row["evidence_id"], row["candidate_id"]): row["fp32_direct_logits"]
        for row in score_rows
    }
    methods = ("v24_layer8_head", "native_resolved_log_odds", "unweighted_sum")
    diagnostics = []
    for scene in scenes:
        evidence_ids = [row["id"] for row in scene["agent_input"]["evidence"]]
        candidate_ids = [row["id"] for row in scene["agent_input"]["atom_candidates"]]
        evidence_index = {value: index for index, value in enumerate(evidence_ids)}
        candidate_index = {value: index for index, value in enumerate(candidate_ids)}
        matrices = {
            method: np.full((len(evidence_ids), len(candidate_ids)), -np.inf, dtype=np.float64)
            for method in methods
        }
        for pair in proposals_by_scene[scene["id"]]:
            key = (scene["id"], pair["evidence_id"], pair["candidate_id"])
            logits = logits_by_edge[key]
            v24_score = float(feature_lookup[pair["id"]] @ match_coef + match_intercept)
            native_score = logaddexp(logits["A"], logits["B"]) - logits["C"]
            row = evidence_index[pair["evidence_id"]]
            column = candidate_index[pair["candidate_id"]]
            matrices["v24_layer8_head"][row, column] = v24_score
            matrices["native_resolved_log_odds"][row, column] = native_score
            matrices["unweighted_sum"][row, column] = v24_score + native_score
        for method, matrix in matrices.items():
            assignments = k_best_assignments(matrix, 1)
            if not assignments:
                raise RuntimeError(f"No perfect matching for {scene['id']} under {method}")
            metrics = assignment_metrics(scene, assignments[0][1], logits_by_edge)
            diagnostics.append({
                "scene_id": scene["id"], "episode_id": scene["episode_id"],
                "split": scene["split"], "method": method, **metrics,
            })
    summaries = {}
    for method in methods:
        summaries[method] = {}
        for split in sorted({row["split"] for row in diagnostics}):
            rows = [
                row for row in diagnostics
                if row["method"] == method and row["split"] == split
            ]
            summaries[method][split] = {
                metric: mean([row[metric] for row in rows])
                for metric in (
                    "edge_accuracy", "exact_assignment", "oracle_truth_exact_graph",
                    "native_truth_exact_graph",
                )
            }
    evaluation = {method: summaries[method]["grounding_evaluation"] for method in methods}
    best_method = max(
        methods,
        key=lambda method: (
            evaluation[method]["exact_assignment"],
            evaluation[method]["edge_accuracy"], -methods.index(method),
        ),
    )
    output = {
        "schema_version": 27,
        "experiment": "v27_post_result_native_matching_diagnostic",
        "status": "exploratory_exposed_data_zero_forward",
        "source_result_sha256": file_sha256(result_path),
        "new_model_forward_passes": 0,
        "head_fits": 0,
        "threshold_fits": 0,
        "methods": {
            "v24_layer8_head": "locked V24 binary match log odds",
            "native_resolved_log_odds": "logsumexp(A,B)-C from existing V26/V27 full-depth scores",
            "unweighted_sum": "sum of the preceding two uncalibrated scores",
        },
        "summary_by_method_and_split": summaries,
        "exploratory_best_evaluation_assignment_method": best_method,
        "interpretation": (
            "Preregister a zero-fit matching replacement only if native resolved-vs-unresolved "
            "scoring materially improves evaluation support assignment; otherwise the residual "
            "is program/graph ambiguity rather than the V24 matching head."
        ),
    }
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
