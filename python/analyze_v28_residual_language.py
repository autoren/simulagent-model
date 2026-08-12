"""Zero-forward decomposition of V28's remaining evaluation support graph errors."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from audit_v22r2_grounding import read_jsonl_directory
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT


TOKEN_BY_TRUTH = {"true": "A", "false": "B", "unknown": "C"}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="outputs/v28-marginal-map/evaluation/result.json")
    parser.add_argument("--audit", default="outputs/v28-marginal-map/post-result-audit.json")
    parser.add_argument("--output", default="outputs/v28-marginal-map/residual-language-decomposition.json")
    args = parser.parse_args()
    result_path = (PROJECT_ROOT / args.result).resolve()
    result = json.loads(result_path.read_text())
    audit = json.loads((PROJECT_ROOT / args.audit).read_text())
    if not audit["passed"] or audit["decision"] != "accept_v28_exposed_development_result":
        raise RuntimeError("V28 result is not accepted")
    v28_lock = json.loads((PROJECT_ROOT / "configs/v28-marginal-map-lock.json").read_text())
    v27_lock = json.loads((PROJECT_ROOT / v28_lock["source"]["sourceV27Lock"]).read_text())
    v26_lock = json.loads((PROJECT_ROOT / v27_lock["source"]["v26_lock"]).read_text())
    v25_lock = json.loads((PROJECT_ROOT / v26_lock["source"]["v25_lock"]).read_text())
    v24_lock = json.loads((PROJECT_ROOT / v25_lock["source"]["v24_lock"]).read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = [
        row for row in read_jsonl_directory(
            PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes"
        )
        if row["split"] == "grounding_evaluation" and row["role"] == "support"
    ]
    predictions = {
        row["scene_id"]: row for row in jsonl(PROJECT_ROOT / result["grounding_predictions"])
    }
    v27_config = v27_lock["config_payload"]
    edge_metadata = json.loads(
        (PROJECT_ROOT / v28_lock["source"]["sourceV27EdgeMetadata"]).read_text()
    )
    scores = [
        *[row for row in jsonl(PROJECT_ROOT / v27_config["sourceV26Scores"]) if row["role"] == "support"],
        *jsonl(PROJECT_ROOT / edge_metadata["score_artifact"]),
    ]
    score_lookup = {
        (row["scene_id"], row["evidence_id"], row["candidate_id"]): row["fp32_direct_logits"]
        for row in scores
    }
    scene_rows = []
    atom_errors = []
    raw_native_correct = 0
    selected_truth_errors = 0
    for scene in scenes:
        prediction = predictions[scene["id"]]
        target = scene["target"]["atom_groundings"]
        target_by_evidence = {row["evidence_id"]: row for row in target}
        target_by_candidate = {row["candidate_id"]: row for row in target}
        assigned = {row["evidence_id"]: row for row in prediction["rows"]}
        exact_assignment = all(
            assigned[evidence_id]["candidate_id"] == row["candidate_id"]
            for evidence_id, row in target_by_evidence.items()
        )
        error_causes = set()
        truth_exact = True
        for evidence_id, row in assigned.items():
            candidate_target = target_by_candidate[row["candidate_id"]]
            evidence_target = target_by_evidence[evidence_id]
            if row["truth_label"] == evidence_target["truth_label"]:
                continue
            truth_exact = False
            selected_truth_errors += 1
            alignment_correct = row["candidate_id"] == evidence_target["candidate_id"]
            cause = "truth_on_correct_edge" if alignment_correct else "truth_after_wrong_alignment"
            error_causes.add(cause)
            logits = score_lookup[(scene["id"], evidence_id, row["candidate_id"])]
            raw_token = min(("A", "B", "C"), key=lambda token: (-logits[token], token))
            gold_token = TOKEN_BY_TRUTH[evidence_target["truth_label"]]
            raw_correct = raw_token == gold_token
            raw_native_correct += raw_correct
            atom_errors.append({
                "scene_id": scene["id"], "episode_id": scene["episode_id"],
                "evidence_id": evidence_id, "candidate_id": row["candidate_id"],
                "cause": cause, "selected_truth": row["truth_label"],
                "evidence_gold_truth": evidence_target["truth_label"],
                "candidate_gold_truth": candidate_target["truth_label"],
                "raw_native_truth": {"A": "true", "B": "false", "C": "unknown"}[raw_token],
                "raw_native_was_correct": raw_correct,
                "assigned_evidence_operator": evidence_target["semantic_operator"],
                "assigned_evidence_surface_bank": evidence_target["surface_bank"],
                "candidate_gold_operator": candidate_target["semantic_operator"],
                "candidate_gold_surface_bank": candidate_target["surface_bank"],
            })
        if not exact_assignment:
            error_causes.add("assignment")
        exact_graph = exact_assignment and truth_exact
        scene_rows.append({
            "scene_id": scene["id"], "episode_id": scene["episode_id"],
            "exact_assignment": exact_assignment, "exact_graph": exact_graph,
            "error_causes": sorted(error_causes),
        })
    wrong_scenes = [row for row in scene_rows if not row["exact_graph"]]
    output = {
        "schema_version": 28,
        "experiment": "v28_post_result_residual_language_decomposition",
        "status": "exploratory_exposed_data_zero_forward",
        "source_result_sha256": file_sha256(result_path),
        "new_model_forward_passes": 0, "fits": 0, "thresholds": 0,
        "evaluation_support": {
            "scenes": len(scene_rows),
            "exact_graphs": sum(row["exact_graph"] for row in scene_rows),
            "exact_assignments": sum(row["exact_assignment"] for row in scene_rows),
            "wrong_graphs": len(wrong_scenes),
            "wrong_graphs_with_exact_assignment": sum(row["exact_assignment"] for row in wrong_scenes),
            "wrong_graphs_with_assignment_error": sum(not row["exact_assignment"] for row in wrong_scenes),
            "selected_incorrect_atom_values": selected_truth_errors,
            "incorrect_atom_values_where_raw_native_was_correct": raw_native_correct,
        },
        "incorrect_atom_value_counts": {
            "by_cause": dict(sorted(Counter(row["cause"] for row in atom_errors).items())),
            "by_assigned_evidence_operator": dict(sorted(Counter(row["assigned_evidence_operator"] for row in atom_errors).items())),
            "by_assigned_evidence_surface_bank": dict(sorted(Counter(row["assigned_evidence_surface_bank"] for row in atom_errors).items())),
            "by_evidence_gold_truth": dict(sorted(Counter(row["evidence_gold_truth"] for row in atom_errors).items())),
            "by_candidate_gold_truth": dict(sorted(Counter(row["candidate_gold_truth"] for row in atom_errors).items())),
        },
        "wrong_scene_rows": wrong_scenes,
        "incorrect_atom_rows": atom_errors,
        "interpretation_rule": (
            "If most wrong graphs retain exact assignment and raw native truth is already wrong, "
            "the next experiment must change language representations/scores on a fresh surface "
            "development protocol; another posterior decision rule is not justified."
        ),
    }
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
