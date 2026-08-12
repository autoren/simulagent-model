"""Zero-forward residual and oracle-component decomposition for V26."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from audit_v22r2_grounding import read_jsonl_directory
from evaluate_v22r2_relational_grounding import (
    grounding_summary,
    integration_condition,
    mean,
)
from v22r2_grounding import PROJECT_ROOT, predicted_epistemic_rows


def prediction_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rebuild_prediction(scene, rows):
    return {
        "scene_id": scene["id"], "episode_id": scene["episode_id"],
        "split": scene["split"], "role": scene["role"], "rows": rows,
        "epistemic_state": predicted_epistemic_rows(scene, rows),
    }


def oracle_truth_predictions(scenes, predictions):
    source = {row["scene_id"]: row for row in predictions}
    result = []
    for scene in scenes:
        target = {row["evidence_id"]: row for row in scene["target"]["atom_groundings"]}
        rows = [
            {**row, "truth_label": target[row["evidence_id"]]["truth_label"]}
            for row in source[scene["id"]]["rows"]
        ]
        result.append(rebuild_prediction(scene, rows))
    return result


def optimistic_oracle_assignment_predictions(scenes, predictions):
    """Repair candidate IDs; preserve decoder truth only where it scored the gold candidate."""
    source = {row["scene_id"]: row for row in predictions}
    result = []
    for scene in scenes:
        target = {row["evidence_id"]: row for row in scene["target"]["atom_groundings"]}
        rows = []
        for row in source[scene["id"]]["rows"]:
            gold = target[row["evidence_id"]]
            rows.append({
                **row,
                "candidate_id": gold["candidate_id"],
                "truth_label": (
                    row["truth_label"] if row["candidate_id"] == gold["candidate_id"]
                    else gold["truth_label"]
                ),
            })
        result.append(rebuild_prediction(scene, rows))
    return result


def score_margin(row):
    values = sorted(row["fp32_direct_logits"].values(), reverse=True)
    return values[0] - values[1]


def quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "minimum": min(values),
        "q25": float(np.quantile(values, .25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, .75)),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v26-native-truth-decoder-lock.json")
    parser.add_argument("--result", default="outputs/v26-native-truth-decoder/evaluation/result.json")
    parser.add_argument("--output", default="outputs/v26-native-truth-decoder/residual-decomposition.json")
    args = parser.parse_args()
    lock = json.loads((PROJECT_ROOT / args.lock).read_text())
    result = json.loads((PROJECT_ROOT / args.result).read_text())
    v25_lock = json.loads((PROJECT_ROOT / lock["source"]["v25_lock"]).read_text())
    v24_lock = json.loads((PROJECT_ROOT / v25_lock["source"]["v24_lock"]).read_text())
    original_lock = json.loads((PROJECT_ROOT / v24_lock["source"]["v22r2_lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    records = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "records")
    evaluation_records = [row for row in records if row["split"] == "grounding_evaluation"]
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())
    predictions = prediction_rows(PROJECT_ROOT / result["grounding_predictions"])
    score_rows = prediction_rows(PROJECT_ROOT / result["native_decoder_scores"])

    oracle_truth = oracle_truth_predictions(scenes, predictions)
    optimistic_assignment = optimistic_oracle_assignment_predictions(scenes, predictions)
    variants = {
        "v26": predictions,
        "fixed_assignment_oracle_truth": oracle_truth,
        "optimistic_oracle_assignment_v26_truth": optimistic_assignment,
    }
    summaries = {name: grounding_summary(scenes, rows) for name, rows in variants.items()}
    support_integration = {}
    for name, rows in variants.items():
        lookup = {row["scene_id"]: row for row in rows}
        support_integration[name] = integration_condition(
            evaluation_records, "frozen", "oracle", lookup,
            v22_config, original_lock["config_payload"],
        )

    predicted = {row["scene_id"]: row for row in predictions}
    score_lookup = {(row["scene_id"], row["evidence_id"]): row for row in score_rows}
    atom_rows = []
    scene_rows = []
    for scene in scenes:
        if scene["split"] != "grounding_evaluation":
            continue
        target = {row["evidence_id"]: row for row in scene["target"]["atom_groundings"]}
        row_correct = []
        assignment_correct = []
        truth_correct = []
        for row in predicted[scene["id"]]["rows"]:
            gold = target[row["evidence_id"]]
            assignment_ok = row["candidate_id"] == gold["candidate_id"]
            truth_ok = row["truth_label"] == gold["truth_label"]
            value_ok = assignment_ok and truth_ok
            scores = score_lookup[(scene["id"], row["evidence_id"])]
            atom_rows.append({
                "role": scene["role"], "scene_id": scene["id"],
                "episode_id": scene["episode_id"],
                "assignment_correct": assignment_ok, "truth_correct": truth_ok,
                "atom_value_correct": value_ok, "margin": score_margin(scores),
                "surface_bank": gold["surface_bank"],
                "semantic_operator": gold["semantic_operator"],
                "truth_label": gold["truth_label"],
            })
            assignment_correct.append(assignment_ok)
            truth_correct.append(truth_ok)
            row_correct.append(value_ok)
        scene_rows.append({
            "scene_id": scene["id"], "episode_id": scene["episode_id"],
            "role": scene["role"], "atoms": len(row_correct),
            "assignment_exact": all(assignment_correct),
            "truth_exact": all(truth_correct), "value_exact": all(row_correct),
            "atom_errors": sum(not value for value in row_correct),
        })

    by_role = {}
    for role in ("support", "query"):
        atoms = [row for row in atom_rows if row["role"] == role]
        selected_scenes = [row for row in scene_rows if row["role"] == role]
        by_role[role] = {
            "atoms": len(atoms), "scenes": len(selected_scenes),
            "assignment_accuracy": mean([row["assignment_correct"] for row in atoms]),
            "truth_accuracy": mean([row["truth_correct"] for row in atoms]),
            "atom_value_accuracy": mean([row["atom_value_correct"] for row in atoms]),
            "assignment_exact_scene": mean([row["assignment_exact"] for row in selected_scenes]),
            "truth_exact_scene": mean([row["truth_exact"] for row in selected_scenes]),
            "value_exact_scene": mean([row["value_exact"] for row in selected_scenes]),
            "scene_atom_error_counts": {
                str(key): value for key, value in sorted(Counter(
                    row["atom_errors"] for row in selected_scenes
                ).items())
            },
        }
    margin_groups = {
        "correct": quantiles([row["margin"] for row in atom_rows if row["atom_value_correct"]]),
        "incorrect": quantiles([row["margin"] for row in atom_rows if not row["atom_value_correct"]]),
        "support_correct": quantiles([
            row["margin"] for row in atom_rows if row["role"] == "support" and row["atom_value_correct"]
        ]),
        "support_incorrect": quantiles([
            row["margin"] for row in atom_rows if row["role"] == "support" and not row["atom_value_correct"]
        ]),
    }
    evaluation = {
        name: summary["by_split"]["grounding_evaluation"] for name, summary in summaries.items()
    }
    diagnostic = {
        "schema_version": "26-residual",
        "experiment": "v26_zero_fit_zero_forward_residual_decomposition",
        "data_access": {
            "new_model_forward_passes": 0, "new_linear_fits": 0,
            "hyperparameter_selections": 0, "adapter_training_runs": 0,
        },
        "evaluation_grounding_variants": evaluation,
        "frozen_support_oracle_query_variants": support_integration,
        "v26_by_role": by_role,
        "decoder_margin_quantiles": margin_groups,
        "localization": {
            "query_stage_passes_with_oracle_support": (
                result["integration"]["oracle_support_frozen_query"]["transition_set_exact_match"]
                >= lock["gates"]["development"]["minimumOracleSupportFrozenQueryExact"]
            ),
            "native_truth_gate_passes": result["checks"]["evaluation_truth"],
            "support_stage_is_primary_remaining_bottleneck": True,
            "next_registered_direction": (
                "Score all V24 proposal edges only on support scenes with the locked V26 decoder, "
                "then jointly rerank sparse support graphs by V24 match likelihood, V26 truth logits, "
                "one-to-one validity, and consistency with observed transitions. Keep query inference unchanged."
            ),
        },
    }
    output = (PROJECT_ROOT / args.output).resolve()
    output.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
