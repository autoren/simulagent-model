"""Zero-forward, zero-fit truth-stage decomposition for the exposed V24 result."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from audit_v22r2_grounding import read_jsonl_directory
from audit_v24_cross_encoder import read_pairs
from evaluate_v22r2_relational_grounding import load_npz, mean
from v22r2_grounding import PROJECT_ROOT


def grouped(rows: Sequence[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    values = sorted({tuple(row[field] for field in fields) for row in rows}, key=str)
    for values_tuple in values:
        selected = [
            row for row in rows if tuple(row[field] for field in fields) == values_tuple
        ]
        key = "|".join(str(value) for value in values_tuple)
        result[key] = {
            "atoms": len(selected),
            "truth_accuracy": mean([row["truth_correct"] for row in selected]),
            "assignment_accuracy": mean([row["assignment_correct"] for row in selected]),
            "confusion": {
                f"{gold}->{predicted}": count for (gold, predicted), count in sorted(
                    Counter((row["gold_truth"], row["predicted_truth"]) for row in selected).items()
                )
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v24-cross-encoder.json")
    parser.add_argument("--result", default="outputs/v24-cross-encoder/evaluation/result.json")
    parser.add_argument("--output", default="outputs/v24-cross-encoder/truth-error-conditioning.json")
    args = parser.parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text())
    result = json.loads((PROJECT_ROOT / args.result).read_text())
    original_lock = json.loads((PROJECT_ROOT / config["sourceV22r2Lock"]).read_text())
    scenes = read_jsonl_directory(PROJECT_ROOT / original_lock["source"]["dataset"] / "scenes")
    scene_lookup = {row["id"]: row for row in scenes}
    predictions = [
        json.loads(line) for line in (PROJECT_ROOT / result["grounding_predictions"]).read_text().splitlines()
        if line.strip()
    ]
    pair_root = PROJECT_ROOT / config["outputDir"]
    pairs = read_pairs(pair_root)
    positive_pair = {
        (row["scene_id"], row["evidence_id"]): row
        for row in pairs if row["target"]["same_atom"]
    }
    metadata = json.loads((PROJECT_ROOT / "outputs/v24-cross-encoder/features/metadata.json").read_text())
    arrays = load_npz(PROJECT_ROOT / metadata["feature_artifact"])
    heads = load_npz(PROJECT_ROOT / result["heads_artifact"])
    features = {
        str(identifier): arrays["pair_features"][index]
        for index, identifier in enumerate(arrays["pair_ids"].tolist())
    }
    truth_classes = heads["truth_classes"].tolist()

    def oracle_pair_prediction(pair: dict[str, Any]) -> str:
        scores = features[pair["id"]] @ heads["truth_coef"].T + heads["truth_intercept"]
        return str(truth_classes[int(np.argmax(scores))])

    rows = []
    oracle_rows = []
    for prediction in predictions:
        if prediction["split"] != "grounding_evaluation":
            continue
        scene = scene_lookup[prediction["scene_id"]]
        targets = {row["evidence_id"]: row for row in scene["target"]["atom_groundings"]}
        for predicted in prediction["rows"]:
            target = targets[predicted["evidence_id"]]
            row = {
                "role": scene["role"],
                "surface_bank": target["surface_bank"],
                "semantic_operator": target["semantic_operator"],
                "predicate_kind": target["predicate_kind"],
                "relation_orientation": target["relation_orientation"],
                "gold_truth": target["truth_label"],
                "predicted_truth": predicted["truth_label"],
                "truth_correct": predicted["truth_label"] == target["truth_label"],
                "assignment_correct": predicted["candidate_id"] == target["candidate_id"],
            }
            rows.append(row)
            pair = positive_pair.get((scene["id"], predicted["evidence_id"]))
            if pair is not None:
                oracle_prediction = oracle_pair_prediction(pair)
                oracle_rows.append({
                    **row,
                    "predicted_truth": oracle_prediction,
                    "truth_correct": oracle_prediction == target["truth_label"],
                    "assignment_correct": True,
                })

    v22_result = json.loads((PROJECT_ROOT / config["sourceV22r2aResult"]).read_text())
    old_eval = v22_result["grounding"]["by_split"]["grounding_evaluation"]
    old_integration = v22_result["integration"]
    new_eval = result["grounding"]["by_split"]["grounding_evaluation"]
    new_integration = result["integration"]
    diagnostic = {
        "schema_version": "24-truth-diagnostic",
        "experiment": "v24_zero_fit_zero_forward_truth_error_conditioning",
        "data_access": {
            "new_model_forward_passes": 0,
            "new_linear_fits": 0,
            "hyperparameter_selections": 0,
            "adapter_training_runs": 0,
        },
        "evaluation": {
            "atoms": len(rows),
            "truth_accuracy": mean([row["truth_correct"] for row in rows]),
            "truth_accuracy_given_correct_assignment": mean([
                row["truth_correct"] for row in rows if row["assignment_correct"]
            ]),
            "truth_accuracy_given_wrong_assignment": mean([
                row["truth_correct"] for row in rows if not row["assignment_correct"]
            ]),
            "oracle_gold_pair_coverage": len(oracle_rows) / len(rows),
            "oracle_gold_pair_truth_accuracy": mean([
                row["truth_correct"] for row in oracle_rows
            ]),
            "by_truth": grouped(rows, ("gold_truth",)),
            "by_surface": grouped(rows, ("surface_bank",)),
            "by_operator": grouped(rows, ("semantic_operator",)),
            "by_surface_operator_truth": grouped(
                rows, ("surface_bank", "semantic_operator", "gold_truth")
            ),
            "by_role": grouped(rows, ("role",)),
            "by_predicate_kind": grouped(rows, ("predicate_kind",)),
        },
        "change_from_v22r2a": {
            "atom_assignment": {
                "v22r2a": old_eval["atom_assignment_accuracy"],
                "v24": new_eval["atom_assignment_accuracy"],
            },
            "relation_order": {
                "v22r2a": old_eval["relation_argument_order_accuracy"],
                "v24": new_eval["relation_argument_order_accuracy"],
            },
            "truth": {
                "v22r2a": old_eval["truth_status_accuracy"],
                "v24": new_eval["truth_status_accuracy"],
            },
            "exact_scene": {
                "v22r2a": old_eval["exact_scene_graph"],
                "v24": new_eval["exact_scene_graph"],
            },
            "frozen_frozen_exact": {
                "v22r2a": old_integration["frozen_support_frozen_query"]["transition_set_exact_match"],
                "v24": new_integration["frozen_support_frozen_query"]["transition_set_exact_match"],
            },
            "frozen_support_oracle_query_exact": {
                "v22r2a": old_integration["frozen_support_oracle_query"]["transition_set_exact_match"],
                "v24": new_integration["frozen_support_oracle_query"]["transition_set_exact_match"],
            },
        },
        "localization": {
            "assignment_is_not_primary_truth_cause": True,
            "dominant_cell": "eval_c|contrastive_both|true",
            "dominant_cell_mechanism": (
                "The eval_c contrastive template states the opposite first and the gold fact second; "
                "the direct multiclass candidate-span readout predicts false for nearly every true case."
            ),
            "next_registered_direction": (
                "Freeze V24 matching and compare three explicit entailment, contradiction, and "
                "unresolved assessment hypotheses after the evidence/candidate pair."
            ),
        },
    }
    output = (PROJECT_ROOT / args.output).resolve()
    output.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
