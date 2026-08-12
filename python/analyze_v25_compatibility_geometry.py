"""Zero-forward, zero-fit geometry diagnostic for the V25 truth representation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from audit_v25_truth_hypotheses import read_rows
from evaluate_v22r2_relational_grounding import load_npz
from v22r2_grounding import PROJECT_ROOT


ASSESSMENTS = ("entailed", "contradicted", "unresolved")


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v25-truth-hypotheses-lock.json")
    parser.add_argument("--result", default="outputs/v25-truth-hypotheses/evaluation/result.json")
    parser.add_argument("--output", default="outputs/v25-truth-hypotheses/compatibility-geometry.json")
    args = parser.parse_args()
    lock = json.loads((PROJECT_ROOT / args.lock).read_text())
    result = json.loads((PROJECT_ROOT / args.result).read_text())
    rows = read_rows(PROJECT_ROOT / lock["source"]["corpus"])
    metadata = json.loads((PROJECT_ROOT / "outputs/v25-truth-hypotheses/features/metadata.json").read_text())
    arrays = load_npz(PROJECT_ROOT / metadata["feature_artifact"])
    head = load_npz(PROJECT_ROOT / result["truth_compatibility_head"])
    features = {
        str(identifier): arrays["truth_features"][index]
        for index, identifier in enumerate(arrays["row_ids"].tolist())
    }

    def score(row):
        return float(sigmoid(features[row["id"]] @ head["coef"][0] + head["intercept"][0]))

    fixed = defaultdict(list)
    for row in rows:
        if "v24_fixed_assignment" in row["selection_sources"]:
            fixed[(row["scene_id"], row["evidence_id"])].append(row)
    by_split = {}
    for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation"):
        triples = [values for values in fixed.values() if values[0]["split"] == split]
        predictions = []
        for triple in triples:
            values = {row["assessment_id"]: score(row) for row in triple}
            selected = max(values, key=values.get)
            predictions.append({
                "gold": triple[0]["target"]["truth_label"],
                "selected": selected,
                "scores": values,
            })
        by_split[split] = {
            "groups": len(predictions),
            "selected_assessment_counts": dict(sorted(Counter(
                row["selected"] for row in predictions
            ).items())),
            "confusion": {
                f"{gold}->{selected}": count for (gold, selected), count in sorted(
                    Counter((row["gold"], row["selected"]) for row in predictions).items()
                )
            },
            "mean_scores_by_gold": {
                gold: {
                    assessment: float(np.mean([
                        row["scores"][assessment] for row in predictions if row["gold"] == gold
                    ]))
                    for assessment in ASSESSMENTS
                }
                for gold in ("false", "true", "unknown")
                if any(row["gold"] == gold for row in predictions)
            },
        }

    fit = [row for row in rows if row["target"]["use_for_fit"]]
    centroids = {}
    counts = {}
    for assessment in ASSESSMENTS:
        for compatible in (False, True):
            selected = np.stack([
                features[row["id"]] for row in fit
                if row["assessment_id"] == assessment
                and row["target"]["compatible"] == compatible
            ])
            centroids[(assessment, compatible)] = selected.mean(axis=0)
            counts[f"{assessment}|{str(compatible).lower()}"] = len(selected)
    within = {
        assessment: float(np.linalg.norm(
            centroids[(assessment, True)] - centroids[(assessment, False)]
        )) for assessment in ASSESSMENTS
    }
    between = {}
    for index, left in enumerate(ASSESSMENTS):
        for right in ASSESSMENTS[index + 1:]:
            left_center = (centroids[(left, False)] + centroids[(left, True)]) / 2
            right_center = (centroids[(right, False)] + centroids[(right, True)]) / 2
            between[f"{left}|{right}"] = float(np.linalg.norm(left_center - right_center))
    diagnostic = {
        "schema_version": "25-geometry",
        "experiment": "v25_zero_fit_zero_forward_compatibility_geometry",
        "data_access": {
            "new_model_forward_passes": 0,
            "new_linear_fits": 0,
            "hyperparameter_selections": 0,
            "adapter_training_runs": 0,
        },
        "prediction_geometry": by_split,
        "centroid_counts": counts,
        "within_assessment_compatibility_distances": within,
        "between_assessment_identity_distances": between,
        "localization": {
            "assessment_identity_dominates_compatibility": min(between.values()) > max(within.values()),
            "unresolved_never_selected_on_evaluation": (
                by_split["grounding_evaluation"]["selected_assessment_counts"].get("unresolved", 0) == 0
            ),
            "reject_more_layer8_linear_head_variants": True,
            "next_registered_direction": (
                "Use the frozen model's full-depth native decoder and float32 A/B/C label projection "
                "on one candidate-conditioned truth prompt, with no fitted head."
            ),
        },
    }
    output = (PROJECT_ROOT / args.output).resolve()
    output.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
