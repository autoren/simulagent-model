"""Run the single sealed V21 multi-mechanic final evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import beta

from audit_v19_compatibility import read_scenes
from evaluate_v19_frozen_integration import evaluate_condition as evaluate_hard_condition
from evaluate_v19_frozen_integration import load_npz
from evaluate_v20_probabilistic_interface import (
    apply_interface, evaluate_condition as evaluate_probabilistic_condition, score_scenes,
)
from v10_protocol import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def exact_interval(successes: int, trials: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [lower, upper]


def hard_predictions(scored: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for scene in scored:
        result.append({
            **{key: scene[key] for key in (
                "scene_id", "episode_id", "source_item_id", "view", "split", "axis", "item_kind"
            )},
            "groundings": [
                {
                    "determinant_id": value["determinant_id"],
                    "allowed_values": value["hard_allowed_values"],
                    "predicted_temporal_status": value["predicted_temporal_status"],
                    "predicted_current_value": value["hard_current_value"],
                    "selected_evidence_index": value["selected_evidence_index"],
                    "span_correct": value["span_correct"],
                    "temporal_correct": value["temporal_correct"],
                }
                for value in scene["groundings"]
            ],
        })
    return result


def grounding_summary(
    scored: Sequence[dict[str, Any]], probabilistic: Sequence[dict[str, Any]],
    scenes_by_id: dict[str, dict[str, Any]], view: str,
) -> dict[str, Any]:
    probabilistic_by_id = {value["scene_id"]: value for value in probabilistic}
    rows = []
    by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scene in scored:
        if scene["view"] != view:
            continue
        source = scenes_by_id[scene["scene_id"]]
        prob = probabilistic_by_id[scene["scene_id"]]
        for hard, uncertain, target in zip(
            scene["groundings"], prob["groundings"],
            source["target"]["determinant_grounding"], strict=True,
        ):
            target_set = set(target["allowed_values"])
            row = {
                "hard_exact": hard["hard_allowed_values"] == target["allowed_values"],
                "probabilistic_exact": uncertain["allowed_values"] == target["allowed_values"],
                "probabilistic_coverage": target_set <= set(uncertain["allowed_values"]),
                "probabilistic_set_size": len(uncertain["allowed_values"]),
                "span_correct": hard["span_correct"],
                "temporal_correct": hard["temporal_correct"],
                "current": target["current_value"] is not None,
                "polarity_correct": (
                    hard["hard_current_value"] == target["current_value"]
                    if target["current_value"] is not None else None
                ),
            }
            rows.append(row)
            by_surface[source["surface_family"]].append(row)
            by_operator[source["semantic_operator_family"]].append(row)

    def summarize(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
        current = [value for value in values if value["current"]]
        return {
            "determinants": len(values),
            "hard_allowed_value_accuracy": float(np.mean([value["hard_exact"] for value in values])),
            "probabilistic_exact_set_accuracy": float(np.mean([value["probabilistic_exact"] for value in values])),
            "probabilistic_label_coverage": float(np.mean([value["probabilistic_coverage"] for value in values])),
            "probabilistic_mean_set_size": float(np.mean([value["probabilistic_set_size"] for value in values])),
            "span_accuracy": float(np.mean([value["span_correct"] for value in values])),
            "temporal_accuracy": float(np.mean([value["temporal_correct"] for value in values])),
            "current_polarity_accuracy": float(np.mean([value["polarity_correct"] for value in current])),
        }
    return {
        "overall": summarize(rows),
        "by_surface": {key: summarize(value) for key, value in sorted(by_surface.items())},
        "by_semantic_operator": {key: summarize(value) for key, value in sorted(by_operator.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", default="configs/v21-final-dataset-seal.json")
    parser.add_argument("--features", default="outputs/v21-final/features")
    parser.add_argument("--output-dir", default="outputs/v21-final/evaluation")
    args = parser.parse_args()
    seal_path = PROJECT_ROOT / args.seal
    seal = json.loads(seal_path.read_text())
    execution_lock_path = PROJECT_ROOT / seal["execution_lock"]
    lock = json.loads(execution_lock_path.read_text())
    output_dir = PROJECT_ROOT / args.output_dir
    if output_dir.exists():
        raise RuntimeError(f"V21 evaluation exists; retry forbidden: {output_dir}")
    if lock["limits"]["finalEvaluationsPermitted"] != 1:
        raise RuntimeError("V21 lock does not authorize exactly one final evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V21 locked implementation changed: {path}")
    if file_sha256(execution_lock_path) != seal["execution_lock_sha256"]:
        raise RuntimeError("V21 execution lock changed after dataset sealing")
    for source_name in ("deployment_heads", "v20_result"):
        source_path = PROJECT_ROOT / lock["source"][source_name]
        if file_sha256(source_path) != lock["source"][f"{source_name}_sha256"]:
            raise RuntimeError(f"V21 source artifact changed: {source_name}")
    metadata_path = PROJECT_ROOT / args.features / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["dataset_seal_sha256"] != file_sha256(seal_path):
        raise RuntimeError("V21 features and dataset seal differ")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V21 feature artifact changed")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "attempt.json").write_text(json.dumps({
        "schema_version": 21,
        "evaluation_number": 1,
        "dataset_seal": args.seal,
        "dataset_seal_sha256": file_sha256(seal_path),
        "feature_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        "feature_metadata_sha256": file_sha256(metadata_path),
        "retry_authorized": False,
    }, indent=2, sort_keys=True) + "\n")
    arrays = load_npz(feature_path)
    manifest_path = PROJECT_ROOT / seal["manifest"]
    episodes = [
        json.loads(line)
        for line in (manifest_path.parent / "episodes.jsonl").read_text().splitlines() if line
    ]
    scenes = read_scenes(manifest_path.parent)
    if arrays["scene_ids"].tolist() != [value["id"] for value in scenes]:
        raise RuntimeError("V21 scene and feature order differ")
    heads = load_npz(PROJECT_ROOT / lock["source"]["deployment_heads"])
    scored = score_scenes(scenes, arrays, heads)
    hard = hard_predictions(scored)
    v20_result = json.loads((PROJECT_ROOT / lock["source"]["v20_result"]).read_text())
    thresholds = {
        view: v20_result["views"][view]["calibration"] for view in lock["config"]["views"]
    }
    probabilistic = apply_interface(scored, thresholds)
    hard_lookup = {
        view: {
            (value["episode_id"], value["source_item_id"]): value
            for value in hard if value["view"] == view
        }
        for view in lock["config"]["views"]
    }
    probabilistic_lookup = {
        view: {
            (value["episode_id"], value["source_item_id"]): value
            for value in probabilistic if value["view"] == view
        }
        for view in lock["config"]["views"]
    }
    hard_modes = {
        "oracle_support_oracle_query": ("oracle", "oracle"),
        "frozen_support_oracle_query": ("frozen", "oracle"),
        "oracle_support_frozen_query": ("oracle", "frozen"),
        "frozen_support_frozen_query": ("frozen", "frozen"),
    }
    views = {}
    for view, role in lock["config"]["views"].items():
        hard_conditions = {
            name: evaluate_hard_condition(episodes, hard_lookup[view], support_mode, query_mode)
            for name, (support_mode, query_mode) in hard_modes.items()
        }
        probabilistic_full = evaluate_probabilistic_condition(
            episodes, probabilistic_lookup[view], "probabilistic", "probabilistic",
            lock["probabilistic_interface"]["credibleProgramMass"],
        )
        views[view] = {
            "role": role,
            "grounding": grounding_summary(scored, probabilistic, {value["id"]: value for value in scenes}, view),
            "hard_conditions": hard_conditions,
            "probabilistic_full": probabilistic_full,
        }
    hard_primary = views["supported"]["hard_conditions"]["frozen_support_frozen_query"]
    oracle = views["supported"]["hard_conditions"]["oracle_support_oracle_query"]
    probabilistic_primary = views["supported"]["probabilistic_full"]
    hard_gate = lock["config"]["gates"]["hardSupported"]
    hard_family_complete = {
        family: hard_primary["by_axis"][family]["complete_episodes"]
        for family in lock["config"]["constructionFamilies"]
    }
    hard_checks = {
        "mechanic_macro": (
            hard_primary["episode_metrics"]["episode_macro_transition_set_exact_match"]
            >= hard_gate["minimumMechanicMacroTransitionSetExact"]
        ),
        "complete_mechanics": (
            hard_primary["episode_metrics"]["complete_episodes"] >= hard_gate["minimumCompleteMechanics"]
        ),
        "complete_per_family": all(
            hard_family_complete[family] >= minimum
            for family, minimum in hard_gate["minimumCompletePerFamily"].items()
        ),
        "target_retention": (
            hard_primary["schema_recovery"]["target_retention_rate"]
            >= hard_gate["minimumTargetRetention"]
        ),
        "empty_version_rate": (
            hard_primary["schema_recovery"]["empty_version_space_rate"]
            <= hard_gate["maximumEmptyVersionRate"]
        ),
        "oracle_ceiling": (
            oracle["episode_metrics"]["episode_macro_transition_set_exact_match"]
            >= hard_gate["minimumOracleCeiling"]
        ),
    }
    hard_pass = all(hard_checks.values())
    prob_gate = lock["config"]["gates"]["probabilisticSupported"]
    prob_checks = {
        "eligible": lock["challenger_eligible"],
        "mechanic_macro": (
            probabilistic_primary["episode_metrics"]["episode_macro_transition_set_exact_match"]
            >= prob_gate["minimumMechanicMacroTransitionSetExact"]
        ),
        "complete_mechanics": (
            probabilistic_primary["episode_metrics"]["complete_episodes"]
            >= prob_gate["minimumCompleteMechanics"]
        ),
        "target_retention": (
            probabilistic_primary["schema_recovery"]["target_credible_retention_rate"]
            >= prob_gate["minimumTargetRetention"]
        ),
        "empty_posterior_rate": (
            probabilistic_primary["schema_recovery"]["empty_posterior_rate"]
            <= prob_gate["maximumEmptyPosteriorRate"]
        ),
        "excess_outcomes": (
            probabilistic_primary["anti_widening"]["mean_excess_outcomes"]
            <= prob_gate["maximumMeanExcessOutcomesPerQuery"]
        ),
    }
    prob_pass = all(prob_checks.values())
    if hard_pass and prob_pass:
        decision = "hard_population_transfer_passes_probabilistic_also_passes_proceed_relational"
    elif hard_pass:
        decision = "hard_population_transfer_passes_probabilistic_fails_proceed_relational"
    elif prob_pass:
        decision = "hard_population_claim_rejected_probabilistic_modularity_supported"
    else:
        decision = "both_supported_systems_fail_population_robustness"
    complete = hard_primary["episode_metrics"]["complete_episodes"]
    result = {
        "schema_version": 21,
        "experiment": lock["config"]["experiment"],
        "dataset_seal": args.seal,
        "dataset_seal_sha256": file_sha256(seal_path),
        "feature_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        "feature_metadata_sha256": file_sha256(metadata_path),
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "evaluation_number": 1,
        "views": views,
        "hard_supported_checks": hard_checks,
        "hard_supported_passed": hard_pass,
        "probabilistic_supported_checks": prob_checks,
        "probabilistic_supported_passed": prob_pass,
        "complete_mechanic_exact_95_interval": exact_interval(complete, 40),
        "complete_mechanics_by_family": hard_family_complete,
        "decision": decision,
        "lora_authorized": False,
        "retry_authorized": False,
        "data_access": {
            "final_records_read": len(episodes),
            "final_scenes_read": len(scenes),
            "final_feature_artifacts_read": 1,
            "final_labels_used_for_fitting_or_selection": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "evaluation_number": 1,
        },
    }
    hard_path = output_dir / "hard-grounding-predictions.jsonl"
    hard_path.write_text("".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in hard
    ))
    prob_path = output_dir / "probabilistic-grounding-predictions.jsonl"
    prob_path.write_text("".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in probabilistic
    ))
    result["hard_predictions"] = str(hard_path.relative_to(PROJECT_ROOT))
    result["hard_predictions_sha256"] = file_sha256(hard_path)
    result["probabilistic_predictions"] = str(prob_path.relative_to(PROJECT_ROOT))
    result["probabilistic_predictions_sha256"] = file_sha256(prob_path)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": decision,
        "hard_supported_checks": hard_checks,
        "probabilistic_supported_checks": prob_checks,
        "hard_supported": hard_primary,
        "probabilistic_supported": probabilistic_primary,
        "complete_mechanic_exact_95_interval": result["complete_mechanic_exact_95_interval"],
        "novel_hard": views["novel_ontology"]["hard_conditions"]["frozen_support_frozen_query"],
        "novel_probabilistic": views["novel_ontology"]["probabilistic_full"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
