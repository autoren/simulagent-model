"""Run the single locked V20 uncertainty-interface development evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from audit_v18_benchmark import read_records
from audit_v19_compatibility import read_scenes
from evaluate_v10_frozen import build_pair_lookup
from evaluate_v15_full_pipeline import nli_pairs_by_base
from evaluate_v19_frozen_integration import frozen_pipeline, load_npz
from run_v18_schema_baselines import episode_summary, prediction_rows, summarize_predictions
from v10_protocol import TEMPORAL_ORDER, VALUE_ORDER, derive_allowed_values, file_sha256
from v18_schema import enumerate_program_hypotheses
from v20_probabilistic_grounding import (
    assignment_distribution,
    conformal_label_set,
    conformal_quantile,
    credible_hypothesis_indices,
    polarity_probabilities,
    posterior_answer,
    posterior_diagnostics,
    program_posterior,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v20-probabilistic-interface-lock.json")
    parser.add_argument("--output-dir", default="outputs/v20-probabilistic-interface/evaluation")
    return parser.parse_args()


def score_scenes(
    scenes: Sequence[dict[str, Any]], arrays: dict[str, np.ndarray], heads: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    base = arrays["base_span_features"].astype(np.float32)
    pair_base = arrays["pair_base_indices"].astype(np.int32)
    pair_scenes = arrays["pair_scene_indices"].astype(np.int32)
    determinant_indices = arrays["determinant_indices"].astype(np.int8)
    evidence_indices = arrays["evidence_indices"].astype(np.int8)
    match_targets = arrays["match_targets"].astype(bool)
    nli = arrays["nli_hypothesis_mean_features"].astype(np.float32)
    nli_by_base = nli_pairs_by_base(
        pair_base, arrays["pair_nli_indices"].astype(np.int32), len(base)
    )
    polarity_features = nli[nli_by_base[:, 0]] - nli[nli_by_base[:, 1]]
    match_scores = frozen_pipeline(heads, "match", base, decision=True)
    temporal_predictions = frozen_pipeline(heads, "temporal", base).astype(np.int8)
    polarity_scores = frozen_pipeline(heads, "polarity", polarity_features, decision=True)
    pair_lookup = build_pair_lookup(pair_scenes, determinant_indices)
    result = []
    for scene_index, scene in enumerate(scenes):
        groundings = []
        for determinant_index, target in enumerate(scene["target"]["determinant_grounding"]):
            candidates = pair_lookup[(scene_index, determinant_index)]
            selected = max(candidates, key=lambda index: float(match_scores[pair_base[index]]))
            gold = next(index for index in candidates if match_targets[index])
            base_index = int(pair_base[selected])
            temporal = TEMPORAL_ORDER[int(temporal_predictions[base_index])]
            hard_current = VALUE_ORDER[int(polarity_scores[base_index] > 0.0)]
            relations = (
                ["ENTAILED", "CONTRADICTED"]
                if hard_current == "active" else ["CONTRADICTED", "ENTAILED"]
            )
            hard_allowed = derive_allowed_values(temporal, relations)
            probabilities = polarity_probabilities(float(polarity_scores[base_index]))
            groundings.append({
                "determinant_id": target["latent_determinant_id"],
                "selected_base_index": base_index,
                "selected_evidence_index": int(evidence_indices[selected]),
                "gold_evidence_index": int(evidence_indices[gold]),
                "span_correct": selected == gold,
                "predicted_temporal_status": temporal,
                "temporal_correct": temporal == target["temporal_status"],
                "hard_current_value": hard_current if temporal == "CURRENT" else None,
                "hard_allowed_values": hard_allowed,
                "polarity_score": float(polarity_scores[base_index]),
                "value_probabilities": probabilities,
                "target_current_value": target["current_value"],
                "target_allowed_values": target["allowed_values"],
            })
        result.append({
            "scene_id": scene["id"],
            "episode_id": scene["episode_id"],
            "source_item_id": scene["source_item_id"],
            "view": scene["view"],
            "split": scene["split"],
            "axis": scene["generalization_axis"],
            "item_kind": scene["item_kind"],
            "groundings": groundings,
        })
    return result


def calibration_threshold(
    scored: Sequence[dict[str, Any]], view: str, split: str, alpha: float
) -> dict[str, Any]:
    unique: dict[int, tuple[str, float]] = {}
    for scene in scored:
        if scene["view"] != view or scene["split"] != split:
            continue
        for grounding in scene["groundings"]:
            target = grounding["target_current_value"]
            if target is None:
                continue
            key = grounding["selected_base_index"]
            score = 1.0 - grounding["value_probabilities"][target]
            existing = unique.get(key)
            if existing is not None and existing[0] != target:
                raise RuntimeError("A selected calibration prompt has conflicting polarity targets")
            unique[key] = (target, score)
    values = [value[1] for value in unique.values()]
    threshold = conformal_quantile(values, alpha)
    return {
        "view": view,
        "split": split,
        "alpha": alpha,
        "unique_selected_current_prompts": len(values),
        "threshold": threshold,
        "empirical_coverage": float(np.mean([value <= threshold for value in values])),
        "maximum_nonconformity": max(values),
    }


def apply_interface(
    scored: Sequence[dict[str, Any]], thresholds: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    predictions = []
    for scene in scored:
        groundings = []
        threshold = thresholds[scene["view"]]["threshold"]
        for value in scene["groundings"]:
            if value["predicted_temporal_status"] == "CURRENT":
                allowed = conformal_label_set(value["value_probabilities"], threshold)
                source = "conformal_current_polarity"
            else:
                allowed = ["inactive", "active"]
                source = "semantic_unresolved"
            groundings.append({
                "determinant_id": value["determinant_id"],
                "allowed_values": allowed,
                "value_probabilities": value["value_probabilities"],
                "hard_allowed_values": value["hard_allowed_values"],
                "target_allowed_values": value["target_allowed_values"],
                "polarity_score": value["polarity_score"],
                "uncertainty_source": source,
                "span_correct": value["span_correct"],
                "temporal_correct": value["temporal_correct"],
            })
        predictions.append({**{key: scene[key] for key in (
            "scene_id", "episode_id", "source_item_id", "view", "split", "axis", "item_kind"
        )}, "groundings": groundings})
    return predictions


def assert_hard_reproduction(
    scored: Sequence[dict[str, Any]], source_predictions_path: Path
) -> None:
    source = {
        value["scene_id"]: value
        for value in (json.loads(line) for line in source_predictions_path.read_text().splitlines())
    }
    if set(source) != {value["scene_id"] for value in scored}:
        raise RuntimeError("V20 hard reproduction scene set differs from V19")
    for scene in scored:
        expected = source[scene["scene_id"]]
        for actual, gold in zip(scene["groundings"], expected["groundings"], strict=True):
            if actual["hard_allowed_values"] != gold["allowed_values"]:
                raise RuntimeError(f"V20 hard grounding differs for {scene['scene_id']}")
            if actual["selected_evidence_index"] != gold["selected_evidence_index"]:
                raise RuntimeError(f"V20 evidence selection differs for {scene['scene_id']}")


def oracle_groundings(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "determinant_id": value["determinant_id"],
            "allowed_values": value["allowed_values"],
            "value_probabilities": {"inactive": 0.5, "active": 0.5},
        }
        for value in values
    ]


def support_traces(
    episode: dict[str, Any], lookup: dict[tuple[str, str], dict[str, Any]], mode: str
) -> list[dict[str, Any]]:
    observed = {
        value["trace_id"]: value["observed_transition_code"]
        for value in episode["agent_input"]["support_traces"]
    }
    rows = []
    for target in episode["oracle_grounding"]["support"]:
        trace_id = target["trace_id"]
        if mode == "oracle":
            groundings = [
                {
                    "determinant_id": identifier,
                    "allowed_values": ["active" if target["assignment"][identifier] else "inactive"],
                    "value_probabilities": {"inactive": 0.0, "active": 1.0}
                    if target["assignment"][identifier]
                    else {"inactive": 1.0, "active": 0.0},
                }
                for identifier in target["assignment"]
            ]
        else:
            groundings = lookup[(episode["id"], trace_id)]["groundings"]
        determinant_ids = tuple(value["id"] for value in episode["agent_input"]["determinant_ontology"])
        rows.append({
            "transition_code": observed[trace_id],
            "assignments": assignment_distribution(determinant_ids, groundings),
        })
    return rows


def evaluate_condition(
    episodes: Sequence[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
    support_mode: str,
    query_mode: str,
    credible_mass: float,
) -> dict[str, Any]:
    rows = []
    episode_rows = []
    for episode in episodes:
        determinant_ids = tuple(value["id"] for value in episode["agent_input"]["determinant_ontology"])
        outcome_bits = episode["agent_input"]["dsl_contract"]["outcome_bits"]
        hypotheses = list(enumerate_program_hypotheses(determinant_ids, outcome_bits))
        target_signature = tuple(episode["target"]["behavioral_signature"])
        target_index = next(
            index for index, value in enumerate(hypotheses) if value.signature == target_signature
        )
        traces = support_traces(episode, lookup, support_mode)
        posterior = program_posterior(hypotheses, determinant_ids, traces)
        if support_mode == "oracle":
            selected = [index for index, value in enumerate(posterior) if value > 0.0]
        else:
            selected = credible_hypothesis_indices(hypotheses, posterior, credible_mass)
        answers = []
        predicted_sizes = []
        target_sizes = []
        excess = []
        missing = []
        for target in episode["oracle_grounding"]["queries"]:
            if query_mode == "oracle":
                groundings = oracle_groundings(target["allowed_values"])
            else:
                groundings = lookup[(episode["id"], target["query_id"])]["groundings"]
            assignments = assignment_distribution(determinant_ids, groundings)
            answer = posterior_answer(hypotheses, selected, assignments, outcome_bits)
            answers.append(answer)
            predicted = set(answer["possible_transition_codes"])
            gold = set(target["possible_transition_codes"])
            predicted_sizes.append(len(predicted))
            target_sizes.append(len(gold))
            excess.append(len(predicted - gold))
            missing.append(len(gold - predicted))
        query_rows = prediction_rows(episode, answers)
        rows.extend(query_rows)
        query_accuracy = float(np.mean([value["transition_set_exact"] for value in query_rows]))
        diagnostics = posterior_diagnostics(posterior, selected)
        episode_rows.append({
            "episode_id": episode["id"],
            "axis": episode["generalization_axis"],
            "query_accuracy": query_accuracy,
            "complete": query_accuracy == 1.0,
            "target_nonzero": bool(posterior[target_index] > 0.0),
            "target_credible": target_index in selected,
            "target_posterior": float(posterior[target_index]),
            "empty": not selected,
            "mean_predicted_set_size": float(np.mean(predicted_sizes)),
            "mean_target_set_size": float(np.mean(target_sizes)),
            "mean_excess_outcomes": float(np.mean(excess)),
            "mean_missing_target_outcomes": float(np.mean(missing)),
            **diagnostics,
        })
    anti_widening = {
        key: float(np.mean([value[key] for value in episode_rows]))
        for key in (
            "mean_predicted_set_size", "mean_target_set_size",
            "mean_excess_outcomes", "mean_missing_target_outcomes",
        )
    }
    schema = {
        "target_nonzero_retention_rate": float(np.mean([value["target_nonzero"] for value in episode_rows])),
        "target_credible_retention_rate": float(np.mean([value["target_credible"] for value in episode_rows])),
        "empty_posterior_rate": float(np.mean([value["empty"] for value in episode_rows])),
        "mean_target_posterior": float(np.mean([value["target_posterior"] for value in episode_rows])),
        "median_nonzero_programs": float(np.median([value["nonzero_programs"] for value in episode_rows])),
        "median_credible_programs": float(np.median([value["credible_programs"] for value in episode_rows])),
        "mean_posterior_entropy_nats": float(np.mean([value["posterior_entropy_nats"] for value in episode_rows])),
        "mean_posterior_effective_programs": float(np.mean([value["posterior_effective_programs"] for value in episode_rows])),
    }
    by_axis = {}
    for axis in sorted({value["axis"] for value in episode_rows}):
        axis_rows = [value for value in rows if value["axis"] == axis]
        axis_episodes = [value for value in episode_rows if value["axis"] == axis]
        by_axis[axis] = {
            **summarize_predictions(axis_rows),
            **episode_summary(axis_rows),
            "target_credible_retention_rate": float(np.mean([value["target_credible"] for value in axis_episodes])),
            "empty_posterior_rate": float(np.mean([value["empty"] for value in axis_episodes])),
            "mean_excess_outcomes": float(np.mean([value["mean_excess_outcomes"] for value in axis_episodes])),
        }
    return {
        "support_mode": support_mode,
        "query_mode": query_mode,
        "query_metrics": summarize_predictions(rows),
        "episode_metrics": episode_summary(rows),
        "schema_recovery": schema,
        "anti_widening": anti_widening,
        "by_axis": by_axis,
        "episodes": episode_rows,
    }


def coverage_summary(predictions: Sequence[dict[str, Any]], view: str, split: str) -> dict[str, Any]:
    selected = [value for value in predictions if value["view"] == view and value["split"] == split]
    current = [
        grounding for scene in selected for grounding in scene["groundings"]
        if grounding["uncertainty_source"] == "conformal_current_polarity"
    ]
    return {
        "scenes": len(selected),
        "current_determinants": len(current),
        "marginal_label_coverage": float(np.mean([
            grounding["target_allowed_values"][0] in grounding["allowed_values"] for grounding in current
        ])),
        "mean_label_set_size": float(np.mean([len(grounding["allowed_values"]) for grounding in current])),
        "singleton_rate": float(np.mean([len(grounding["allowed_values"]) == 1 for grounding in current])),
    }


def main() -> None:
    args = parse_args()
    lock_path = PROJECT_ROOT / args.lock
    lock = json.loads(lock_path.read_text())
    output_dir = PROJECT_ROOT / args.output_dir
    if output_dir.exists():
        raise RuntimeError(f"V20 evaluation already exists; retry forbidden: {output_dir}")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V20 locked implementation changed: {path}")
    config = lock["config"]
    v19_lock_path = PROJECT_ROOT / config["sourceV19Lock"]
    v19_lock = json.loads(v19_lock_path.read_text())
    metadata_path = PROJECT_ROOT / config["sourceV19Features"]
    metadata = json.loads(metadata_path.read_text())
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != lock["source"]["v19_feature_artifact_sha256"]:
        raise RuntimeError("V19 feature artifact changed")
    arrays = load_npz(feature_path)
    scenes = read_scenes(PROJECT_ROOT / v19_lock["source"]["v19_dataset"])
    if arrays["scene_ids"].tolist() != [value["id"] for value in scenes]:
        raise RuntimeError("V19 scenes and saved features differ")
    heads = load_npz(PROJECT_ROOT / config["sourceDeploymentHeads"])
    scored = score_scenes(scenes, arrays, heads)
    v19_result = json.loads((PROJECT_ROOT / config["sourceV19Result"]).read_text())
    assert_hard_reproduction(scored, PROJECT_ROOT / v19_result["grounding_predictions"])
    thresholds = {
        view: calibration_threshold(
            scored, view, config["calibrationSplit"], config["interface"]["alpha"]
        )
        for view in config["views"]
    }
    predictions = apply_interface(scored, thresholds)
    prediction_lookup = {
        view: {
            (value["episode_id"], value["source_item_id"]): value
            for value in predictions if value["view"] == view
        }
        for view in config["views"]
    }
    episodes = [
        value for value in read_records(PROJECT_ROOT / v19_lock["source"]["v18_dataset"])
        if value["split"] == config["evaluationSplit"]
    ]
    condition_modes = {
        "probabilistic_support_oracle_query": ("probabilistic", "oracle"),
        "oracle_support_probabilistic_query": ("oracle", "probabilistic"),
        "probabilistic_support_probabilistic_query": ("probabilistic", "probabilistic"),
    }
    views = {}
    for view, role in config["views"].items():
        conditions = {
            "hard_support_hard_query": v19_result["views"][view]["conditions"]["frozen_support_frozen_query"]
        }
        for name, (support_mode, query_mode) in condition_modes.items():
            conditions[name] = evaluate_condition(
                episodes, prediction_lookup[view], support_mode, query_mode,
                config["interface"]["credibleProgramMass"],
            )
        views[view] = {
            "role": role,
            "calibration": thresholds[view],
            "calibration_coverage": coverage_summary(
                predictions, view, config["calibrationSplit"]
            ),
            "development_coverage": coverage_summary(
                predictions, view, config["evaluationSplit"]
            ),
            "conditions": conditions,
        }
    supported = views["supported"]["conditions"]["probabilistic_support_probabilistic_query"]
    novel = views["novel_ontology"]["conditions"]["probabilistic_support_probabilistic_query"]
    novel_hard = views["novel_ontology"]["conditions"]["hard_support_hard_query"]
    preservation = config["gates"]["supportedPreservation"]
    improvement = config["gates"]["novelDiagnosticImprovement"]
    checks = {
        "supported_episode_macro_preserved": (
            supported["episode_metrics"]["episode_macro_transition_set_exact_match"]
            >= preservation["minimumEpisodeMacroTransitionSetExact"]
        ),
        "supported_complete_episodes_preserved": (
            supported["episode_metrics"]["complete_episodes"] >= preservation["minimumCompleteEpisodes"]
        ),
        "supported_credible_target_retention": (
            supported["schema_recovery"]["target_credible_retention_rate"]
            >= preservation["minimumCredibleTargetRetention"]
        ),
        "supported_no_empty_posterior": (
            supported["schema_recovery"]["empty_posterior_rate"]
            <= preservation["maximumEmptyPosteriorRate"]
        ),
        "novel_episode_macro_nonnegative_gain": (
            novel["episode_metrics"]["episode_macro_transition_set_exact_match"]
            - novel_hard["episode_metrics"]["episode_macro_transition_set_exact_match"]
            >= improvement["minimumEpisodeMacroGainOverHard"]
        ),
        "novel_target_retention_nonnegative_gain": (
            novel["schema_recovery"]["target_credible_retention_rate"]
            - novel_hard["schema_recovery"]["target_retention_rate"]
            >= improvement["minimumCredibleTargetRetentionGainOverHard"]
        ),
        "novel_empty_posterior_bounded": (
            novel["schema_recovery"]["empty_posterior_rate"]
            <= improvement["maximumEmptyPosteriorRate"]
        ),
        "novel_excess_outcomes_bounded": (
            novel["anti_widening"]["mean_excess_outcomes"]
            <= improvement["maximumMeanExcessOutcomesPerQuery"]
        ),
    }
    supported_pass = all(value for key, value in checks.items() if key.startswith("supported_"))
    novel_pass = all(value for key, value in checks.items() if key.startswith("novel_"))
    decision = (
        "freeze_probabilistic_interface_as_final_challenger"
        if supported_pass and novel_pass else
        "preserve_only_negative_probabilistic_development_result"
        if supported_pass else
        "exclude_probabilistic_interface_from_final_scoring"
    )
    result = {
        "schema_version": 20,
        "experiment": config["experiment"],
        "protocol_lock": args.lock,
        "protocol_lock_sha256": file_sha256(lock_path),
        "evaluation_number": 1,
        "views": views,
        "checks": checks,
        "supported_preservation_passed": supported_pass,
        "novel_diagnostic_improvement_passed": novel_pass,
        "passed": supported_pass and novel_pass,
        "decision": decision,
        "lora_authorized": False,
        "data_access": {
            "saved_feature_artifacts_read": 1,
            "new_model_forward_passes": 0,
            "new_feature_extractions": 0,
            "new_linear_fits": 0,
            "adapter_training_runs": 0,
            "final_suite_records_created_or_read": 0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "probabilistic-grounding-predictions.jsonl"
    prediction_path.write_text("".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in predictions
    ))
    result["predictions"] = str(prediction_path.relative_to(PROJECT_ROOT))
    result["predictions_sha256"] = file_sha256(prediction_path)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "checks": checks,
        "decision": decision,
        "thresholds": thresholds,
        "supported": supported,
        "novel": novel,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
