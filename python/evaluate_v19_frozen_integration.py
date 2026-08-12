"""Evaluate the locked V19 two-view, two-by-two frozen grounding integration."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from audit_v18_benchmark import read_records
from audit_v19_compatibility import read_scenes
from evaluate_v10_frozen import build_pair_lookup
from evaluate_v15_full_pipeline import nli_pairs_by_base
from run_v18_schema_baselines import (
    compatible_assignment_indices, episode_summary, outcome_vocabulary,
    prediction_rows, summarize_predictions, version_space_answer,
)
from v10_protocol import RELATION_ORDER, TEMPORAL_ORDER, VALUE_ORDER, derive_allowed_values, file_sha256
from v18_schema import allowed_trace_consistent_hypotheses, enumerate_program_hypotheses


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v19-frozen-integration-lock.json")
    parser.add_argument("--features", default="outputs/v19-frozen-integration/features")
    parser.add_argument("--output-dir", default="outputs/v19-frozen-integration/evaluation")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def frozen_pipeline(
    payload: dict[str, np.ndarray], prefix: str, values: np.ndarray,
    decision: bool = False,
) -> np.ndarray:
    scaled = (values - payload[f"{prefix}_scaler_mean"]) / payload[f"{prefix}_scaler_scale"]
    scores = scaled @ payload[f"{prefix}_coef"].T + payload[f"{prefix}_intercept"]
    classes = payload[f"{prefix}_classes"]
    if scores.shape[1] == 1:
        if decision:
            return scores[:, 0].astype(np.float32)
        return classes[(scores[:, 0] > 0).astype(np.int8)]
    if decision:
        return scores.astype(np.float32)
    return classes[np.argmax(scores, axis=1)]


def ground_scenes(
    scenes: list[dict[str, Any]], arrays: dict[str, np.ndarray], heads: dict[str, np.ndarray]
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
    polarity = nli[nli_by_base[:, 0]] - nli[nli_by_base[:, 1]]
    unique_match_scores = frozen_pipeline(heads, "match", base, decision=True)
    unique_temporal = frozen_pipeline(heads, "temporal", base).astype(np.int8)
    unique_current = frozen_pipeline(heads, "polarity", polarity).astype(np.int8)
    pair_lookup = build_pair_lookup(pair_scenes, determinant_indices)
    predictions = []
    for scene_index, scene in enumerate(scenes):
        groundings = []
        span_correct = []
        temporal_correct = []
        polarity_correct = []
        allowed_correct = []
        for determinant_index, target in enumerate(scene["target"]["determinant_grounding"]):
            candidates = pair_lookup[(scene_index, determinant_index)]
            selected = max(candidates, key=lambda index: float(unique_match_scores[pair_base[index]]))
            gold = next(index for index in candidates if match_targets[index])
            base_index = int(pair_base[selected])
            temporal = TEMPORAL_ORDER[int(unique_temporal[base_index])]
            current = VALUE_ORDER[int(unique_current[base_index])]
            relations = (
                ["ENTAILED", "CONTRADICTED"] if current == "active"
                else ["CONTRADICTED", "ENTAILED"]
            )
            allowed = derive_allowed_values(temporal, relations)
            span_ok = selected == gold
            temporal_ok = temporal == target["temporal_status"]
            allowed_ok = allowed == target["allowed_values"]
            span_correct.append(span_ok)
            temporal_correct.append(temporal_ok)
            allowed_correct.append(allowed_ok)
            if target["current_value"] is not None:
                polarity_correct.append(current == target["current_value"])
            groundings.append({
                "determinant_id": target["latent_determinant_id"],
                "allowed_values": allowed,
                "predicted_temporal_status": temporal,
                "predicted_current_value": current if temporal == "CURRENT" else None,
                "selected_evidence_index": int(evidence_indices[selected]),
                "span_correct": span_ok,
                "temporal_correct": temporal_ok,
                "allowed_correct": allowed_ok,
            })
        predictions.append({
            "scene_id": scene["id"],
            "episode_id": scene["episode_id"],
            "source_item_id": scene["source_item_id"],
            "view": scene["view"],
            "split": scene["split"],
            "axis": scene["generalization_axis"],
            "item_kind": scene["item_kind"],
            "groundings": groundings,
            "span_accuracy": float(np.mean(span_correct)),
            "temporal_accuracy": float(np.mean(temporal_correct)),
            "polarity_accuracy": float(np.mean(polarity_correct)) if polarity_correct else None,
            "allowed_value_accuracy": float(np.mean(allowed_correct)),
            "scene_grounding_exact": all(allowed_correct),
            "grounding_errors": len(allowed_correct) - sum(allowed_correct),
        })
    return predictions


def grounding_summary(
    predictions: Sequence[dict[str, Any]], scenes_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    determinant_rows = []
    episode_support_exact: dict[str, list[bool]] = defaultdict(list)
    for prediction in predictions:
        scene = scenes_by_id[prediction["scene_id"]]
        if prediction["item_kind"] == "support":
            episode_support_exact[prediction["episode_id"]].append(prediction["scene_grounding_exact"])
        for predicted, target in zip(
            prediction["groundings"], scene["target"]["determinant_grounding"], strict=True
        ):
            target_class = (
                "unresolved" if target["allowed_values"] == ["inactive", "active"]
                else target["allowed_values"][0]
            )
            determinant_rows.append({
                "item_kind": prediction["item_kind"],
                "target_class": target_class,
                "allowed_correct": predicted["allowed_correct"],
                "span_correct": predicted["span_correct"],
                "temporal_correct": predicted["temporal_correct"],
                "polarity_correct": (
                    predicted["predicted_current_value"] == target["current_value"]
                    if target["current_value"] is not None else None
                ),
            })
    result = {
        "scenes": len(predictions),
        "determinants": len(determinant_rows),
        "span_accuracy": float(np.mean([value["span_correct"] for value in determinant_rows])),
        "temporal_accuracy": float(np.mean([value["temporal_correct"] for value in determinant_rows])),
        "allowed_value_accuracy": float(np.mean([value["allowed_correct"] for value in determinant_rows])),
        "exact_scene_grounding": float(np.mean([value["scene_grounding_exact"] for value in predictions])),
        "episodes_with_all_supports_exact": float(np.mean([
            all(values) for values in episode_support_exact.values()
        ])),
        "by_target_class": {},
    }
    current = [value for value in determinant_rows if value["polarity_correct"] is not None]
    result["current_polarity_accuracy"] = float(np.mean([value["polarity_correct"] for value in current]))
    for target_class in ("active", "inactive", "unresolved"):
        rows = [value for value in determinant_rows if value["target_class"] == target_class]
        result["by_target_class"][target_class] = {
            "determinants": len(rows),
            "allowed_value_accuracy": float(np.mean([value["allowed_correct"] for value in rows])),
        }
    return result


def oracle_support(episode: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = {
        value["trace_id"]: value["observed_transition_code"]
        for value in episode["agent_input"]["support_traces"]
    }
    return [
        {
            "allowed_values": [
                {
                    "determinant_id": identifier,
                    "allowed_values": ["active" if grounding["assignment"][identifier] else "inactive"],
                }
                for identifier in grounding["assignment"]
            ],
            "transition_code": outcomes[grounding["trace_id"]],
        }
        for grounding in episode["oracle_grounding"]["support"]
    ]


def predicted_support(
    episode: dict[str, Any], prediction_lookup: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    outcomes = {
        value["trace_id"]: value["observed_transition_code"]
        for value in episode["agent_input"]["support_traces"]
    }
    return [
        {
            "allowed_values": prediction_lookup[(episode["id"], grounding["trace_id"])]["groundings"],
            "transition_code": outcomes[grounding["trace_id"]],
        }
        for grounding in episode["oracle_grounding"]["support"]
    ]


def error_bucket(errors: int) -> str:
    return "zero" if errors == 0 else "one" if errors == 1 else "multiple"


def evaluate_condition(
    episodes: Sequence[dict[str, Any]],
    prediction_lookup: dict[tuple[str, str], dict[str, Any]],
    support_mode: str,
    query_mode: str,
) -> dict[str, Any]:
    query_rows = []
    episode_rows = []
    prefix: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        determinant_ids = tuple(value["id"] for value in episode["agent_input"]["determinant_ontology"])
        outcome_bits = episode["agent_input"]["dsl_contract"]["outcome_bits"]
        hypotheses = list(enumerate_program_hypotheses(determinant_ids, outcome_bits))
        target_signature = tuple(episode["target"]["behavioral_signature"])
        oracle = oracle_support(episode)
        support = oracle if support_mode == "oracle" else predicted_support(episode, prediction_lookup)
        support_error_count = 0
        if support_mode == "frozen":
            for predicted, target in zip(support, oracle, strict=True):
                support_error_count += sum(
                    value["allowed_values"] != gold["allowed_values"]
                    for value, gold in zip(predicted["allowed_values"], target["allowed_values"], strict=True)
                )
        current = hypotheses
        for index, trace in enumerate(support, start=1):
            current = allowed_trace_consistent_hypotheses(current, [trace], determinant_ids)
            retained = any(value.signature == target_signature for value in current)
            prefix[index].append({
                "target_retained": retained,
                "empty": not current,
                "version_space": len(current),
            })
        final_retained = any(value.signature == target_signature for value in current)
        final_empty = not current
        final_unique = len(current) == 1 and current[0].signature == target_signature
        answers = []
        query_grounding_errors = 0
        for query in episode["oracle_grounding"]["queries"]:
            if query_mode == "oracle":
                allowed = query["allowed_values"]
            else:
                predicted = prediction_lookup[(episode["id"], query["query_id"])]
                allowed = predicted["groundings"]
                query_grounding_errors += sum(
                    value["allowed_values"] != gold["allowed_values"]
                    for value, gold in zip(allowed, query["allowed_values"], strict=True)
                )
            if final_empty:
                possible = outcome_vocabulary(outcome_bits)
                answer = {"possible_transition_codes": possible, "identifiable": False}
            else:
                indices = compatible_assignment_indices(determinant_ids, allowed)
                answer = version_space_answer(current, indices)
            answers.append(answer)
        rows = prediction_rows(episode, answers)
        query_rows.extend(rows)
        episode_accuracy = sum(value["transition_set_exact"] for value in rows) / len(rows)
        episode_rows.append({
            "episode_id": episode["id"],
            "axis": episode["generalization_axis"],
            "query_accuracy": episode_accuracy,
            "complete": episode_accuracy == 1.0,
            "target_retained": final_retained,
            "empty": final_empty,
            "unique_target": final_unique,
            "remaining_hypotheses": len(current),
            "support_grounding_errors": support_error_count,
            "query_grounding_errors": query_grounding_errors,
        })
    schema = {
        "target_retention_rate": float(np.mean([value["target_retained"] for value in episode_rows])),
        "empty_version_space_rate": float(np.mean([value["empty"] for value in episode_rows])),
        "unique_target_recovery_rate": float(np.mean([value["unique_target"] for value in episode_rows])),
        "median_remaining_hypotheses": float(np.median([value["remaining_hypotheses"] for value in episode_rows])),
        "maximum_remaining_hypotheses": max(value["remaining_hypotheses"] for value in episode_rows),
        "prefix": {
            str(index): {
                "episodes": len(values),
                "target_retention_rate": float(np.mean([value["target_retained"] for value in values])),
                "empty_version_space_rate": float(np.mean([value["empty"] for value in values])),
                "median_version_space": float(np.median([value["version_space"] for value in values])),
            }
            for index, values in sorted(prefix.items())
        },
    }
    by_axis = {}
    for axis in sorted({value["generalization_axis"] for value in episodes}):
        rows = [value for value in query_rows if value["axis"] == axis]
        eps = [value for value in episode_rows if value["axis"] == axis]
        by_axis[axis] = {
            **summarize_predictions(rows),
            **episode_summary(rows),
            "target_retention_rate": float(np.mean([value["target_retained"] for value in eps])),
            "empty_version_space_rate": float(np.mean([value["empty"] for value in eps])),
        }
    conditioned = {}
    for bucket in ("zero", "one", "multiple"):
        selected = [value for value in episode_rows if error_bucket(value["support_grounding_errors"]) == bucket]
        conditioned[bucket] = {
            "episodes": len(selected),
            "episode_macro_transition_set_exact_match": (
                float(np.mean([value["query_accuracy"] for value in selected])) if selected else None
            ),
            "target_retention_rate": (
                float(np.mean([value["target_retained"] for value in selected])) if selected else None
            ),
            "empty_version_space_rate": (
                float(np.mean([value["empty"] for value in selected])) if selected else None
            ),
        }
    return {
        "support_mode": support_mode,
        "query_mode": query_mode,
        "query_metrics": summarize_predictions(query_rows),
        "episode_metrics": episode_summary(query_rows),
        "schema_recovery": schema,
        "by_axis": by_axis,
        "conditioned_on_support_grounding_errors": conditioned,
        "support_grounding_errors": dict(Counter(value["support_grounding_errors"] for value in episode_rows)),
        "query_grounding_errors": dict(Counter(value["query_grounding_errors"] for value in episode_rows)),
    }


def condition_modes(name: str) -> tuple[str, str]:
    mapping = {
        "oracle_support_oracle_query": ("oracle", "oracle"),
        "frozen_support_oracle_query": ("frozen", "oracle"),
        "oracle_support_frozen_query": ("oracle", "frozen"),
        "frozen_support_frozen_query": ("frozen", "frozen"),
    }
    return mapping[name]


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# V19 results: frozen grounding × executable schema induction",
        "",
        f"Decision: `{result['decision']}`.",
        "",
        "## Grounding views",
        "",
        "| View | Role | Allowed-value accuracy | Exact scenes | All-support episodes |",
        "|---|---|---:|---:|---:|",
    ]
    for view, values in result["grounding"].items():
        lines.append(
            f"| `{view}` | {values['role']} | {values['development']['allowed_value_accuracy']:.3f} | {values['development']['exact_scene_grounding']:.3f} | {values['development']['episodes_with_all_supports_exact']:.3f} |"
        )
    lines.extend([
        "",
        "## Supported-view decomposition",
        "",
        "| Condition | Episode-macro transition-set exact | Complete episodes | Target retained | Empty version |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, values in result["views"]["supported"]["conditions"].items():
        episode = values["episode_metrics"]
        schema = values["schema_recovery"]
        lines.append(
            f"| `{name}` | {episode['episode_macro_transition_set_exact_match']:.3f} | {episode['complete_episodes']}/{episode['episodes']} | {schema['target_retention_rate']:.3f} | {schema['empty_version_space_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        result["interpretation"],
        "",
        "The novel-ontology view is diagnostic and does not affect the primary decision. V19 uses",
        "no adapter training, head refitting, target-guided repair, support deletion, or DSL expansion.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    feature_root = (PROJECT_ROOT / args.features).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        raise RuntimeError(f"V19 evaluation directory already exists; refusing a retry: {output_dir}")
    lock = json.loads(lock_path.read_text())
    if lock["limits"]["integrationEvaluationsPermitted"] != 1:
        raise RuntimeError("V19 lock does not authorize exactly one integration evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V19 locked implementation changed: {path}")
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path) or metadata["feature_extraction_number"] != 1:
        raise RuntimeError("V19 features do not share the one-shot lock")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V19 feature artifact changed")
    arrays = load_npz(feature_path)
    dataset_dir = PROJECT_ROOT / lock["source"]["v19_dataset"]
    scenes = read_scenes(dataset_dir)
    if arrays["scene_ids"].tolist() != [value["id"] for value in scenes]:
        raise RuntimeError("V19 scene and feature order differ")
    head_path = PROJECT_ROOT / lock["source"]["deployment_heads"]
    if file_sha256(head_path) != lock["source"]["deployment_heads_sha256"]:
        raise RuntimeError("Frozen V15 deployment heads changed")
    heads = load_npz(head_path)
    predictions = ground_scenes(scenes, arrays, heads)
    prediction_lookup_by_view = {
        view: {
            (value["episode_id"], value["source_item_id"]): value
            for value in predictions if value["view"] == view
        }
        for view in lock["views"]
    }
    episodes = read_records(PROJECT_ROOT / lock["source"]["v18_dataset"])
    primary_episodes = [value for value in episodes if value["split"] == lock["primary_split"]]
    scenes_by_id = {value["id"]: value for value in scenes}
    grounding = {}
    views = {}
    for view, role in lock["views"].items():
        view_predictions = [value for value in predictions if value["view"] == view]
        development_predictions = [value for value in view_predictions if value["split"] == lock["primary_split"]]
        grounding[view] = {
            "role": role,
            "all": grounding_summary(view_predictions, scenes_by_id),
            "development": grounding_summary(development_predictions, scenes_by_id),
        }
        views[view] = {"role": role, "conditions": {}}
        for condition in lock["conditions"]:
            support_mode, query_mode = condition_modes(condition)
            views[view]["conditions"][condition] = evaluate_condition(
                primary_episodes, prediction_lookup_by_view[view], support_mode, query_mode
            )

    primary = views["supported"]["conditions"]["frozen_support_frozen_query"]
    oracle = views["supported"]["conditions"]["oracle_support_oracle_query"]
    gates = lock["gates"]["integration"]
    checks = {
        "oracle_ceiling_reproduced": oracle["episode_metrics"]["episode_macro_transition_set_exact_match"] == 1.0,
        "supported_end_to_end_episode_macro": (
            primary["episode_metrics"]["episode_macro_transition_set_exact_match"]
            >= gates["minimumSupportedEndToEndEpisodeMacroTransitionSetExact"]
        ),
        "supported_empty_version_space": (
            primary["schema_recovery"]["empty_version_space_rate"]
            <= gates["maximumSupportedEmptyVersionSpaceRate"]
        ),
        "supported_target_retention": (
            primary["schema_recovery"]["target_retention_rate"]
            >= gates["minimumSupportedTargetRetentionAfterAllSupports"]
        ),
    }
    passed = all(checks.values())
    support_partial = views["supported"]["conditions"]["frozen_support_oracle_query"]
    query_partial = views["supported"]["conditions"]["oracle_support_frozen_query"]
    if passed:
        decision = "authorize_fresh_multi_mechanic_final_suite_design"
        interpretation = (
            "The frozen supported-language grounder composes successfully with exact schema induction. "
            "The next eligible step is to freeze a fresh multi-mechanic final-suite design; LoRA remains unauthorized."
        )
    elif support_partial["episode_metrics"]["episode_macro_transition_set_exact_match"] < query_partial["episode_metrics"]["episode_macro_transition_set_exact_match"]:
        decision = "support_grounding_bottleneck_no_lora"
        interpretation = (
            "Support grounding is the dominant bottleneck: its errors corrupt schema recovery more than query-only grounding errors. "
            "The next experiment should preserve multiple frozen grounding hypotheses without refitting the model."
        )
    else:
        decision = "query_grounding_bottleneck_no_lora"
        interpretation = (
            "Query grounding is at least as limiting as support grounding. The next experiment should isolate unresolved-value detection "
            "and polarity on queries without changing schema induction."
        )
    result = {
        "schema_version": 19,
        "experiment": lock["experiment"],
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "evaluation_number": 1,
        "grounding": grounding,
        "views": views,
        "checks": checks,
        "passed": passed,
        "decision": decision,
        "interpretation": interpretation,
        "empty_version_policy": lock["empty_version_policy"],
        "lora_authorized": False,
        "data_access": {
            "v17_head_artifacts_read": 1,
            "v17_records_read": 0,
            "v17_model_results_read": 0,
            "adapter_training_runs": 0,
            "new_linear_fits": 0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" for value in predictions))
    result["grounding_predictions"] = str(predictions_path.relative_to(PROJECT_ROOT))
    result["grounding_predictions_sha256"] = file_sha256(predictions_path)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (PROJECT_ROOT / "docs/v19-results.md").write_text(markdown(result))
    print(json.dumps({
        "checks": checks,
        "decision": decision,
        "passed": passed,
        "supported_grounding": grounding["supported"]["development"],
        "supported_conditions": {
            name: {
                "episode_metrics": value["episode_metrics"],
                "schema_recovery": {key: val for key, val in value["schema_recovery"].items() if key != "prefix"},
            }
            for name, value in views["supported"]["conditions"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
