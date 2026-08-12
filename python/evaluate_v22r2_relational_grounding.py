#!/usr/bin/env python3
"""Fit the two locked V22r2 heads once and run the four-way integration."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

from audit_v22r2_grounding import read_jsonl_directory
from run_v22_oracle_baselines import outcome_vocabulary
from v10_protocol import file_sha256
from v22_relational import enumerate_program_hypotheses, execute_partial, rows_to_epistemic
from v22r2_grounding import PROJECT_ROOT, predicted_epistemic_rows, validate_scene_prediction


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def pair_features(evidence: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    evidence_rows = np.broadcast_to(evidence, candidates.shape)
    return np.concatenate(
        (np.abs(evidence_rows - candidates), evidence_rows * candidates), axis=1
    ).astype(np.float32)


def feature_lookup(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    candidates = {
        str(identifier): arrays["candidate_features"][index]
        for index, identifier in enumerate(arrays["candidate_ids"].tolist())
    }
    evidence = {
        str(identifier): arrays["evidence_features"][index]
        for index, identifier in enumerate(arrays["evidence_ids"].tolist())
    }
    return candidates, evidence


def deterministic_negatives(
    candidate_ids: Sequence[str], positive: str, count: int, token: str,
) -> list[str]:
    choices = [value for value in candidate_ids if value != positive]
    choices.sort(key=lambda value: hashlib.sha256(f"{token}|{value}".encode()).hexdigest())
    if len(choices) < count:
        raise ValueError("A V22r2 scene has too few atom candidates for negative sampling")
    return choices[:count]


def build_training_arrays(
    scenes: Sequence[dict[str, Any]], candidate_lookup: dict[str, np.ndarray],
    evidence_lookup: dict[str, np.ndarray], config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_rows = []
    pair_targets = []
    truth_rows = []
    truth_targets = []
    negatives = config["heads"]["negativeSampling"]["negativesPerPositive"]
    for scene in scenes:
        if scene["split"] != config["heads"]["fitSplit"]:
            continue
        candidate_ids = [row["id"] for row in scene["agent_input"]["atom_candidates"]]
        for target in scene["target"]["atom_groundings"]:
            evidence = evidence_lookup[target["evidence_id"]]
            positive = target["candidate_id"]
            chosen = [positive, *deterministic_negatives(
                candidate_ids, positive, negatives,
                f"{config['seed']}|{scene['id']}|{target['evidence_id']}",
            )]
            for candidate_id in chosen:
                pair_rows.append(pair_features(
                    evidence[None, :], candidate_lookup[candidate_id][None, :]
                )[0])
                pair_targets.append(candidate_id == positive)
            truth_rows.append(evidence)
            truth_targets.append(target["truth_label"])
    return (
        np.stack(pair_rows).astype(np.float32),
        np.asarray(pair_targets, dtype=np.uint8),
        np.stack(truth_rows).astype(np.float32),
        np.asarray(truth_targets),
    )


def fit_heads(
    scenes: Sequence[dict[str, Any]], arrays: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[LogisticRegression, LogisticRegression, dict[str, Any]]:
    candidates, evidence = feature_lookup(arrays)
    pair_x, pair_y, truth_x, truth_y = build_training_arrays(
        scenes, candidates, evidence, config
    )
    atom_spec = config["heads"]["atomMatching"]
    truth_spec = config["heads"]["truthStatus"]
    atom_head = LogisticRegression(
        C=atom_spec["C"], class_weight=atom_spec["classWeight"],
        solver=atom_spec["solver"], max_iter=atom_spec["maximumIterations"],
        random_state=config["seed"],
    ).fit(pair_x, pair_y)
    truth_head = LogisticRegression(
        C=truth_spec["C"], class_weight=truth_spec["classWeight"],
        solver=truth_spec["solver"], max_iter=truth_spec["maximumIterations"],
        random_state=config["seed"],
    ).fit(truth_x, truth_y)
    diagnostics = {
        "atom_matching_rows": len(pair_y),
        "atom_matching_positive_rate": float(np.mean(pair_y)),
        "truth_rows": len(truth_y),
        "truth_class_counts": dict(sorted(Counter(truth_y.tolist()).items())),
        "atom_matching_iterations": atom_head.n_iter_.tolist(),
        "truth_iterations": truth_head.n_iter_.tolist(),
    }
    return atom_head, truth_head, diagnostics


def predict_scenes(
    scenes: Sequence[dict[str, Any]], arrays: dict[str, np.ndarray],
    atom_head: LogisticRegression, truth_head: LogisticRegression,
) -> list[dict[str, Any]]:
    candidate_lookup, evidence_lookup = feature_lookup(arrays)
    positive_column = int(np.flatnonzero(atom_head.classes_ == 1)[0])
    predictions = []
    for scene in scenes:
        candidate_ids = [row["id"] for row in scene["agent_input"]["atom_candidates"]]
        evidence_ids = [row["id"] for row in scene["agent_input"]["evidence"]]
        candidate_x = np.stack([candidate_lookup[value] for value in candidate_ids])
        evidence_x = np.stack([evidence_lookup[value] for value in evidence_ids])
        scores = np.empty((len(evidence_ids), len(candidate_ids)), dtype=np.float32)
        for evidence_index, vector in enumerate(evidence_x):
            scores[evidence_index] = atom_head.predict_proba(
                pair_features(vector[None, :], candidate_x)
            )[:, positive_column]
        evidence_indices, candidate_indices = linear_sum_assignment(-scores)
        assignment = dict(zip(evidence_indices.tolist(), candidate_indices.tolist(), strict=True))
        truth = truth_head.predict(evidence_x).tolist()
        rows = [
            {
                "evidence_id": evidence_id,
                "candidate_id": candidate_ids[assignment[index]],
                "truth_label": truth[index],
                "assignment_score": float(scores[index, assignment[index]]),
            }
            for index, evidence_id in enumerate(evidence_ids)
        ]
        validate_scene_prediction(scene, rows)
        predictions.append({
            "scene_id": scene["id"],
            "episode_id": scene["episode_id"],
            "split": scene["split"],
            "role": scene["role"],
            "rows": rows,
            "epistemic_state": predicted_epistemic_rows(scene, rows),
        })
    return predictions


def mean(values: Sequence[bool | float]) -> float:
    return float(np.mean(values)) if values else 0.0


def grounding_summary(
    scenes: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    predicted = {row["scene_id"]: row for row in predictions}
    atom_rows = []
    scene_rows = []
    support_by_episode: dict[str, list[bool]] = defaultdict(list)
    for scene in scenes:
        prediction = predicted[scene["id"]]
        target_by_evidence = {
            row["evidence_id"]: row for row in scene["target"]["atom_groundings"]
        }
        assignment_correct = []
        truth_correct = []
        for row in prediction["rows"]:
            target = target_by_evidence[row["evidence_id"]]
            atom_ok = row["candidate_id"] == target["candidate_id"]
            truth_ok = row["truth_label"] == target["truth_label"]
            assignment_correct.append(atom_ok)
            truth_correct.append(truth_ok)
            atom_rows.append({
                "split": scene["split"], "role": scene["role"],
                "predicate_kind": target["predicate_kind"],
                "truth_label": target["truth_label"],
                "semantic_operator": target["semantic_operator"],
                "relation_orientation": target["relation_orientation"],
                "surface_bank": target["surface_bank"],
                "assignment_correct": atom_ok, "truth_correct": truth_ok,
                "atom_value_correct": atom_ok and truth_ok,
            })
        exact = all(assignment_correct) and all(truth_correct)
        scene_rows.append({
            "scene_id": scene["id"], "episode_id": scene["episode_id"],
            "split": scene["split"], "role": scene["role"], "exact": exact,
        })
        if scene["role"] == "support":
            support_by_episode[scene["episode_id"]].append(exact)

    def subset_summary(split: str) -> dict[str, Any]:
        atoms = [row for row in atom_rows if row["split"] == split]
        selected_scenes = [row for row in scene_rows if row["split"] == split]
        relations = [row for row in atoms if row["predicate_kind"] == "relation"]
        supports = [row for row in selected_scenes if row["role"] == "support"]
        queries = [row for row in selected_scenes if row["role"] == "query"]
        episode_ids = {row["episode_id"] for row in selected_scenes}
        return {
            "scenes": len(selected_scenes), "atoms": len(atoms),
            "atom_assignment_accuracy": mean([row["assignment_correct"] for row in atoms]),
            "relation_argument_order_accuracy": mean([
                row["assignment_correct"] for row in relations
            ]),
            "truth_status_accuracy": mean([row["truth_correct"] for row in atoms]),
            "atom_value_accuracy": mean([row["atom_value_correct"] for row in atoms]),
            "exact_scene_graph": mean([row["exact"] for row in selected_scenes]),
            "exact_support_graph": mean([row["exact"] for row in supports]),
            "exact_query_graph": mean([row["exact"] for row in queries]),
            "episodes_with_all_support_graphs_exact": mean([
                all(support_by_episode[episode]) for episode in episode_ids
            ]),
            "by_surface_bank": {
                bank: {
                    "atoms": len(rows),
                    "atom_value_accuracy": mean([row["atom_value_correct"] for row in rows]),
                }
                for bank in sorted({row["surface_bank"] for row in atoms})
                for rows in [[row for row in atoms if row["surface_bank"] == bank]]
            },
            "by_truth_status": {
                label: {
                    "atoms": len(rows),
                    "truth_status_accuracy": mean([row["truth_correct"] for row in rows]),
                }
                for label in ("false", "true", "unknown")
                for rows in [[row for row in atoms if row["truth_label"] == label]]
            },
        }

    return {
        "by_split": {
            split: subset_summary(split)
            for split in ("grounding_fit", "grounding_calibration", "grounding_evaluation")
        }
    }


def support_version_space(
    record: dict[str, Any], support_mode: str,
    prediction_lookup: dict[str, dict[str, Any]], v22_config: dict[str, Any],
    maximum_unknown: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    bits = record["agent_input"]["dsl_contract"]["outcome_bits"]
    current = list(enumerate_program_hypotheses(bits))
    target_key = record["target"]["program_key"]
    prefixes = []
    public_outcomes = {
        row["id"]: row["observed_transition_code"]
        for row in record["agent_input"]["support_traces"]
    }
    for support in record["oracle_grounding"]["support"]:
        rows = (
            support["epistemic_state"] if support_mode == "oracle"
            else prediction_lookup[support["id"]]["epistemic_state"]
        )
        state = rows_to_epistemic(rows)
        unknown = sum(len(values) == 2 for values in state.values())
        if unknown > maximum_unknown:
            current = []
        else:
            observed = public_outcomes[support["id"]]
            current = [
                hypothesis for hypothesis in current
                if observed in execute_partial(
                    [hypothesis.program], v22_config, support["entities"], state,
                    support["action_binding"], maximum_unknown,
                )["possible_transition_codes"]
            ]
        prefixes.append({
            "prefix": len(prefixes) + 1,
            "target_retained": any(row.key == target_key for row in current),
            "empty": not current,
            "version_space": len(current),
        })
    return current, prefixes


def integration_condition(
    records: Sequence[dict[str, Any]], support_mode: str, query_mode: str,
    prediction_lookup: dict[str, dict[str, Any]], v22_config: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    query_rows = []
    episode_rows = []
    prefix_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    maximum_unknown = config["excessUnknownPolicy"]["maximumPredictedUnknownAtomsPerScene"]
    for record in records:
        survivors, prefixes = support_version_space(
            record, support_mode, prediction_lookup, v22_config, maximum_unknown
        )
        for prefix in prefixes:
            prefix_rows[prefix["prefix"]].append(prefix)
        target_retained = any(row.key == record["target"]["program_key"] for row in survivors)
        episode_exact = []
        for query in record["oracle_grounding"]["queries"]:
            rows = (
                query["epistemic_state"] if query_mode == "oracle"
                else prediction_lookup[query["id"]]["epistemic_state"]
            )
            state = rows_to_epistemic(rows)
            unknown = sum(len(values) == 2 for values in state.values())
            if not survivors or unknown > maximum_unknown:
                prediction = {
                    "possible_transition_codes": outcome_vocabulary(
                        record["agent_input"]["dsl_contract"]["outcome_bits"]
                    ),
                    "identifiable": False,
                }
            else:
                prediction = execute_partial(
                    [row.program for row in survivors], v22_config, query["entities"], state,
                    query["action_binding"], maximum_unknown,
                )
            exact = prediction["possible_transition_codes"] == query["possible_transition_codes"]
            episode_exact.append(exact)
            query_rows.append({
                "episode_id": record["id"],
                "family": record["oracle_metadata"]["construction_family"],
                "axis": query["query_axis"], "entity_count": query["entity_count"],
                "unknown_effect": query["unknown_effect"], "exact": exact,
                "target_identifiable": query["identifiable"],
                "predicted_identifiable": prediction["identifiable"],
            })
        episode_rows.append({
            "episode_id": record["id"], "exact": all(episode_exact),
            "query_accuracy": mean(episode_exact), "target_retained": target_retained,
            "empty": not survivors, "version_space": len(survivors),
        })

    def grouped(field: str) -> dict[str, Any]:
        return {
            str(value): {
                "queries": len(rows),
                "transition_set_exact_match": mean([row["exact"] for row in rows]),
            }
            for value in sorted({row[field] for row in query_rows}, key=str)
            for rows in [[row for row in query_rows if row[field] == value]]
        }

    return {
        "support_mode": support_mode, "query_mode": query_mode,
        "queries": len(query_rows),
        "transition_set_exact_match": mean([row["exact"] for row in query_rows]),
        "episode_macro_transition_set_exact_match": mean([
            row["query_accuracy"] for row in episode_rows
        ]),
        "complete_episodes": sum(row["exact"] for row in episode_rows),
        "episodes": len(episode_rows),
        "target_retention_rate": mean([row["target_retained"] for row in episode_rows]),
        "empty_version_space_rate": mean([row["empty"] for row in episode_rows]),
        "median_version_space": float(np.median([row["version_space"] for row in episode_rows])),
        "prefix": {
            str(index): {
                "episodes": len(rows),
                "target_retention_rate": mean([row["target_retained"] for row in rows]),
                "empty_version_space_rate": mean([row["empty"] for row in rows]),
                "median_version_space": float(np.median([row["version_space"] for row in rows])),
            }
            for index, rows in sorted(prefix_rows.items())
        },
        "by_family": grouped("family"), "by_axis": grouped("axis"),
        "by_entity_count": grouped("entity_count"),
        "by_unknown_effect": grouped("unknown_effect"),
    }


def condition_modes(name: str) -> tuple[str, str]:
    return {
        "oracle_support_oracle_query": ("oracle", "oracle"),
        "frozen_support_oracle_query": ("frozen", "oracle"),
        "oracle_support_frozen_query": ("oracle", "frozen"),
        "frozen_support_frozen_query": ("frozen", "frozen"),
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v22r2-relational-grounding-lock.json")
    parser.add_argument("--features", default="outputs/v22r2-relational-grounding/features")
    parser.add_argument("--output-dir", default="outputs/v22r2-relational-grounding/evaluation")
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    feature_root = (PROJECT_ROOT / args.features).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V22r2 evaluation was already attempted")
    lock = json.loads(lock_path.read_text())
    if (
        lock["limits"]["atomMatchingHeadFits"] != 1
        or lock["limits"]["truthStatusHeadFits"] != 1
        or lock["limits"]["integrationEvaluations"] != 1
    ):
        raise RuntimeError("V22r2 lock does not authorize the registered one-shot evaluation")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V22r2 locked implementation changed: {path}")
    metadata_path = feature_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["protocol_lock_sha256"] != file_sha256(lock_path):
        raise RuntimeError("V22r2 features do not share the evaluation lock")
    feature_path = PROJECT_ROOT / metadata["feature_artifact"]
    if file_sha256(feature_path) != metadata["feature_artifact_sha256"]:
        raise RuntimeError("V22r2 frozen feature artifact changed")
    arrays = load_npz(feature_path)
    scenes = read_jsonl_directory(PROJECT_ROOT / lock["source"]["dataset"] / "scenes")
    scenes.sort(key=lambda row: row["id"])
    if arrays["scene_ids"].tolist() != [row["id"] for row in scenes]:
        raise RuntimeError("V22r2 feature and scene ordering differ")
    records = read_jsonl_directory(PROJECT_ROOT / lock["source"]["dataset"] / "records")
    evaluation_records = [row for row in records if row["split"] == "grounding_evaluation"]
    v22_config = json.loads((PROJECT_ROOT / lock["source"]["v22_config"]).read_text())

    attempt_path.write_text(json.dumps({
        "schema_version": "22r2", "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "status": "started_before_head_fitting",
    }, indent=2, sort_keys=True) + "\n")
    atom_head, truth_head, fit_diagnostics = fit_heads(scenes, arrays, lock["config_payload"])
    predictions = predict_scenes(scenes, arrays, atom_head, truth_head)
    grounding = grounding_summary(scenes, predictions)
    prediction_lookup = {row["scene_id"]: row for row in predictions}
    integration = {}
    for condition in lock["integration_conditions"]:
        support_mode, query_mode = condition_modes(condition)
        integration[condition] = integration_condition(
            evaluation_records, support_mode, query_mode, prediction_lookup,
            v22_config, lock["config_payload"],
        )

    gates = lock["gates"]["development"]
    fit = grounding["by_split"]["grounding_fit"]
    evaluation = grounding["by_split"]["grounding_evaluation"]
    oracle = integration["oracle_support_oracle_query"]
    primary = integration["frozen_support_frozen_query"]
    frozen_support = integration["frozen_support_oracle_query"]
    checks = {
        "oracle_oracle_transition_set_exact": (
            oracle["transition_set_exact_match"] >= gates["minimumOracleOracleTransitionSetExact"]
        ),
        "fit_atom_assignment": (
            fit["atom_assignment_accuracy"] >= gates["minimumFitAtomAssignmentAccuracy"]
        ),
        "evaluation_atom_assignment": (
            evaluation["atom_assignment_accuracy"] >= gates["minimumEvaluationAtomAssignmentAccuracy"]
        ),
        "evaluation_truth_status": (
            evaluation["truth_status_accuracy"] >= gates["minimumEvaluationTruthStatusAccuracy"]
        ),
        "evaluation_relation_orientation": (
            evaluation["relation_argument_order_accuracy"]
            >= gates["minimumEvaluationRelationOrientationAccuracy"]
        ),
        "evaluation_exact_scene_graph": (
            evaluation["exact_scene_graph"] >= gates["minimumEvaluationExactSceneGraph"]
        ),
        "frozen_frozen_transition_set_exact": (
            primary["transition_set_exact_match"] >= gates["minimumFrozenFrozenTransitionSetExact"]
        ),
        "frozen_support_target_retention": (
            frozen_support["target_retention_rate"] >= gates["minimumFrozenSupportTargetRetention"]
        ),
        "frozen_support_empty_version_space": (
            frozen_support["empty_version_space_rate"]
            <= gates["maximumFrozenSupportEmptyVersionSpaceRate"]
        ),
    }
    passed = all(checks.values())
    query_only = integration["oracle_support_frozen_query"]
    if passed:
        decision = "authorize_separate_relational_final_design"
        interpretation = (
            "The fixed hard grounder transfers to held-out surfaces and composes with the V22 "
            "schema inducer within the declared ontology. A separately frozen final design is eligible."
        )
    elif frozen_support["transition_set_exact_match"] < query_only["transition_set_exact_match"]:
        decision = "develop_probabilistic_support_interface_no_lora"
        interpretation = (
            "Support grounding is the larger downstream bottleneck. Preserve multiple support "
            "groundings in a separately registered development experiment; do not train model weights."
        )
    else:
        decision = "repair_relational_language_grounding_no_lora"
        interpretation = (
            "Held-out language grounding or query graph assembly is the larger bottleneck. Expand "
            "or factor the supported language interface before any final suite or weight adaptation."
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    heads_path = output_dir / "heads.npz"
    np.savez_compressed(
        heads_path,
        atom_classes=atom_head.classes_, atom_coef=atom_head.coef_.astype(np.float32),
        atom_intercept=atom_head.intercept_.astype(np.float32),
        truth_classes=truth_head.classes_, truth_coef=truth_head.coef_.astype(np.float32),
        truth_intercept=truth_head.intercept_.astype(np.float32),
    )
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in predictions
    ))
    result = {
        "schema_version": "22r2",
        "experiment": lock["experiment"],
        "protocol_lock": str(lock_path.relative_to(PROJECT_ROOT)),
        "protocol_lock_sha256": file_sha256(lock_path),
        "feature_artifact_sha256": metadata["feature_artifact_sha256"],
        "evaluation_number": 1,
        "fit_diagnostics": fit_diagnostics,
        "grounding": grounding,
        "integration": integration,
        "checks": checks, "passed": passed, "decision": decision,
        "interpretation": interpretation,
        "heads_artifact": str(heads_path.relative_to(PROJECT_ROOT)),
        "heads_artifact_sha256": file_sha256(heads_path),
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "lora_authorized": False, "final_suite_constructed": False,
        "data_access": {
            "atom_matching_head_fits": 1, "truth_status_head_fits": 1,
            "hyperparameter_selections": 0, "adapter_training_runs": 0,
            "neural_challenger_runs": 0, "v21_final_records_read": 0,
            "v21_final_model_results_read": 0,
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({
        "status": "completed", "result": str(result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(result_path),
    })
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
