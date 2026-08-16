#!/usr/bin/env python3
"""Conditionally replay the selected three-seed V31 language system through V28 once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from audit_v22r2_grounding import read_jsonl_directory
from evaluate_v22r2_relational_grounding import condition_modes, grounding_summary, integration_condition, mean
from train_v31_lora_readout_mlx import load_seed_parameters
from v10_protocol import file_sha256
from v22_relational import canonical_json, enumerate_program_hypotheses
from v22r2_grounding import PROJECT_ROOT, validate_scene_prediction
from v28_marginal_map import compatibility_matrix_deduplicated, select_marginal_episode_map
from v31_integration import assemble_scene_prediction
from v31_structured_model import (
    AdaptedStructuredGrounder, StructuredPointerHead, features_from_hidden,
    prompt_tokens_and_entity_spans,
)


def evidence_row(scene: dict, evidence: dict) -> dict:
    return {
        "id": f"{scene['id']}|{evidence['id']}", "scene_id": scene["id"],
        "split": scene["split"],
        "agent_input": {
            "entities": scene["agent_input"]["entities"],
            "predicate_ontology": {}, "evidence_text": evidence["text"],
        },
    }


def output_values(outputs) -> dict[str, np.ndarray]:
    mx.eval(*outputs)
    keys = ("predicate", "argument1", "argument2", "truth")
    return {
        key: np.asarray(value[0], dtype=np.float32)
        for key, value in zip(keys, outputs, strict=True)
    }


def load_frozen_heads(trained: dict, config: dict) -> list[StructuredPointerHead]:
    heads = []
    for seed in config["training"]["seeds"]:
        head = StructuredPointerHead(
            config["model"]["hiddenSize"], config["sharedStructuredHead"]["width"],
            len(config["sharedStructuredHead"]["predicateClasses"]),
            len(config["sharedStructuredHead"]["truthClasses"]),
        )
        path = PROJECT_ROOT / trained["frozen_readout"]["seeds"][str(seed)]["parameters"]
        values = mx.load(path)
        if set(values) != {name for name, _ in tree_flatten(head.parameters())}:
            raise RuntimeError("V31 integration frozen-head keys changed")
        head.load_weights(list(values.items()), strict=True)
        head.eval()
        heads.append(head)
    return heads


def mean_scores(values: list[dict[str, np.ndarray]]) -> dict[str, list[float]]:
    return {
        key: np.mean(np.stack([row[key] for row in values]), axis=0).tolist()
        for key in values[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-lock", default="configs/v31-trained-systems-lock.json")
    parser.add_argument("--language-result", default="outputs/v31-signed-fact-adaptation/sealed-evaluation/result.json")
    parser.add_argument("--language-audit", default="outputs/v31-signed-fact-adaptation/post-result-audit.json")
    parser.add_argument("--output-dir", default="outputs/v31-signed-fact-adaptation/integration")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    trained_path = (PROJECT_ROOT / args.trained_lock).resolve()
    result_path = (PROJECT_ROOT / args.language_result).resolve()
    audit_path = (PROJECT_ROOT / args.language_audit).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "integration-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V31 V28 integration was already attempted")
    trained = json.loads(trained_path.read_text())
    protocol_path = PROJECT_ROOT / trained["protocol_lock"]
    protocol = json.loads(protocol_path.read_text())
    config = protocol["config_payload"]
    language_result = json.loads(result_path.read_text())
    language_audit = json.loads(audit_path.read_text())
    if protocol["limits"]["v28IntegrationReplays"] != 1:
        raise RuntimeError("V31 protocol does not authorize one integration replay")
    if not (
        language_result["passed"] and language_result["v28_integration_authorized"]
        and language_result["selected_system"] in ("frozen_readout", "lora_readout")
        and language_result["trained_system_lock_sha256"] == file_sha256(trained_path)
    ):
        raise RuntimeError("V31 language result does not authorize integration")
    if not (
        language_audit["passed"] and language_audit["decision"] == "accept_v31_language_result"
        and language_audit["result_sha256"] == file_sha256(result_path)
    ):
        raise RuntimeError("V31 post-result audit does not authorize integration")
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V31 locked implementation changed: {path}")
    v22_lock_path = PROJECT_ROOT / config["sourceV22r2Lock"]
    if file_sha256(v22_lock_path) != protocol["source"]["v22r2_lock_sha256"]:
        raise RuntimeError("V31 V22r2 source lock changed")
    original_lock = json.loads(v22_lock_path.read_text())
    dataset = PROJECT_ROOT / original_lock["source"]["dataset"]
    scenes = sorted([
        row for row in read_jsonl_directory(dataset / "scenes")
        if row["split"] == "grounding_evaluation"
    ], key=lambda row: row["id"])
    records = sorted([
        row for row in read_jsonl_directory(dataset / "records")
        if row["split"] == "grounding_evaluation"
    ], key=lambda row: row["id"])
    indexed_rows = []
    for scene in scenes:
        for evidence in scene["agent_input"]["evidence"]:
            indexed_rows.append((scene, evidence, evidence_row(scene, evidence)))
    if len(indexed_rows) != protocol["conditional_integration"]["evidence_clauses"]:
        raise RuntimeError("V31 integration evidence population differs from lock")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 31, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(protocol_path),
        "trained_system_lock_sha256": file_sha256(trained_path),
        "language_result_sha256": file_sha256(result_path),
        "selected_system": language_result["selected_system"], "status": "started_before_model_load",
    }, indent=2, sort_keys=True) + "\n")
    specification = config["model"]
    seeds = config["training"]["seeds"]
    per_record: list[list[dict[str, np.ndarray]]] = [[] for _ in indexed_rows]
    forward_passes = 0
    if language_result["selected_system"] == "frozen_readout":
        base, tokenizer, _ = load(
            specification["model"], revision=specification["revision"], return_config=True
        )
        base.eval()
        heads = load_frozen_heads(trained, config)
        maximum = max(config["construction"]["entityCounts"])
        for index, (_, _, row) in enumerate(indexed_rows, start=1):
            tokens, spans, _ = prompt_tokens_and_entity_spans(row, config, tokenizer)
            hidden = base.language_model.model(mx.array([tokens]))[0]
            clause, entities, mask = features_from_hidden(hidden, spans, maximum)
            for head in heads:
                per_record[index - 1].append(output_values(head(
                    clause[None, :], entities[None, :, :], mask[None, :]
                )))
            forward_passes += 1
            if args.progress_every and (index % args.progress_every == 0 or index == len(indexed_rows)):
                print(f"v31 frozen integration: {index}/{len(indexed_rows)} clauses", file=sys.stderr, flush=True)
            mx.clear_cache()
        del heads, base
    else:
        lora = config["systems"]["loraReadout"]
        for seed in seeds:
            base, tokenizer, _ = load(
                specification["model"], revision=specification["revision"], return_config=True
            )
            core = base.language_model.model
            core.freeze()
            mx.random.seed(seed)
            linear_to_lora_layers(core, lora["lastLayers"], {
                "rank": lora["rank"], "scale": lora["scale"],
                "dropout": lora["dropout"], "keys": lora["moduleKeys"],
            })
            mx.random.seed(seed)
            head = StructuredPointerHead(
                specification["hiddenSize"], config["sharedStructuredHead"]["width"],
                len(config["sharedStructuredHead"]["predicateClasses"]),
                len(config["sharedStructuredHead"]["truthClasses"]),
            )
            model = AdaptedStructuredGrounder(core, head, max(config["construction"]["entityCounts"]))
            load_seed_parameters(
                model, PROJECT_ROOT / trained["lora_readout"]["seeds"][str(seed)]["parameters"]
            )
            model.eval()
            for index, (_, _, row) in enumerate(indexed_rows, start=1):
                tokens, spans, _ = prompt_tokens_and_entity_spans(row, config, tokenizer)
                per_record[index - 1].append(output_values(model(mx.array([tokens]), spans)))
                forward_passes += 1
                mx.clear_cache()
            print(f"v31 LoRA integration seed {seed}: {len(indexed_rows)} clauses", file=sys.stderr, flush=True)
            del model, head, core, base
            mx.clear_cache()
    all_scores = []
    scores_by_scene: dict[str, list[dict]] = {}
    for (scene, evidence, _), seed_values in zip(indexed_rows, per_record, strict=True):
        if len(seed_values) != len(seeds):
            raise RuntimeError("V31 integration did not retain every registered seed")
        score = {
            "scene_id": scene["id"], "evidence_id": evidence["id"],
            "split": scene["split"], "role": scene["role"],
            "mean_logits": mean_scores(seed_values), "registered_seed_count": len(seed_values),
        }
        all_scores.append(score)
        scores_by_scene.setdefault(scene["id"], []).append(score)
    predictions = []
    for scene in scenes:
        prediction = assemble_scene_prediction(scene, scores_by_scene[scene["id"]], config)
        validate_scene_prediction(scene, prediction["rows"])
        predictions.append(prediction)
    grounding = grounding_summary(scenes, predictions)
    prediction_lookup = {row["scene_id"]: row for row in predictions}
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())
    integration = {}
    for condition in original_lock["integration_conditions"]:
        support_mode, query_mode = condition_modes(condition)
        integration[condition] = integration_condition(
            records, support_mode, query_mode, prediction_lookup,
            v22_config, original_lock["config_payload"],
        )
    observed = {
        row["id"]: row["observed_transition_code"]
        for record in records for row in record["agent_input"]["support_traces"]
    }
    episode_diagnostics = []
    for record in records:
        hypotheses = list(enumerate_program_hypotheses(record["agent_input"]["dsl_contract"]["outcome_bits"]))
        graph_rows, compatibility_rows = [], []
        for support in record["oracle_grounding"]["support"]:
            prediction = prediction_lookup[support["id"]]
            graphs = [{
                "log_score": 0.0, "graph_key": canonical_json(prediction["epistemic_state"]),
                "epistemic_state": prediction["epistemic_state"], "prediction_rows": prediction["rows"],
            }]
            compatibility, _ = compatibility_matrix_deduplicated(
                graphs, hypotheses, support, observed[support["id"]], v22_config,
                original_lock["config_payload"]["excessUnknownPolicy"]["maximumPredictedUnknownAtomsPerScene"],
            )
            graph_rows.append(graphs)
            compatibility_rows.append(compatibility)
        selection = select_marginal_episode_map(hypotheses, graph_rows, compatibility_rows)
        target_index = [row.key for row in hypotheses].index(record["target"]["program_key"])
        if selection is None:
            selected_target, target_rank, target_posterior = False, None, 0.0
            selected_key, finite_programs = None, 0
        else:
            posterior = selection["posterior"]
            ordered = sorted(range(len(hypotheses)), key=lambda index: (-float(posterior[index]), hypotheses[index].key))
            selected_target = selection["program_index"] == target_index
            target_rank = ordered.index(target_index) + 1
            target_posterior = float(posterior[target_index])
            selected_key, finite_programs = selection["program_key"], selection["finite_programs"]
        episode_diagnostics.append({
            "episode_id": record["id"], "selected_program_key": selected_key,
            "target_program_selected": selected_target, "target_program_rank": target_rank,
            "target_program_posterior": target_posterior, "finite_programs": finite_programs,
            "fallback": selection is None,
        })
    evaluation = grounding["by_split"]["grounding_evaluation"]
    oracle = integration["oracle_support_oracle_query"]
    support = integration["frozen_support_oracle_query"]
    query = integration["oracle_support_frozen_query"]
    frozen = integration["frozen_support_frozen_query"]
    target_top1 = mean([row["target_program_selected"] for row in episode_diagnostics])
    gates = config["integration"]["gates"]
    checks = {
        "oracle_oracle_exact": oracle["transition_set_exact_match"] >= gates["minimumOracleOracleExact"],
        "evaluation_support_exact_graph": evaluation["exact_support_graph"] >= gates["minimumEvaluationSupportExactGraph"],
        "frozen_support_oracle_query_exact": support["transition_set_exact_match"] >= gates["minimumFrozenSupportOracleQueryExact"],
        "oracle_support_frozen_query_exact": query["transition_set_exact_match"] >= gates["minimumOracleSupportFrozenQueryExact"],
        "frozen_frozen_exact": frozen["transition_set_exact_match"] >= gates["minimumFrozenFrozenExact"],
        "frozen_support_target_retention": support["target_retention_rate"] >= gates["minimumFrozenSupportTargetRetention"],
        "frozen_support_empty_version_space": support["empty_version_space_rate"] <= gates["maximumFrozenSupportEmptyVersionSpace"],
    }
    passed = all(checks.values())
    v28_result = json.loads((PROJECT_ROOT / config["sourceV28Result"]).read_text())
    v28_eval, v28_integration = v28_result["grounding"]["by_split"]["grounding_evaluation"], v28_result["integration"]
    reference = {
        "evaluation_support_exact_graph": v28_eval["exact_support_graph"],
        "frozen_support_oracle_query_exact": v28_integration["frozen_support_oracle_query"]["transition_set_exact_match"],
        "oracle_support_frozen_query_exact": v28_integration["oracle_support_frozen_query"]["transition_set_exact_match"],
        "frozen_frozen_exact": v28_integration["frozen_support_frozen_query"]["transition_set_exact_match"],
        "target_program_top1": v28_result["marginal_search"]["evaluation_target_program_selection_rate"],
        "target_retention": v28_integration["frozen_support_oracle_query"]["target_retention_rate"],
        "empty_version_space": v28_integration["frozen_support_oracle_query"]["empty_version_space_rate"],
    }
    current = {
        "evaluation_support_exact_graph": evaluation["exact_support_graph"],
        "frozen_support_oracle_query_exact": support["transition_set_exact_match"],
        "oracle_support_frozen_query_exact": query["transition_set_exact_match"],
        "frozen_frozen_exact": frozen["transition_set_exact_match"],
        "target_program_top1": target_top1, "target_retention": support["target_retention_rate"],
        "empty_version_space": support["empty_version_space_rate"],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    def write_jsonl(path: Path, rows: list[dict]):
        path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    scores_path = output_dir / "ensemble-signed-fact-scores.jsonl"
    predictions_path = output_dir / "grounding-predictions.jsonl"
    diagnostics_path = output_dir / "episode-marginal-diagnostics.jsonl"
    write_jsonl(scores_path, all_scores)
    write_jsonl(predictions_path, predictions)
    write_jsonl(diagnostics_path, episode_diagnostics)
    result = {
        "schema_version": 31, "experiment": "v31_conditional_selected_system_v28_replay",
        "protocol_lock_sha256": file_sha256(protocol_path),
        "trained_system_lock_sha256": file_sha256(trained_path),
        "language_result_sha256": file_sha256(result_path), "integration_replay_number": 1,
        "selected_system": language_result["selected_system"], "registered_seed_count": len(seeds),
        "grounding": grounding, "integration": integration,
        "marginal_search": {
            "evaluation_target_program_selection_rate": target_top1,
            "median_target_program_rank": float(np.median([row["target_program_rank"] for row in episode_diagnostics if row["target_program_rank"] is not None])),
            "episode_fallback_rate": mean([row["fallback"] for row in episode_diagnostics]),
        },
        "v28_reference": reference, "current_metrics": current,
        "deltas": {key: current[key] - reference[key] for key in current},
        "checks": checks, "passed": passed,
        "decision": "selected_language_v28_replay_pass" if passed else "language_pass_but_v28_replay_insufficient",
        "scores": str(scores_path.relative_to(PROJECT_ROOT)), "scores_sha256": file_sha256(scores_path),
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "episode_diagnostics": str(diagnostics_path.relative_to(PROJECT_ROOT)),
        "episode_diagnostics_sha256": file_sha256(diagnostics_path),
        "data_access": {
            "model_forward_passes": forward_passes, "registered_seeds_ensembled": len(seeds),
            "seed_selections": 0, "v28_integration_replays": 1,
            "checkpoint_selections": 0, "hyperparameter_selections": 0,
        },
    }
    integration_path = output_dir / "result.json"
    integration_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "result_sha256": file_sha256(integration_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
