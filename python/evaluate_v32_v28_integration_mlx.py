#!/usr/bin/env python3
"""Conditionally replay one absolute-passing V32 system through unchanged V28."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from audit_v22r2_grounding import read_jsonl_directory
from evaluate_v22r2_relational_grounding import condition_modes, grounding_summary, integration_condition, mean
from v10_protocol import file_sha256
from v22_relational import canonical_json, enumerate_program_hypotheses
from v22r2_grounding import PROJECT_ROOT, validate_scene_prediction
from v28_marginal_map import compatibility_matrix_deduplicated, select_marginal_episode_map
from v32_integration import assemble_scene_prediction
from v32_structured_model import features_from_hidden, make_head, prompt_tokens_and_entity_spans


def evidence_row(scene: dict, evidence: dict) -> dict:
    return {"id": f"{scene['id']}|{evidence['id']}", "scene_id": scene["id"], "split": scene["split"], "agent_input": {"entities": scene["agent_input"]["entities"], "predicate_ontology": {}, "evidence_text": evidence["text"]}}


def output_values(outputs) -> dict[str, np.ndarray]:
    mx.eval(*outputs)
    keys = ("predicate", "argument1", "argument2", "truth", "lexical_sign", "outer_operation")
    return {key: np.asarray(value[0], dtype=np.float32) for key, value in zip(keys, outputs, strict=True)}


def mean_scores(values):
    return {key: np.mean(np.stack([row[key] for row in values]), axis=0).tolist() for key in values[0]}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-lock", default="configs/v32-trained-systems-lock.json")
    parser.add_argument("--language-result", default="outputs/v32-factorized-semantics/sealed-evaluation/result.json")
    parser.add_argument("--language-audit", default="outputs/v32-factorized-semantics/post-result-audit.json")
    parser.add_argument("--output-dir", default="outputs/v32-factorized-semantics/integration")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    trained_path, result_path, audit_path, output_dir = map(lambda value: (PROJECT_ROOT / value).resolve(), (args.trained_lock, args.language_result, args.language_audit, args.output_dir))
    attempt_path = output_dir.parent / "integration-attempt.json"
    if output_dir.exists() or attempt_path.exists(): raise RuntimeError("V32 V28 integration was already attempted")
    trained, result, audit = map(lambda path: json.loads(path.read_text()), (trained_path, result_path, audit_path))
    protocol_path = PROJECT_ROOT / trained["protocol_lock"]
    protocol, config = json.loads(protocol_path.read_text()), json.loads(protocol_path.read_text())["config_payload"]
    selected = result["selected_system"]
    if protocol["limits"]["v28IntegrationReplays"] != 1 or not result["passed"] or not result["v28_integration_authorized"] or selected not in ("monolithic", "auxiliaryDirect", "factorizedCompiled") or result["trained_system_lock_sha256"] != file_sha256(trained_path):
        raise RuntimeError("V32 language result does not authorize integration")
    if not audit["passed"] or audit["decision"] != "accept_v32_result" or audit["result_sha256"] != file_sha256(result_path):
        raise RuntimeError("V32 post-result audit does not authorize integration")
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected: raise RuntimeError(f"V32 locked implementation changed: {path}")
    v22_path = PROJECT_ROOT / config["sourceV22r2Lock"]
    if file_sha256(v22_path) != protocol["source"]["sourceV22r2Lock_sha256"]: raise RuntimeError("V32 V22r2 source changed")
    original_lock = json.loads(v22_path.read_text())
    dataset = PROJECT_ROOT / original_lock["source"]["dataset"]
    scenes = sorted([row for row in read_jsonl_directory(dataset / "scenes") if row["split"] == "grounding_evaluation"], key=lambda row: row["id"])
    records = sorted([row for row in read_jsonl_directory(dataset / "records") if row["split"] == "grounding_evaluation"], key=lambda row: row["id"])
    indexed = [(scene, evidence, evidence_row(scene, evidence)) for scene in scenes for evidence in scene["agent_input"]["evidence"]]
    if len(indexed) != protocol["conditional_integration"]["evidence_clauses"]: raise RuntimeError("V32 integration population differs from lock")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({"schema_version": 32, "attempt_number": 1, "protocol_lock_sha256": file_sha256(protocol_path), "trained_system_lock_sha256": file_sha256(trained_path), "language_result_sha256": file_sha256(result_path), "selected_system": selected, "status": "started_before_model_load"}, indent=2, sort_keys=True) + "\n")
    specification = config["model"]
    base, tokenizer, _ = load(specification["model"], revision=specification["revision"], return_config=True)
    base.eval()
    artifact = "monolithic" if selected == "monolithic" else "joint_auxiliary"
    heads = []
    for seed in config["training"]["seeds"]:
        head = make_head(config)
        values = mx.load(PROJECT_ROOT / trained["systems"][artifact]["seeds"][str(seed)]["parameters"])
        head.load_weights(list(values.items()), strict=True); head.eval(); heads.append(head)
    maximum, per_record = max(config["construction"]["entityCounts"]), []
    for index, (_, _, row) in enumerate(indexed, start=1):
        tokens, spans, _ = prompt_tokens_and_entity_spans(row, config, tokenizer)
        hidden = base.language_model.model(mx.array([tokens]))[0]
        clause, entities, mask = features_from_hidden(hidden, spans, maximum)
        values = [output_values(head(clause[None, :], entities[None, :, :], mask[None, :])) for head in heads]
        per_record.append(values)
        if args.progress_every and (index % args.progress_every == 0 or index == len(indexed)): print(f"v32 integration: {index}/{len(indexed)} clauses", file=sys.stderr, flush=True)
        mx.clear_cache()
    del heads, base
    all_scores, by_scene = [], {}
    for (scene, evidence, _), values in zip(indexed, per_record, strict=True):
        score = {"scene_id": scene["id"], "evidence_id": evidence["id"], "split": scene["split"], "role": scene["role"], "mean_logits": mean_scores(values), "registered_seed_count": len(values)}
        all_scores.append(score); by_scene.setdefault(scene["id"], []).append(score)
    predictions = []
    for scene in scenes:
        prediction = assemble_scene_prediction(scene, by_scene[scene["id"]], selected, config)
        validate_scene_prediction(scene, prediction["rows"]); predictions.append(prediction)
    grounding = grounding_summary(scenes, predictions)
    prediction_lookup = {row["scene_id"]: row for row in predictions}
    v22_config = json.loads((PROJECT_ROOT / original_lock["source"]["v22_config"]).read_text())
    integration = {}
    for condition in original_lock["integration_conditions"]:
        support_mode, query_mode = condition_modes(condition)
        integration[condition] = integration_condition(records, support_mode, query_mode, prediction_lookup, v22_config, original_lock["config_payload"])
    observed = {row["id"]: row["observed_transition_code"] for record in records for row in record["agent_input"]["support_traces"]}
    diagnostics = []
    for record in records:
        hypotheses = list(enumerate_program_hypotheses(record["agent_input"]["dsl_contract"]["outcome_bits"]))
        graph_rows, matrices = [], []
        for support in record["oracle_grounding"]["support"]:
            prediction = prediction_lookup[support["id"]]
            graphs = [{"log_score": 0.0, "graph_key": canonical_json(prediction["epistemic_state"]), "epistemic_state": prediction["epistemic_state"], "prediction_rows": prediction["rows"]}]
            compatibility, _ = compatibility_matrix_deduplicated(graphs, hypotheses, support, observed[support["id"]], v22_config, original_lock["config_payload"]["excessUnknownPolicy"]["maximumPredictedUnknownAtomsPerScene"])
            graph_rows.append(graphs); matrices.append(compatibility)
        selection = select_marginal_episode_map(hypotheses, graph_rows, matrices)
        target_index = [row.key for row in hypotheses].index(record["target"]["program_key"])
        if selection is None:
            selected_target, rank, posterior, key, finite = False, None, 0.0, None, 0
        else:
            ordered = sorted(range(len(hypotheses)), key=lambda i: (-float(selection["posterior"][i]), hypotheses[i].key))
            selected_target, rank, posterior = selection["program_index"] == target_index, ordered.index(target_index) + 1, float(selection["posterior"][target_index])
            key, finite = selection["program_key"], selection["finite_programs"]
        diagnostics.append({"episode_id": record["id"], "selected_program_key": key, "target_program_selected": selected_target, "target_program_rank": rank, "target_program_posterior": posterior, "finite_programs": finite, "fallback": selection is None})
    evaluation, oracle = grounding["by_split"]["grounding_evaluation"], integration["oracle_support_oracle_query"]
    support, query, frozen = integration["frozen_support_oracle_query"], integration["oracle_support_frozen_query"], integration["frozen_support_frozen_query"]
    target_top1, gates = mean([row["target_program_selected"] for row in diagnostics]), config["integration"]["gates"]
    checks = {
        "oracle_oracle_exact": oracle["transition_set_exact_match"] >= gates["minimumOracleOracleExact"],
        "evaluation_support_exact_graph": evaluation["exact_support_graph"] >= gates["minimumEvaluationSupportExactGraph"],
        "frozen_support_oracle_query_exact": support["transition_set_exact_match"] >= gates["minimumFrozenSupportOracleQueryExact"],
        "oracle_support_frozen_query_exact": query["transition_set_exact_match"] >= gates["minimumOracleSupportFrozenQueryExact"],
        "frozen_frozen_exact": frozen["transition_set_exact_match"] >= gates["minimumFrozenFrozenExact"],
        "frozen_support_target_retention": support["target_retention_rate"] >= gates["minimumFrozenSupportTargetRetention"],
        "frozen_support_empty_version_space": support["empty_version_space_rate"] <= gates["maximumFrozenSupportEmptyVersionSpace"],
    }
    v28 = json.loads((PROJECT_ROOT / config["sourceV28Result"]).read_text())
    v28_eval, v28_integration = v28["grounding"]["by_split"]["grounding_evaluation"], v28["integration"]
    reference = {"evaluation_support_exact_graph": v28_eval["exact_support_graph"], "frozen_support_oracle_query_exact": v28_integration["frozen_support_oracle_query"]["transition_set_exact_match"], "oracle_support_frozen_query_exact": v28_integration["oracle_support_frozen_query"]["transition_set_exact_match"], "frozen_frozen_exact": v28_integration["frozen_support_frozen_query"]["transition_set_exact_match"], "target_program_top1": v28["marginal_search"]["evaluation_target_program_selection_rate"], "target_retention": v28_integration["frozen_support_oracle_query"]["target_retention_rate"], "empty_version_space": v28_integration["frozen_support_oracle_query"]["empty_version_space_rate"]}
    current = {"evaluation_support_exact_graph": evaluation["exact_support_graph"], "frozen_support_oracle_query_exact": support["transition_set_exact_match"], "oracle_support_frozen_query_exact": query["transition_set_exact_match"], "frozen_frozen_exact": frozen["transition_set_exact_match"], "target_program_top1": target_top1, "target_retention": support["target_retention_rate"], "empty_version_space": support["empty_version_space_rate"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    scores_path, predictions_path, diagnostics_path = output_dir / "ensemble-scores.jsonl", output_dir / "grounding-predictions.jsonl", output_dir / "episode-diagnostics.jsonl"
    write_jsonl(scores_path, all_scores); write_jsonl(predictions_path, predictions); write_jsonl(diagnostics_path, diagnostics)
    result_payload = {"schema_version": 32, "experiment": "v32_conditional_v28_replay", "protocol_lock_sha256": file_sha256(protocol_path), "trained_system_lock_sha256": file_sha256(trained_path), "language_result_sha256": file_sha256(result_path), "integration_replay_number": 1, "selected_system": selected, "registered_seed_count": len(config["training"]["seeds"]), "grounding": grounding, "integration": integration, "marginal_search": {"evaluation_target_program_selection_rate": target_top1, "median_target_program_rank": float(np.median([row["target_program_rank"] for row in diagnostics if row["target_program_rank"] is not None])), "episode_fallback_rate": mean([row["fallback"] for row in diagnostics])}, "v28_reference": reference, "current_metrics": current, "deltas": {key: current[key] - reference[key] for key in current}, "checks": checks, "passed": all(checks.values()), "decision": "selected_language_v28_replay_pass" if all(checks.values()) else "language_pass_but_v28_replay_insufficient", "scores": str(scores_path.relative_to(PROJECT_ROOT)), "scores_sha256": file_sha256(scores_path), "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)), "grounding_predictions_sha256": file_sha256(predictions_path), "episode_diagnostics": str(diagnostics_path.relative_to(PROJECT_ROOT)), "episode_diagnostics_sha256": file_sha256(diagnostics_path), "data_access": {"model_forward_passes": len(indexed), "registered_seeds_ensembled": 3, "seed_selections": 0, "v28_integration_replays": 1, "checkpoint_selections": 0, "hyperparameter_selections": 0}}
    integration_path = output_dir / "result.json"
    integration_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text()); attempt.update({"status": "completed", "result_sha256": file_sha256(integration_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result_payload, indent=2, sort_keys=True))


if __name__ == "__main__": main()
