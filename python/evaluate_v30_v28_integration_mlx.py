#!/usr/bin/env python3
"""Conditionally run exactly one frozen signed-fact/V28 integration replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from audit_v22r2_grounding import read_jsonl_directory
from evaluate_v22r2_relational_grounding import (
    condition_modes, grounding_summary, integration_condition, mean,
)
from evaluate_v30_signed_fact_language_mlx import dequantized_label_rows, score_prompt
from extract_v10_features_mlx import chat_prompt
from v10_protocol import file_sha256
from v22_relational import canonical_json, enumerate_program_hypotheses
from v22r2_grounding import PROJECT_ROOT, validate_scene_prediction
from v28_marginal_map import compatibility_matrix_deduplicated, select_marginal_episode_map
from v30_integration import assemble_scene_prediction
from v30_language import LABEL_TOKENS, primary_field_prompt, select_option


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evidence_row(scene: dict, evidence: dict) -> dict:
    return {
        "id": f"{scene['id']}|{evidence['id']}",
        "scene_id": scene["id"], "split": scene["split"],
        "agent_input": {
            "entities": scene["agent_input"]["entities"],
            "predicate_ontology": {}, "evidence_text": evidence["text"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/v30-signed-fact-language-lock.json")
    parser.add_argument("--language-result", default="outputs/v30-signed-fact-language/evaluation/result.json")
    parser.add_argument("--language-audit", default="outputs/v30-signed-fact-language/post-result-audit.json")
    parser.add_argument("--output-dir", default="outputs/v30-signed-fact-language/integration")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    lock_path = (PROJECT_ROOT / args.lock).resolve()
    result_path = (PROJECT_ROOT / args.language_result).resolve()
    audit_path = (PROJECT_ROOT / args.language_audit).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "integration-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V30 V28 integration was already attempted")
    lock = json.loads(lock_path.read_text())
    config = lock["config_payload"]
    language_result = json.loads(result_path.read_text())
    language_audit = json.loads(audit_path.read_text())
    if lock["limits"]["v28IntegrationReplays"] != 1:
        raise RuntimeError("V30 lock does not authorize one conditional integration replay")
    if not (
        language_result["passed"]
        and language_result["decision"] == "signed_fact_pass_authorize_one_v28_reintegration"
        and language_result["v28_integration_authorized"]
        and language_result["protocol_lock_sha256"] == file_sha256(lock_path)
    ):
        raise RuntimeError("V30 language result does not authorize integration")
    if not (
        language_audit["passed"]
        and language_audit["decision"] == "accept_v30_language_result"
        and language_audit["result_sha256"] == file_sha256(result_path)
    ):
        raise RuntimeError("V30 post-result audit does not authorize integration")
    for path, expected in lock["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V30 locked implementation changed: {path}")
    v22r2_lock_path = PROJECT_ROOT / config["sourceV22r2Lock"]
    if file_sha256(v22r2_lock_path) != lock["source"]["v22r2_lock_sha256"]:
        raise RuntimeError("V30 source V22r2 lock changed")
    original_lock = json.loads(v22r2_lock_path.read_text())
    dataset = PROJECT_ROOT / original_lock["source"]["dataset"]
    scenes = [
        row for row in read_jsonl_directory(dataset / "scenes")
        if row["split"] == "grounding_evaluation"
    ]
    scenes.sort(key=lambda row: row["id"])
    records = [
        row for row in read_jsonl_directory(dataset / "records")
        if row["split"] == "grounding_evaluation"
    ]
    records.sort(key=lambda row: row["id"])
    expected_evidence = sum(len(scene["agent_input"]["evidence"]) for scene in scenes)
    if expected_evidence != lock["conditional_integration"]["planned_evidence_clauses"]:
        raise RuntimeError("V30 integration evidence population differs from lock")

    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": 30, "attempt_number": 1,
        "protocol_lock_sha256": file_sha256(lock_path),
        "language_result_sha256": file_sha256(result_path),
        "status": "started_before_model_load",
    }, indent=2, sort_keys=True) + "\n")
    specification = config["model"]
    model, tokenizer, model_config = load(
        specification["model"], revision=specification["revision"], return_config=True
    )
    model.eval()
    text_config = model_config["text_config"]
    if (
        text_config["num_hidden_layers"] != specification["totalLayers"]
        or text_config["hidden_size"] != specification["hiddenSize"]
    ):
        raise RuntimeError("V30 integration model architecture differs from lock")
    encoded = {
        token: tokenizer.encode(token, add_special_tokens=False) for token in LABEL_TOKENS
    }
    if any(len(values) != 1 for values in encoded.values()):
        raise RuntimeError(f"V30 integration labels are not single tokens: {encoded}")
    token_ids = [encoded[token][0] for token in LABEL_TOKENS]
    label_rows = dequantized_label_rows(model, token_ids)
    mx.eval(label_rows)

    all_scores = []
    predictions = []
    prompt_lengths = []
    evidence_count = 0
    for scene_index, scene in enumerate(scenes, start=1):
        scene_scores = []
        for evidence in scene["agent_input"]["evidence"]:
            row = evidence_row(scene, evidence)
            selected_fields = {}
            field_logits = {}
            field_option_rows = {}
            for field in config["methods"]["primary"]["fields"]:
                content, options = primary_field_prompt(row, field, config)
                logits, length, _ = score_prompt(
                    content, specification["primarySystemPrompt"], len(options), model,
                    tokenizer, label_rows, specification["maxSequenceLength"],
                )
                selected = select_option(logits, options)
                selected_fields[field] = selected["value"]
                field_logits[field] = {
                    option["token"]: logits[position]
                    for position, option in enumerate(options)
                }
                field_option_rows[field] = options
                prompt_lengths.append(length)
            score = {
                "scene_id": scene["id"], "evidence_id": evidence["id"],
                "split": scene["split"], "role": scene["role"],
                "selected_fields": selected_fields, "field_logits": field_logits,
                "field_options": field_option_rows,
            }
            scene_scores.append(score)
            all_scores.append(score)
            evidence_count += 1
        prediction = assemble_scene_prediction(scene, scene_scores, config)
        validate_scene_prediction(scene, prediction["rows"])
        predictions.append(prediction)
        if args.progress_every and (
            scene_index % args.progress_every == 0 or scene_index == len(scenes)
        ):
            print(
                f"v30 integration: scored {scene_index}/{len(scenes)} scenes "
                f"({evidence_count}/{expected_evidence} clauses)",
                file=sys.stderr, flush=True,
            )

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

    scene_lookup = {row["id"]: row for row in scenes}
    observed = {
        row["id"]: row["observed_transition_code"]
        for record in records for row in record["agent_input"]["support_traces"]
    }
    episode_diagnostics = []
    for record in records:
        hypotheses = list(enumerate_program_hypotheses(
            record["agent_input"]["dsl_contract"]["outcome_bits"]
        ))
        graph_rows = []
        compatibility_rows = []
        for support in record["oracle_grounding"]["support"]:
            prediction = prediction_lookup[support["id"]]
            graph = {
                "log_score": 0.0, "graph_key": canonical_json(prediction["epistemic_state"]),
                "epistemic_state": prediction["epistemic_state"],
                "prediction_rows": prediction["rows"],
            }
            graphs = [graph]
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
            ordered = sorted(
                range(len(hypotheses)),
                key=lambda index: (-float(posterior[index]), hypotheses[index].key),
            )
            selected_target = selection["program_index"] == target_index
            target_rank = ordered.index(target_index) + 1
            target_posterior = float(posterior[target_index])
            selected_key = selection["program_key"]
            finite_programs = selection["finite_programs"]
        episode_diagnostics.append({
            "episode_id": record["id"], "selected_program_key": selected_key,
            "target_program_selected": selected_target, "target_program_rank": target_rank,
            "target_program_posterior": target_posterior,
            "finite_programs": finite_programs, "fallback": selection is None,
        })

    evaluation = grounding["by_split"]["grounding_evaluation"]
    oracle = integration["oracle_support_oracle_query"]
    support = integration["frozen_support_oracle_query"]
    query = integration["oracle_support_frozen_query"]
    frozen = integration["frozen_support_frozen_query"]
    target_top1 = mean([row["target_program_selected"] for row in episode_diagnostics])
    gates = config["gates"]["integration"]
    checks = {
        "oracle_oracle_exact": oracle["transition_set_exact_match"] >= gates["minimumOracleOracleExact"],
        "evaluation_support_exact_graph": evaluation["exact_support_graph"] >= gates["minimumEvaluationSupportExactGraph"],
        "frozen_support_oracle_query_exact": support["transition_set_exact_match"] >= gates["minimumFrozenSupportOracleQueryExact"],
        "oracle_support_frozen_query_exact": query["transition_set_exact_match"] >= gates["minimumOracleSupportFrozenQueryExact"],
        "frozen_frozen_exact": frozen["transition_set_exact_match"] >= gates["minimumFrozenFrozenExact"],
        "target_program_top1": target_top1 >= gates["minimumTargetProgramTop1"],
        "frozen_support_target_retention": support["target_retention_rate"] >= gates["minimumFrozenSupportTargetRetention"],
        "frozen_support_empty_version_space": support["empty_version_space_rate"] <= gates["maximumFrozenSupportEmptyVersionSpace"],
    }
    passed = all(checks.values())
    v28_result = json.loads((PROJECT_ROOT / config["sourceV28Result"]).read_text())
    v28_evaluation = v28_result["grounding"]["by_split"]["grounding_evaluation"]
    v28_integration = v28_result["integration"]
    reference = {
        "evaluation_support_exact_graph": v28_evaluation["exact_support_graph"],
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
        "target_program_top1": target_top1,
        "target_retention": support["target_retention_rate"],
        "empty_version_space": support["empty_version_space_rate"],
    }
    deltas = {key: current[key] - reference[key] for key in current}
    decision = (
        "signed_fact_v28_integration_pass_current_scope_complete"
        if passed else "signed_fact_language_pass_but_v28_integration_insufficient"
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    scores_path = output_dir / "signed-fact-scores.jsonl"
    scores_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in all_scores
    ))
    predictions_path = output_dir / "grounding-predictions.jsonl"
    predictions_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions
    ))
    diagnostics_path = output_dir / "episode-marginal-diagnostics.jsonl"
    diagnostics_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in episode_diagnostics
    ))
    result = {
        "schema_version": 30, "experiment": "v30_conditional_signed_fact_v28_reintegration",
        "protocol_lock_sha256": file_sha256(lock_path),
        "language_result_sha256": file_sha256(result_path), "integration_replay_number": 1,
        "grounding": grounding, "integration": integration,
        "marginal_search": {
            "evaluation_target_program_selection_rate": target_top1,
            "median_target_program_rank": float(np.median([
                row["target_program_rank"] for row in episode_diagnostics
                if row["target_program_rank"] is not None
            ])),
            "episode_fallback_rate": mean([row["fallback"] for row in episode_diagnostics]),
        },
        "v28_reference": reference, "current_metrics": current, "deltas": deltas,
        "checks": checks, "passed": passed, "decision": decision,
        "scores": str(scores_path.relative_to(PROJECT_ROOT)),
        "scores_sha256": file_sha256(scores_path),
        "grounding_predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "grounding_predictions_sha256": file_sha256(predictions_path),
        "episode_diagnostics": str(diagnostics_path.relative_to(PROJECT_ROOT)),
        "episode_diagnostics_sha256": file_sha256(diagnostics_path),
        "data_access": {
            "model_forward_passes": expected_evidence * 4,
            "v28_integration_replays": 1, "head_fits": 0, "threshold_fits": 0,
            "hyperparameter_selections": 0, "adapter_training_runs": 0,
        },
    }
    integration_result_path = output_dir / "result.json"
    integration_result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({
        "status": "completed", "result": str(integration_result_path.relative_to(PROJECT_ROOT)),
        "result_sha256": file_sha256(integration_result_path),
    })
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
