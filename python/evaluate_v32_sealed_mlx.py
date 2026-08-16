#!/usr/bin/env python3
"""Open and score both sealed V32 evaluation strata exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load

from audit_v32_factorized_semantics import read_rows
from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v32_evaluation import family_bootstrap_delta, score_rows, summarize_seed, system_summary
from v32_structured_model import features_from_hidden, make_head, prompt_tokens_and_entity_spans, select_predictions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def load_head(path: Path, config: dict):
    head = make_head(config)
    values = mx.load(path)
    if set(values) != {name for name, _ in tree_flatten(head.parameters())}:
        raise RuntimeError("V32 trained head keys changed")
    head.load_weights(list(values.items()), strict=True)
    head.eval()
    mx.eval(head.parameters())
    return head


def predict(head, rows, clauses, entities, masks, config, decoding):
    result = []
    for start in range(0, len(rows), 128):
        outputs = head(mx.array(clauses[start:start + 128]), mx.array(entities[start:start + 128]), mx.array(masks[start:start + 128]))
        mx.eval(*outputs)
        result.extend(select_predictions(rows[start:start + 128], tuple(np.asarray(value, dtype=np.float32) for value in outputs), config, decoding))
    return result


def subset(records, predictions, split):
    ids = {row["id"] for row in records if row["split"] == split}
    return [row for row in records if row["id"] in ids], [row for row in predictions if row["id"] in ids]


def material_comparison(records, baseline, challenger, config):
    paired = family_bootstrap_delta(records, baseline, challenger, config)
    baseline_seed = [summarize_seed(records, rows, config, False) for rows in baseline.values()]
    challenger_seed = [summarize_seed(records, rows, config, False) for rows in challenger.values()]
    fact = float(np.mean([row["exact_signed_fact_accuracy"] for row in challenger_seed]) - np.mean([row["exact_signed_fact_accuracy"] for row in baseline_seed]))
    scene = float(np.mean([row["exact_scene_accuracy"] for row in challenger_seed]) - np.mean([row["exact_scene_accuracy"] for row in baseline_seed]))
    gates = config["gates"]["selectionMaterialAdvantage"]
    checks = {
        "exact_signed_fact_delta": fact >= gates["minimumMeanExactSignedFactDelta"],
        "exact_scene_delta": scene >= gates["minimumMeanExactSceneDelta"],
        "paired_family_bootstrap_lower_bound": paired["bootstrap_95_interval"][0] > gates["minimumPairedFamilyBootstrapLowerBound"],
    }
    return {"mean_exact_signed_fact_delta": fact, "mean_exact_scene_delta": scene, "paired_family": paired, "checks": checks, "material_advantage": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-lock", default="configs/v32-trained-systems-lock.json")
    parser.add_argument("--output-dir", default="outputs/v32-factorized-semantics/sealed-evaluation")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    trained_path, output_dir = (PROJECT_ROOT / args.trained_lock).resolve(), (PROJECT_ROOT / args.output_dir).resolve()
    attempt_path = output_dir.parent / "evaluation-attempt.json"
    if output_dir.exists() or attempt_path.exists():
        raise RuntimeError("V32 sealed evaluation was already attempted")
    trained = json.loads(trained_path.read_text())
    protocol_path = PROJECT_ROOT / trained["protocol_lock"]
    protocol, config = json.loads(protocol_path.read_text()), json.loads(protocol_path.read_text())["config_payload"]
    if file_sha256(protocol_path) != trained["protocol_lock_sha256"] or protocol["limits"]["sealedEvaluations"] != 1 or trained["proof"]["trained_systems"] != 6:
        raise RuntimeError("V32 trained lock does not authorize this evaluation")
    for path, expected in protocol["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V32 locked implementation changed: {path}")
    for name, expected in protocol["source"]["corpus_file_sha256"].items():
        if file_sha256(PROJECT_ROOT / protocol["source"]["corpus"] / name) != expected:
            raise RuntimeError(f"V32 corpus changed after lock: {name}")
    for entry in trained["systems"].values():
        if file_sha256(PROJECT_ROOT / entry["manifest"]) != entry["manifest_sha256"]:
            raise RuntimeError("V32 training manifest changed")
        for seed in entry["seeds"].values():
            if file_sha256(PROJECT_ROOT / seed["parameters"]) != seed["parameters_sha256"] or file_sha256(PROJECT_ROOT / seed["ledger"]) != seed["ledger_sha256"]:
                raise RuntimeError("V32 trained artifact changed")
    rows = sorted(read_rows(PROJECT_ROOT / protocol["source"]["corpus"], ("factor_evaluation_paraphrase", "factor_evaluation_composition")), key=lambda row: row["id"])
    if len(rows) != protocol["planned_evaluation"]["records"]:
        raise RuntimeError("V32 evaluation population differs from lock")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({"schema_version": 32, "attempt_number": 1, "trained_system_lock_sha256": file_sha256(trained_path), "status": "started_before_model_load", "evaluation_records": len(rows)}, indent=2, sort_keys=True) + "\n")
    output_dir.mkdir(parents=True, exist_ok=False)
    specification = config["model"]
    base, tokenizer, model_config = load(specification["model"], revision=specification["revision"], return_config=True)
    base.eval()
    text_config = model_config["text_config"]
    if text_config["num_hidden_layers"] != specification["totalLayers"] or text_config["hidden_size"] != specification["hiddenSize"]:
        raise RuntimeError("V32 evaluation model architecture differs from lock")
    maximum = max(config["construction"]["entityCounts"])
    clauses, entities, masks, lengths = [], [], [], []
    payload = hashlib.sha256()
    for index, row in enumerate(rows, start=1):
        tokens, spans, content = prompt_tokens_and_entity_spans(row, config, tokenizer)
        if len(tokens) > specification["maxSequenceLength"]:
            raise RuntimeError(f"V32 evaluation prompt exceeds maximum: {row['id']}")
        hidden = base.language_model.model(mx.array([tokens]))[0]
        clause, entity, mask = features_from_hidden(hidden, spans, maximum)
        mx.eval(clause, entity, mask)
        clauses.append(np.asarray(clause, dtype=np.float32)); entities.append(np.asarray(entity, dtype=np.float32)); masks.append(np.asarray(mask, dtype=bool))
        lengths.append(len(tokens)); payload.update(content.encode())
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            print(f"v32 sealed features: {index}/{len(rows)}", file=sys.stderr, flush=True)
        mx.clear_cache()
    clauses, entities, masks = np.stack(clauses), np.stack(entities), np.stack(masks)
    feature_path = output_dir / "evaluation-features.npz"
    np.savez_compressed(feature_path, record_ids=np.asarray([row["id"] for row in rows]), clause_features=clauses, entity_features=entities, entity_mask=masks)
    del base
    mx.clear_cache()
    prediction_sets: dict[str, dict[str, list[dict]]] = {"monolithic": {}, "auxiliaryDirect": {}, "factorizedCompiled": {}}
    seed_summaries = {key: {} for key in prediction_sets}
    mapping = {
        "monolithic": ("monolithic", "direct_truth_head"),
        "auxiliaryDirect": ("joint_auxiliary", "direct_truth_head"),
        "factorizedCompiled": ("joint_auxiliary", "fixed_registered_truth_compiler"),
    }
    for system, (artifact_name, decoding) in mapping.items():
        for seed in config["training"]["seeds"]:
            head = load_head(PROJECT_ROOT / trained["systems"][artifact_name]["seeds"][str(seed)]["parameters"], config)
            predictions = predict(head, rows, clauses, entities, masks, config, decoding)
            prediction_sets[system][str(seed)] = predictions
            seed_summaries[system][str(seed)] = summarize_seed(rows, predictions, config, True)
            write_jsonl(output_dir / f"{system}-seed-{seed}-predictions.jsonl", predictions)
            del head
            mx.clear_cache()
    systems = {name: system_summary(values, config) for name, values in seed_summaries.items()}
    comp_rows = [row for row in rows if row["split"] == "factor_evaluation_composition"]
    comp_predictions = {system: {seed: [row for row in values if row["split"] == "factor_evaluation_composition"] for seed, values in seeds.items()} for system, seeds in prediction_sets.items()}
    comp_seed_summaries = {system: {seed: summarize_seed(comp_rows, values, config, False) for seed, values in seeds.items()} for system, seeds in comp_predictions.items()}
    factor_vs_aux = family_bootstrap_delta(comp_rows, comp_predictions["auxiliaryDirect"], comp_predictions["factorizedCompiled"], config)
    aux_vs_mono = family_bootstrap_delta(comp_rows, comp_predictions["monolithic"], comp_predictions["auxiliaryDirect"], config)
    factor_values = list(comp_seed_summaries["factorizedCompiled"].values())
    factor_gates = config["gates"]["scientificFactorization"]
    factor_checks = {
        "mean_composition_lexical_sign_accuracy": bool(np.mean([row["lexical_sign_accuracy"] for row in factor_values]) >= factor_gates["minimumMeanCompositionLexicalSignAccuracy"]),
        "mean_composition_outer_operation_accuracy": bool(np.mean([row["outer_operation_accuracy"] for row in factor_values]) >= factor_gates["minimumMeanCompositionOuterOperationAccuracy"]),
        "mean_composition_compiled_truth_accuracy": bool(np.mean([row["truth_status_accuracy"] for row in factor_values]) >= factor_gates["minimumMeanCompositionCompiledTruthAccuracy"]),
        "minimum_seed_composition_compiled_truth_accuracy": bool(min(row["truth_status_accuracy"] for row in factor_values) >= factor_gates["minimumMinimumSeedCompositionCompiledTruthAccuracy"]),
        "factorized_minus_auxiliary_composition_exact_fact": bool(factor_vs_aux["mean_exact_signed_fact_delta"] >= factor_gates["minimumFactorizedMinusAuxiliaryCompositionExactFact"]),
        "paired_family_bootstrap_lower_bound": bool(factor_vs_aux["bootstrap_95_interval"][0] > factor_gates["minimumPairedFamilyBootstrapLowerBound"]),
        "oracle_compiler_accuracy": bool(1.0 == factor_gates["requiredOracleCompilerAccuracy"]),
    }
    intermediate_gates = config["gates"]["scientificIntermediateSupervision"]
    intermediate_checks = {
        "auxiliary_minus_monolithic_composition_exact_fact": bool(aux_vs_mono["mean_exact_signed_fact_delta"] >= intermediate_gates["minimumAuxiliaryMinusMonolithicCompositionExactFact"]),
        "paired_family_bootstrap_lower_bound": bool(aux_vs_mono["bootstrap_95_interval"][0] > intermediate_gates["minimumPairedFamilyBootstrapLowerBound"]),
    }
    order, selected, selection_trace = ["monolithic", "auxiliaryDirect", "factorizedCompiled"], None, []
    for candidate in order:
        if not systems[candidate]["passed"]:
            selection_trace.append({"candidate": candidate, "absolute_pass": False, "selected": False})
            continue
        if selected is None:
            selected = candidate
            selection_trace.append({"candidate": candidate, "absolute_pass": True, "selected": True, "reason": "first_simplest_absolute_pass"})
            continue
        comparison = material_comparison(rows, prediction_sets[selected], prediction_sets[candidate], config)
        replace = comparison["material_advantage"]
        selection_trace.append({"candidate": candidate, "absolute_pass": True, "selected": replace, "challenged": selected, "comparison": comparison})
        if replace: selected = candidate
    artifacts = {path.name: file_sha256(path) for path in sorted(output_dir.glob("*predictions.jsonl"))}
    result = {
        "schema_version": 32, "experiment": config["experiment"], "protocol_lock_sha256": file_sha256(protocol_path),
        "trained_system_lock_sha256": file_sha256(trained_path), "evaluation_number": 1,
        "systems": systems, "composition_seed_summaries": comp_seed_summaries,
        "scientific_factorization": {"checks": factor_checks, "passed": all(factor_checks.values()), "factorized_minus_auxiliary": factor_vs_aux},
        "scientific_intermediate_supervision": {"checks": intermediate_checks, "passed": all(intermediate_checks.values()), "auxiliary_minus_monolithic": aux_vs_mono},
        "selected_system": selected, "selection_trace": selection_trace, "passed": selected is not None,
        "decision": "absolute_language_pass_authorizes_one_v28_replay" if selected else "scientific_results_reported_but_no_absolute_language_pass_stop_no_v28",
        "v28_integration_authorized": selected is not None,
        "evaluation_features": str(feature_path.relative_to(PROJECT_ROOT)), "evaluation_features_sha256": file_sha256(feature_path),
        "prediction_artifacts": artifacts, "minimum_prompt_tokens": min(lengths), "maximum_prompt_tokens": max(lengths), "prompt_payload_sha256": payload.hexdigest(), "truncated_prompts": 0,
        "data_access": {"evaluation_records_read": len(rows), "frozen_feature_model_forward_passes": len(rows), "head_prediction_passes": len(rows) * 9, "sealed_evaluations": 1, "seed_selections": 0, "checkpoint_selections": 0, "hyperparameter_selections": 0, "v28_integration_replays": 0},
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    attempt = json.loads(attempt_path.read_text())
    attempt.update({"status": "completed", "result": str(result_path.relative_to(PROJECT_ROOT)), "result_sha256": file_sha256(result_path)})
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
