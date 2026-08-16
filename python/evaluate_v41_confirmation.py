#!/usr/bin/env python3
"""Run the single sealed V41 end-to-end relational confirmation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import enumerate_program_hypotheses, execute_partial, rows_to_epistemic
from v41_interface import assemble_epistemic_graph, compile_language_scene


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate(values):
    return sum(values) / len(values) if values else 0.0


def outcome_vocabulary(bits):
    return ["transition_" + format(value, f"0{bits}b") for value in range(2 ** bits)]


def reference_by_id(record, role):
    return {row["id"]: row for row in record["language_reference"][role]}


def oracle_by_id(record, role):
    key = "support" if role == "support" else "queries"
    return {row["id"]: row for row in record["oracle_grounding"][key]}


def score_record(record, v22_config, v32_config):
    bits = record["oracle_metadata"]["outcome_bits"]
    family = record["construction_family"]
    target_key = record["target"]["program_key"]
    clause_rows = []
    scene_rows = []
    graph_lookup = {"support": {}, "queries": {}}
    public_scenes = {"support": record["agent_input"]["support_traces"], "queries": record["agent_input"]["queries"]}
    for role in ("support", "queries"):
        references = reference_by_id(record, role)
        oracles = oracle_by_id(record, role)
        for public in public_scenes[role]:
            reference = references[public["id"]]
            compiled = compile_language_scene(public, v32_config)
            assembled = assemble_epistemic_graph(public, compiled, reference["entity_alias_to_canonical"], v32_config)
            expected_clauses = {row["id"]: row for row in reference["clause_references"]}
            clause_exact = []
            for compiled_clause in compiled["clauses"]:
                expected = expected_clauses[compiled_clause["id"]]
                result = compiled_clause["compiler_result"]
                exact = result.get("status") == "ok" and result.get("parse") == expected["expected_parse"]
                clause_exact.append(exact)
                clause_rows.append({"role": role, "scene_id": public["id"], "clause_id": compiled_clause["id"], "exact": exact})
            target_rows = sorted(oracles[public["id"]]["epistemic_state"], key=lambda row: row["atom"])
            predicted_rows = sorted(assembled["epistemic_state"], key=lambda row: row["atom"])
            exact_graph = assembled["complete"] and predicted_rows == target_rows
            scene_rows.append({"role": role, "scene_id": public["id"], "exact_graph": exact_graph, "all_clauses_exact": all(clause_exact)})
            graph_lookup[role][public["id"]] = assembled
    hypotheses = list(enumerate_program_hypotheses(bits))
    public_outcomes = {row["id"]: row["observed_transition_code"] for row in record["agent_input"]["support_traces"]}
    prefixes = []
    for support in record["oracle_grounding"]["support"]:
        graph = graph_lookup["support"][support["id"]]
        if not graph["complete"]:
            hypotheses = []
        else:
            state = rows_to_epistemic(graph["epistemic_state"])
            observed = public_outcomes[support["id"]]
            hypotheses = [
                hypothesis for hypothesis in hypotheses
                if observed in execute_partial([hypothesis.program], v22_config, support["entities"], state, support["action_binding"], 4)["possible_transition_codes"]
            ]
        prefixes.append({"version_space": len(hypotheses), "target_retained": any(row.key == target_key for row in hypotheses)})
    target_retained = any(row.key == target_key for row in hypotheses)
    schema_recovered = len(hypotheses) == 1 and hypotheses[0].key == target_key
    query_rows = []
    empirical_support = {
        support["canonical_state_binding_hash"]: support["transition_code"]
        for support in record["oracle_grounding"]["support"]
    }
    for query in record["oracle_grounding"]["queries"]:
        graph = graph_lookup["queries"][query["id"]]
        if not graph["complete"] or not hypotheses:
            prediction = outcome_vocabulary(bits)
        else:
            prediction = execute_partial(
                [row.program for row in hypotheses], v22_config, query["entities"], rows_to_epistemic(graph["epistemic_state"]),
                query["action_binding"], 4,
            )["possible_transition_codes"]
        exact = prediction == query["possible_transition_codes"]
        empirical = [empirical_support[query["canonical_state_binding_hash"]]] if query["canonical_state_binding_hash"] in empirical_support and not query["unknown_atoms"] else outcome_vocabulary(bits)
        query_rows.append({
            "query_id": query["id"], "axis": query["query_axis"], "family": family,
            "exact": exact, "empirical_exact": empirical == query["possible_transition_codes"],
        })
    metrics = {
        "clause_rows": clause_rows, "scene_rows": scene_rows, "query_rows": query_rows,
        "target_retained": target_retained, "schema_recovered": schema_recovered,
        "empty_version_space": not hypotheses, "version_space": len(hypotheses),
        "complete_episode_exact": all(row["exact"] for row in query_rows),
        "prefixes": prefixes,
    }
    prediction = {
        "id": record["id"], "family": family, "target_retained": target_retained,
        "schema_recovered": schema_recovered, "version_space": len(hypotheses),
        "support_exact_graph": all(row["exact_graph"] for row in scene_rows if row["role"] == "support"),
        "query_exact_graph": all(row["exact_graph"] for row in scene_rows if row["role"] == "queries"),
        "transition_set_exact": rate([row["exact"] for row in query_rows]),
        "empirical_lookup_exact": rate([row["empirical_exact"] for row in query_rows]),
    }
    return metrics, prediction


def aggregate(all_metrics):
    clauses = [row for metrics in all_metrics for row in metrics["clause_rows"]]
    scenes = [row for metrics in all_metrics for row in metrics["scene_rows"]]
    queries = [row for metrics in all_metrics for row in metrics["query_rows"]]
    families = sorted({row["family"] for row in queries})
    axes = sorted({row["axis"] for row in queries})
    return {
        "mechanics": len(all_metrics),
        "clauses": len(clauses),
        "clause_exact_parse": rate([row["exact"] for row in clauses]),
        "support_exact_graph": rate([row["exact_graph"] for row in scenes if row["role"] == "support"]),
        "query_exact_graph": rate([row["exact_graph"] for row in scenes if row["role"] == "queries"]),
        "schema_recovery": rate([row["schema_recovered"] for row in all_metrics]),
        "target_retention": rate([row["target_retained"] for row in all_metrics]),
        "empty_version_space": rate([row["empty_version_space"] for row in all_metrics]),
        "transition_set_exact": rate([row["exact"] for row in queries]),
        "complete_episode_exact": rate([row["complete_episode_exact"] for row in all_metrics]),
        "by_family_transition_set_exact": {family: rate([row["exact"] for row in queries if row["family"] == family]) for family in families},
        "by_query_axis_transition_set_exact": {axis: rate([row["exact"] for row in queries if row["axis"] == axis]) for axis in axes},
        "empirical_lookup_transition_set_exact": rate([row["empirical_exact"] for row in queries]),
        "median_version_space": float(np.median([row["version_space"] for row in all_metrics])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v41-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v41-relational-mechanic-confirmation/evaluation")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "evaluation-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V41 confirmation already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V41 implementation changed: {path}")
    artifact = PROJECT_ROOT / seal["corpus"]["path"]
    if file_sha256(artifact) != seal["corpus"]["sha256"]:
        raise RuntimeError("V41 sealed corpus changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({"schema_version": 41, "status": "started", "confirmation_evaluation": 1, "corpus_seal_sha256": file_sha256(seal_path)}, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    records = read(artifact)
    all_metrics, predictions = [], []
    for record in records:
        metrics, prediction = score_record(record, implementation["v22_config_payload"], implementation["v32_config_payload"])
        all_metrics.append(metrics)
        predictions.append(prediction)
    metrics = aggregate(all_metrics)
    gates = implementation["config_payload"]["gates"]
    checks = {
        "clause_exact_parse": metrics["clause_exact_parse"] >= gates["minimumClauseExactParse"],
        "support_exact_graph": metrics["support_exact_graph"] >= gates["minimumSupportExactGraph"],
        "query_exact_graph": metrics["query_exact_graph"] >= gates["minimumQueryExactGraph"],
        "schema_recovery": metrics["schema_recovery"] >= gates["minimumSchemaRecovery"],
        "target_retention": metrics["target_retention"] >= gates["minimumTargetRetention"],
        "empty_version_space": metrics["empty_version_space"] <= gates["maximumEmptyVersionSpace"],
        "transition_set_exact": metrics["transition_set_exact"] >= gates["minimumTransitionSetExact"],
        "complete_episode_exact": metrics["complete_episode_exact"] >= gates["minimumCompleteEpisodeExact"],
        "every_family": min(metrics["by_family_transition_set_exact"].values()) >= gates["minimumEveryFamilyTransitionSetExact"],
        "every_query_axis": min(metrics["by_query_axis_transition_set_exact"].values()) >= gates["minimumEveryQueryAxisTransitionSetExact"],
        "empirical_lookup_imperfect": metrics["empirical_lookup_transition_set_exact"] <= gates["maximumEmpiricalLookupTransitionSetExact"],
    }
    passed = all(checks.values())
    if passed:
        decision = "accept_declared_scope_relational_mechanic_claim_begin_architecture_breaking_benchmark"
    elif not checks["clause_exact_parse"] or not checks["support_exact_graph"] or not checks["query_exact_graph"]:
        decision = "reject_end_to_end_language_transfer_no_v41_repair_or_repeat"
    elif not checks["empirical_lookup_imperfect"]:
        decision = "do_not_attribute_result_to_lifted_mechanic_induction"
    else:
        decision = "revisit_symbolic_hypothesis_coverage_in_new_experiment"
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "episode-predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions))
    result = {
        "schema_version": 41, "experiment": implementation["config_payload"]["experiment"],
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)), "corpus_seal_sha256": file_sha256(seal_path),
        "confirmation_evaluation_number": 1, "metrics": metrics,
        "qualification": {"passed": passed, "checks": checks}, "decision": decision,
        "predictions": str(prediction_path.relative_to(PROJECT_ROOT)), "predictions_sha256": file_sha256(prediction_path),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {"confirmation_evaluations": 1, "mechanics_scored": len(records), "selection_on_confirmation": 0, "model_forward_passes": 0, "v22r2_evaluation_records_read": 0, "v28_runs": 0, "adapter_training_runs": 0},
        "authorization": {"begin_architecture_breaking_benchmark": passed, "construct_architecture_breaking_benchmark": False, "v22r2_evaluation": False, "v28": False, "adapter_training": False, "change_compiler": False, "change_semantic_kernel": False},
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    state = json.loads(attempt.read_text()); state.update({"status": "completed", "result_sha256": file_sha256(result_path)})
    attempt.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
