#!/usr/bin/env python3
"""Run the single sealed V45 paired delayed-language development."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Sequence

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from evaluate_v44_delayed import aggregate as aggregate_v44
from evaluate_v44_delayed import evaluate_record as evaluate_v44_record
from v43_language import compile_state
from v43r1_measurement import graph_equal
from v44_delayed import mechanic_registry
from v45_language import compile_action_sequence, evaluate_safety_challenge


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clause_checks(compiled: dict[str, Any], reference: dict[str, Any]) -> list[bool]:
    expected = {row["id"]: row for row in reference["clauses"]}
    return [
        row["id"] in expected
        and row["compiler_result"].get("status") == "ok"
        and row["compiler_result"].get("parse") == expected[row["id"]]["expected_parse"]
        for row in compiled["clauses"]
    ]


def _observed_rows(compiled: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in compiled["epistemic_state"]:
        if len(row["allowed_values"]) != 1:
            return []
        rows.append({"atom": row["atom"], "value": row["allowed_values"][0]})
    return rows


def compile_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    agent = record["agent_input"]
    predicate, operator, action = agent["predicate_ontology"], agent["operator_ontology"], agent["action_ontology"]
    support_references = {row["id"]: row for row in record["reference"]["support_sequences"]}
    query_references = {row["id"]: row for row in record["reference"]["queries"]}
    clauses, graphs, bound_commands, wait_commands, sequences = [], [], [], [], []
    supports = []
    for public in agent["support_sequences"]:
        reference = support_references[public["id"]]
        initial = compile_state(public["initial_state_language"], public["entities"], predicate, operator)
        clauses.extend(_clause_checks(initial, reference["initial_state"]))
        graphs.append(graph_equal(initial["epistemic_state"], reference["initial_state"]["epistemic_state"]))
        actions = compile_action_sequence(public["action_language"], public["entities"], action)
        exact = actions.get("status") == "ok" and actions.get("actions") == reference["actions"]["actions"]
        sequences.append(exact)
        for kind in reference["actions"]["command_kinds"]:
            (wait_commands if kind == "wait" else bound_commands).append(exact)
        observed = []
        for language_state, state_reference in zip(public["observed_step_state_language"], reference["observed_step_states"]):
            compiled = compile_state(language_state, public["entities"], predicate, operator)
            clauses.extend(_clause_checks(compiled, state_reference))
            graphs.append(graph_equal(compiled["epistemic_state"], state_reference["epistemic_state"]))
            observed.append(_observed_rows(compiled))
        supports.append({
            "id": public["id"],
            "entities": public["entities"],
            "initial_state": initial["epistemic_state"],
            "actions": actions.get("actions", []),
            "observed_step_states": observed,
            "structural_key": public["language_structural_key"],
        })
    queries = []
    for public in agent["queries"]:
        reference = query_references[public["id"]]
        initial = compile_state(public["initial_state_language"], public["entities"], predicate, operator)
        clauses.extend(_clause_checks(initial, reference["initial_state"]))
        graphs.append(graph_equal(initial["epistemic_state"], reference["initial_state"]["epistemic_state"]))
        actions = compile_action_sequence(public["action_language"], public["entities"], action)
        exact = actions.get("status") == "ok" and actions.get("actions") == reference["actions"]["actions"]
        sequences.append(exact)
        for kind in reference["actions"]["command_kinds"]:
            (wait_commands if kind == "wait" else bound_commands).append(exact)
        queries.append({
            "id": public["id"],
            "entities": public["entities"],
            "initial_state": initial["epistemic_state"],
            "actions": actions.get("actions", []),
            "structural_key": public["language_structural_key"],
            **{key: public[key] for key in (
                "sequence_length", "entity_count", "partial_initial_state",
                "wait_counterfactual_group", "wait_counterfactual_role", "wait_placement_effect",
            )},
        })
    safety = [
        evaluate_safety_challenge(challenge, agent["entity_catalog"], predicate, operator, action)
        for challenge in agent["safety_challenges"]
    ]
    symbolic = {
        "id": record["id"],
        "schema_version": 45,
        "split": record["split"],
        "construction_family": record["construction_family"],
        "agent_input": {"support_sequences": supports, "queries": queries},
        "target": record["target"],
        "oracle_queries": record["oracle_queries"],
    }
    return symbolic, {
        "clause_checks": clauses,
        "graph_checks": graphs,
        "bound_command_checks": bound_commands,
        "wait_command_checks": wait_commands,
        "sequence_checks": sequences,
        "safety_checks": safety,
    }


def evaluate_record(record: dict[str, Any], registry: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    symbolic, language = compile_record(record)
    delayed, delayed_prediction = evaluate_v44_record(symbolic, registry)
    prediction = {
        **delayed_prediction,
        "state_clause_exact_parse": rate(language["clause_checks"]),
        "canonical_state_graph_exact": rate(language["graph_checks"]),
        "action_command_exact_parse": rate(language["bound_command_checks"]),
        "wait_command_exact_parse": rate(language["wait_command_checks"]),
        "action_sequence_exact": rate(language["sequence_checks"]),
        "safety_abstention": rate(language["safety_checks"]),
    }
    return {"delayed": delayed, "language": language}, prediction


def aggregate(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    delayed = aggregate_v44([row["delayed"] for row in metrics])
    clauses = [value for row in metrics for value in row["language"]["clause_checks"]]
    graphs = [value for row in metrics for value in row["language"]["graph_checks"]]
    bound = [value for row in metrics for value in row["language"]["bound_command_checks"]]
    waits = [value for row in metrics for value in row["language"]["wait_command_checks"]]
    sequences = [value for row in metrics for value in row["language"]["sequence_checks"]]
    safety = [value for row in metrics for value in row["language"]["safety_checks"]]
    return {
        "mechanics": delayed["mechanics"],
        "queries": delayed["queries"],
        "state_clauses": len(clauses),
        "state_clause_exact_parse": rate(clauses),
        "state_graphs": len(graphs),
        "canonical_state_graph_exact": rate(graphs),
        "bound_action_commands": len(bound),
        "action_command_exact_parse": rate(bound),
        "wait_commands": len(waits),
        "wait_command_exact_parse": rate(waits),
        "action_sequences": len(sequences),
        "action_sequence_exact": rate(sequences),
        "safety_challenges": len(safety),
        "safety_abstention": rate(safety),
        "compiled_target_retention": delayed["queued_target_retention"],
        "compiled_schema_recovery": delayed["queued_schema_recovery"],
        "compiled_empty_version_space": delayed["queued_empty_version_space"],
        "compiled_next_state_exact": delayed["queued_next_state_exact"],
        "compiled_final_observation_exact": delayed["queued_final_observation_exact"],
        "compiled_complete_mechanic_exact": delayed["queued_complete_mechanic_exact"],
        "compiled_by_family_final_exact": delayed["queued_by_family_final_exact"],
        "compiled_by_sequence_length_final_exact": delayed["queued_by_sequence_length_final_exact"],
        "compiled_by_split_final_exact": delayed["queued_by_split_final_exact"],
        "compiled_partial_initial_final_exact": delayed["queued_partial_initial_final_exact"],
        "compiled_wait_counterfactual_pairs": delayed["wait_counterfactual_pairs"],
        "compiled_wait_counterfactual_accuracy": delayed["wait_placement_counterfactual_accuracy"],
        "collapsed_delay_final_exact": delayed["collapsed_delay_final_exact"],
        "end_flush_final_exact": delayed["end_flush_final_exact"],
        "literal_language_lookup_final_exact": delayed["literal_lookup_final_exact"],
        "median_final_version_space": delayed["median_final_version_space"],
    }


def qualification(metrics: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    checks = {
        "state_clause_exact_parse": metrics["state_clause_exact_parse"] >= gates["minimumStateClauseExactParse"],
        "canonical_state_graph_exact": metrics["canonical_state_graph_exact"] >= gates["minimumCanonicalStateGraphExact"],
        "action_command_exact_parse": metrics["action_command_exact_parse"] >= gates["minimumActionCommandExactParse"],
        "wait_command_exact_parse": metrics["wait_command_exact_parse"] >= gates["minimumWaitCommandExactParse"],
        "action_sequence_exact": metrics["action_sequence_exact"] >= gates["minimumActionSequenceExact"],
        "safety_abstention": metrics["safety_abstention"] >= gates["minimumSafetyAbstention"],
        "compiled_target_retention": metrics["compiled_target_retention"] >= gates["minimumCompiledTargetRetention"],
        "compiled_schema_recovery": metrics["compiled_schema_recovery"] >= gates["minimumCompiledSchemaRecovery"],
        "compiled_empty_version_space": metrics["compiled_empty_version_space"] <= gates["maximumCompiledEmptyVersionSpace"],
        "compiled_next_state_exact": metrics["compiled_next_state_exact"] >= gates["minimumCompiledNextStateExact"],
        "compiled_final_observation_exact": metrics["compiled_final_observation_exact"] >= gates["minimumCompiledFinalObservationExact"],
        "compiled_every_family": min(metrics["compiled_by_family_final_exact"].values()) >= gates["minimumCompiledEveryFamilyExact"],
        "compiled_every_sequence_length": min(metrics["compiled_by_sequence_length_final_exact"].values()) >= gates["minimumCompiledEverySequenceLengthExact"],
        "compiled_wait_counterfactual_accuracy": metrics["compiled_wait_counterfactual_accuracy"] >= gates["minimumCompiledWaitCounterfactualAccuracy"],
        "collapsed_delay_inadequate": metrics["collapsed_delay_final_exact"] <= gates["maximumCollapsedDelayFinalExact"],
        "end_flush_inadequate": metrics["end_flush_final_exact"] <= gates["maximumEndFlushFinalExact"],
        "literal_language_lookup_inadequate": metrics["literal_language_lookup_final_exact"] <= gates["maximumLiteralLanguageLookupFinalExact"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v45-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v45-delayed-language-grounding/development")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "development-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V45 paired development already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V45 implementation changed: {path}")
    records = []
    for artifact in seal["corpora"].values():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"V45 sealed corpus changed: {artifact['path']}")
        records.extend(read(path))
    records.sort(key=lambda row: row["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 45,
        "status": "started",
        "paired_development_run": 1,
        "corpus_seal_sha256": file_sha256(seal_path),
    }, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    registry = mechanic_registry()
    all_metrics, predictions = [], []
    for record in records:
        metric, prediction = evaluate_record(record, registry)
        all_metrics.append(metric)
        predictions.append(prediction)
    metrics = aggregate(all_metrics)
    qualified = qualification(metrics, implementation["config_payload"]["gates"])
    checks = qualified["checks"]
    language_keys = (
        "state_clause_exact_parse", "canonical_state_graph_exact", "action_command_exact_parse",
        "wait_command_exact_parse", "action_sequence_exact", "safety_abstention",
    )
    execution_keys = (
        "compiled_target_retention", "compiled_schema_recovery", "compiled_next_state_exact",
        "compiled_final_observation_exact", "compiled_wait_counterfactual_accuracy",
    )
    if qualified["passed"]:
        decision = "declared_delayed_interface_pass_preregister_stochastic_foundation"
    elif not all(checks[key] for key in language_keys):
        decision = "repair_declared_language_compiler_only"
    elif not all(checks[key] for key in execution_keys):
        decision = "audit_translation_boundary_keep_v44_reasoner_frozen"
    else:
        decision = "redesign_inadequacy_control_before_claim"
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "mechanic-predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions))
    result = {
        "schema_version": 45,
        "experiment": implementation["config_payload"]["experiment"],
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "paired_development_run_number": 1,
        "metrics": metrics,
        "qualification": qualified,
        "decision": decision,
        "predictions": str(prediction_path.relative_to(PROJECT_ROOT)),
        "predictions_sha256": file_sha256(prediction_path),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "paired_development_runs": 1,
            "mechanics_scored": len(records),
            "selection_on_development_evaluation": 0,
            "v44_records_read_during_evaluation": 0,
            "model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
        "authorization": {
            "preregister_stochastic_transition_foundation": qualified["passed"],
            "construct_stochastic_transition_foundation": False,
            "active_intervention_selection": False,
            "open_ontology": False,
            "final_evaluation": False,
            "model_access": False,
        },
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    state = json.loads(attempt.read_text())
    state.update({"status": "completed", "result_sha256": file_sha256(result_path)})
    attempt.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
