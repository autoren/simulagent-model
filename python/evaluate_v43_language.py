#!/usr/bin/env python3
"""Run the single sealed V43 paired sequential-language development."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time
from typing import Any, Sequence

from v10_protocol import file_sha256
from v22_relational import canonical_json
from v22r2_grounding import PROJECT_ROOT
from evaluate_v42_sequential import aggregate as aggregate_v42
from evaluate_v42_sequential import evaluate_record as evaluate_v42_record
from evaluate_v42_sequential import version_space
from v42_stateful import execute_partial, mechanic_registry
from v43_language import compile_action_sequence, compile_state, evaluate_safety_challenge


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
        values = row["allowed_values"]
        if len(values) != 1:
            return []
        rows.append({"atom": row["atom"], "value": values[0]})
    return rows


def compile_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    agent = record["agent_input"]
    predicate = agent["predicate_ontology"]
    operator = agent["operator_ontology"]
    action = agent["action_ontology"]
    support_references = {row["id"]: row for row in record["reference"]["support_sequences"]}
    query_references = {row["id"]: row for row in record["reference"]["queries"]}
    clause_checks, graph_checks, command_checks, sequence_checks = [], [], [], []
    supports = []
    for public in agent["support_sequences"]:
        reference = support_references[public["id"]]
        initial = compile_state(public["initial_state_language"], public["entities"], predicate, operator)
        clause_checks.extend(_clause_checks(initial, reference["initial_state"]))
        graph_checks.append(initial["epistemic_state"] == reference["initial_state"]["epistemic_state"])
        actions = compile_action_sequence(public["action_language"], public["entities"], action)
        expected_actions = reference["actions"]["actions"]
        exact_actions = actions.get("status") == "ok" and actions.get("actions") == expected_actions
        sequence_checks.append(exact_actions)
        command_checks.extend([exact_actions] * len(expected_actions))
        observed = []
        for language_state, state_reference in zip(public["observed_step_state_language"], reference["observed_step_states"]):
            compiled = compile_state(language_state, public["entities"], predicate, operator)
            clause_checks.extend(_clause_checks(compiled, state_reference))
            graph_checks.append(compiled["epistemic_state"] == state_reference["epistemic_state"])
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
        clause_checks.extend(_clause_checks(initial, reference["initial_state"]))
        graph_checks.append(initial["epistemic_state"] == reference["initial_state"]["epistemic_state"])
        actions = compile_action_sequence(public["action_language"], public["entities"], action)
        expected_actions = reference["actions"]["actions"]
        exact_actions = actions.get("status") == "ok" and actions.get("actions") == expected_actions
        sequence_checks.append(exact_actions)
        command_checks.extend([exact_actions] * len(expected_actions))
        queries.append({
            "id": public["id"],
            "entities": public["entities"],
            "initial_state": initial["epistemic_state"],
            "actions": actions.get("actions", []),
            "structural_key": public["language_structural_key"],
            **{key: public[key] for key in (
                "sequence_length", "entity_count", "partial_initial_state",
                "order_counterfactual_group", "order_counterfactual_role", "order_effect",
            )},
        })
    safety_checks = [
        evaluate_safety_challenge(challenge, agent["entity_catalog"], predicate, operator, action)
        for challenge in agent["safety_challenges"]
    ]
    symbolic = {
        "id": record["id"],
        "schema_version": 43,
        "split": record["split"],
        "construction_family": record["construction_family"],
        "agent_input": {"support_sequences": supports, "queries": queries},
        "target": record["target"],
        "oracle_queries": record["oracle_queries"],
    }
    language_metrics = {
        "clause_checks": clause_checks,
        "graph_checks": graph_checks,
        "command_checks": command_checks,
        "sequence_checks": sequence_checks,
        "safety_checks": safety_checks,
    }
    return symbolic, language_metrics


def evaluate_record(record: dict[str, Any], registry: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    symbolic, language = compile_record(record)
    v42_metrics, v42_prediction = evaluate_v42_record(symbolic, registry)
    survivors, _ = version_space(symbolic, registry)
    survivor_programs = [row["program"] for row in survivors]
    targets = {row["id"]: row["target"] for row in record["oracle_queries"]}
    bag_rows = []
    for query in symbolic["agent_input"]["queries"]:
        canonical_actions = sorted(query["actions"], key=canonical_json)
        prediction = execute_partial(survivor_programs, query["entities"], query["initial_state"], canonical_actions)
        bag_rows.append({
            "group": query["order_counterfactual_group"],
            "role": query["order_counterfactual_role"],
            "order_effect": query["order_effect"],
            "predicted": prediction["possible_final_observations"],
            "target": targets[query["id"]]["possible_final_observations"],
        })
    groups = defaultdict(list)
    for row in bag_rows:
        if row["order_effect"]:
            groups[row["group"]].append(row)
    bag_checks = []
    for rows in groups.values():
        valid = len(rows) == 2 and {row["role"] for row in rows} == {"forward", "reversed"}
        bag_checks.append(
            valid
            and rows[0]["target"] != rows[1]["target"]
            and rows[0]["predicted"] != rows[1]["predicted"]
            and all(row["predicted"] == row["target"] for row in rows)
        )
    metrics = {"v42": v42_metrics, "language": language, "bag_order_checks": bag_checks}
    prediction = {
        **v42_prediction,
        "state_clause_exact_parse": rate(language["clause_checks"]),
        "state_graph_exact": rate(language["graph_checks"]),
        "action_command_exact_parse": rate(language["command_checks"]),
        "action_sequence_exact": rate(language["sequence_checks"]),
        "safety_abstention": rate(language["safety_checks"]),
        "bag_of_actions_order_counterfactual_accuracy": rate(bag_checks),
    }
    return metrics, prediction


def aggregate(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    v42 = aggregate_v42([row["v42"] for row in metrics])
    clauses = [value for row in metrics for value in row["language"]["clause_checks"]]
    graphs = [value for row in metrics for value in row["language"]["graph_checks"]]
    commands = [value for row in metrics for value in row["language"]["command_checks"]]
    sequences = [value for row in metrics for value in row["language"]["sequence_checks"]]
    safety = [value for row in metrics for value in row["language"]["safety_checks"]]
    bag = [value for row in metrics for value in row["bag_order_checks"]]
    return {
        "mechanics": v42["mechanics"],
        "queries": v42["queries"],
        "state_clauses": len(clauses),
        "state_clause_exact_parse": rate(clauses),
        "state_graphs": len(graphs),
        "state_graph_exact": rate(graphs),
        "action_commands": len(commands),
        "action_command_exact_parse": rate(commands),
        "action_sequences": len(sequences),
        "action_sequence_exact": rate(sequences),
        "safety_challenges": len(safety),
        "safety_abstention": rate(safety),
        "compiled_target_retention": v42["stateful_target_retention"],
        "compiled_schema_recovery": v42["stateful_schema_recovery"],
        "compiled_empty_version_space": v42["stateful_empty_version_space"],
        "compiled_next_state_exact": v42["stateful_next_state_exact"],
        "compiled_final_observation_exact": v42["stateful_final_observation_exact"],
        "compiled_complete_mechanic_exact": v42["stateful_complete_mechanic_exact"],
        "compiled_by_family_final_exact": v42["stateful_by_family_final_exact"],
        "compiled_by_sequence_length_final_exact": v42["stateful_by_sequence_length_final_exact"],
        "compiled_by_split_final_exact": v42["stateful_by_split_final_exact"],
        "compiled_partial_initial_final_exact": v42["stateful_partial_initial_final_exact"],
        "compiled_order_counterfactual_pairs": v42["order_counterfactual_pairs"],
        "compiled_order_counterfactual_accuracy": v42["order_counterfactual_accuracy"],
        "bag_of_actions_order_counterfactual_accuracy": rate(bag),
        "literal_language_lookup_final_exact": v42["literal_lookup_final_observation_exact"],
        "median_final_version_space": v42["median_final_version_space"],
    }


def qualification(metrics: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    checks = {
        "state_clause_exact_parse": metrics["state_clause_exact_parse"] >= gates["minimumStateClauseExactParse"],
        "state_graph_exact": metrics["state_graph_exact"] >= gates["minimumStateGraphExact"],
        "action_command_exact_parse": metrics["action_command_exact_parse"] >= gates["minimumActionCommandExactParse"],
        "action_sequence_exact": metrics["action_sequence_exact"] >= gates["minimumActionSequenceExact"],
        "safety_abstention": metrics["safety_abstention"] >= gates["minimumSafetyAbstention"],
        "compiled_target_retention": metrics["compiled_target_retention"] >= gates["minimumCompiledTargetRetention"],
        "compiled_schema_recovery": metrics["compiled_schema_recovery"] >= gates["minimumCompiledSchemaRecovery"],
        "compiled_empty_version_space": metrics["compiled_empty_version_space"] <= gates["maximumCompiledEmptyVersionSpace"],
        "compiled_next_state_exact": metrics["compiled_next_state_exact"] >= gates["minimumCompiledNextStateExact"],
        "compiled_final_observation_exact": metrics["compiled_final_observation_exact"] >= gates["minimumCompiledFinalObservationExact"],
        "compiled_every_family": min(metrics["compiled_by_family_final_exact"].values()) >= gates["minimumCompiledEveryFamilyExact"],
        "compiled_every_sequence_length": min(metrics["compiled_by_sequence_length_final_exact"].values()) >= gates["minimumCompiledEverySequenceLengthExact"],
        "compiled_order_counterfactual_accuracy": metrics["compiled_order_counterfactual_accuracy"] >= gates["minimumCompiledOrderCounterfactualAccuracy"],
        "bag_of_actions_inadequate": metrics["bag_of_actions_order_counterfactual_accuracy"] <= gates["maximumBagOfActionsOrderCounterfactualAccuracy"],
        "literal_language_lookup_inadequate": metrics["literal_language_lookup_final_exact"] <= gates["maximumLiteralLanguageLookupFinalExact"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v43-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v43-sequential-language-grounding/development")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "development-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V43 paired development already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V43 implementation changed: {path}")
    records = []
    for artifact in seal["corpora"].values():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"V43 sealed corpus changed: {artifact['path']}")
        records.extend(read(path))
    records.sort(key=lambda row: row["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 43,
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
    if qualified["passed"]:
        decision = "declared_sequential_interface_pass_preregister_deterministic_delay"
    elif not all(checks[key] for key in ("state_clause_exact_parse", "state_graph_exact", "action_command_exact_parse", "action_sequence_exact", "safety_abstention")):
        decision = "repair_declared_language_compiler_only"
    elif not all(checks[key] for key in ("compiled_target_retention", "compiled_schema_recovery", "compiled_next_state_exact", "compiled_final_observation_exact")):
        decision = "audit_translation_boundary_keep_v42_reasoner_frozen"
    else:
        decision = "redesign_inadequacy_control_before_claim"
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "mechanic-predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions))
    result = {
        "schema_version": 43,
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
            "v42_records_read_during_evaluation": 0,
            "language_model_forward_passes": 0,
            "adapter_training_runs": 0,
        },
        "authorization": {
            "preregister_deterministic_delayed_effects": qualified["passed"],
            "construct_delayed_effects_benchmark": False,
            "add_stochasticity": False,
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
