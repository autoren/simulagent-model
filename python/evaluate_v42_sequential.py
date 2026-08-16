#!/usr/bin/env python3
"""Run the single sealed V42 oracle sequential development evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import time
from typing import Any, Sequence

import numpy as np

from v10_protocol import file_sha256
from v22r2_grounding import PROJECT_ROOT
from v22_relational import canonical_json
from v42_stateful import (
    compatible_worlds,
    execute_partial,
    execute_sequence,
    mechanic_registry,
    world_signature,
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def observed_signatures(support: dict[str, Any]) -> list[str]:
    return [
        canonical_json(sorted(step, key=lambda row: row["atom"]))
        for step in support["observed_step_states"]
    ]


def predicted_complete_signatures(program: dict[str, Any], support: dict[str, Any]) -> list[str]:
    worlds = compatible_worlds(support["initial_state"])
    if len(worlds) != 1:
        raise ValueError("V42 support states must be complete")
    return [world_signature(world) for world in execute_sequence(
        program, support["entities"], worlds[0], support["actions"]
    )]


def version_space(record: dict[str, Any], registry: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    survivors = list(registry)
    prefixes = []
    target_key = record["target"]["program_key"]
    for support in record["agent_input"]["support_sequences"]:
        observed = observed_signatures(support)
        survivors = [
            mechanic for mechanic in survivors
            if predicted_complete_signatures(mechanic["program"], support) == observed
        ]
        prefixes.append({
            "prefix": len(prefixes) + 1,
            "version_space": len(survivors),
            "target_retained": any(row["key"] == target_key for row in survivors),
        })
    return survivors, prefixes


def evaluate_record(record: dict[str, Any], registry: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_key = {row["key"]: row for row in registry}
    target_key = record["target"]["program_key"]
    target_program = by_key[target_key]["program"]
    supports = record["agent_input"]["support_sequences"]
    queries = record["agent_input"]["queries"]
    oracle_queries = {row["id"]: row["target"] for row in record["oracle_queries"]}
    support_validation = [
        predicted_complete_signatures(target_program, support) == observed_signatures(support)
        for support in supports
    ]
    query_validation = []
    for query in queries:
        reproduced = execute_partial([target_program], query["entities"], query["initial_state"], query["actions"])
        query_validation.append(reproduced == oracle_queries[query["id"]])
    survivors, prefixes = version_space(record, registry)
    survivor_programs = [row["program"] for row in survivors]
    support_lookup = {
        support["structural_key"]: observed_signatures(support)[-1]
        for support in supports
    }
    query_rows = []
    for query in queries:
        target = oracle_queries[query["id"]]
        primary = execute_partial(survivor_programs, query["entities"], query["initial_state"], query["actions"])
        memoryless = execute_partial([target_program], query["entities"], query["initial_state"], query["actions"], memoryless=True)
        lookup = [support_lookup[query["structural_key"]]] if query["structural_key"] in support_lookup else ["__unconstrained_unseen_sequence__"]
        step_exact = primary["possible_step_states"] == target["possible_step_states"]
        final_exact = primary["possible_final_observations"] == target["possible_final_observations"]
        query_rows.append({
            "id": query["id"],
            "family": record["construction_family"],
            "split": record["split"],
            "sequence_length": query["sequence_length"],
            "entity_count": query["entity_count"],
            "partial_initial_state": query["partial_initial_state"],
            "order_counterfactual_group": query["order_counterfactual_group"],
            "order_counterfactual_role": query["order_counterfactual_role"],
            "order_effect": query["order_effect"],
            "step_exact": step_exact,
            "final_exact": final_exact,
            "memoryless_final_exact": memoryless["possible_final_observations"] == target["possible_final_observations"],
            "lookup_final_exact": lookup == target["possible_final_observations"],
            "predicted_final": primary["possible_final_observations"],
            "target_final": target["possible_final_observations"],
        })
    groups = defaultdict(list)
    for row in query_rows:
        if row["order_effect"]:
            groups[row["order_counterfactual_group"]].append(row)
    order_checks = []
    for rows in groups.values():
        valid_pair = len(rows) == 2 and {row["order_counterfactual_role"] for row in rows} == {"forward", "reversed"}
        target_differs = valid_pair and rows[0]["target_final"] != rows[1]["target_final"]
        predicted_differs = valid_pair and rows[0]["predicted_final"] != rows[1]["predicted_final"]
        order_checks.append(valid_pair and target_differs and predicted_differs and all(row["final_exact"] for row in rows))
    target_retained = any(row["key"] == target_key for row in survivors)
    schema_recovered = len(survivors) == 1 and survivors[0]["key"] == target_key
    metrics = {
        "program_validation": all(support_validation) and all(query_validation),
        "target_retained": target_retained,
        "schema_recovered": schema_recovered,
        "empty_version_space": not survivors,
        "version_space": len(survivors),
        "query_rows": query_rows,
        "order_checks": order_checks,
        "complete_mechanic_exact": all(row["step_exact"] and row["final_exact"] for row in query_rows),
        "prefixes": prefixes,
    }
    prediction = {
        "id": record["id"],
        "split": record["split"],
        "family": record["construction_family"],
        "program_validation": metrics["program_validation"],
        "target_retained": target_retained,
        "schema_recovered": schema_recovered,
        "version_space": len(survivors),
        "next_state_exact": rate([row["step_exact"] for row in query_rows]),
        "final_observation_exact": rate([row["final_exact"] for row in query_rows]),
        "memoryless_final_observation_exact": rate([row["memoryless_final_exact"] for row in query_rows]),
        "literal_lookup_final_observation_exact": rate([row["lookup_final_exact"] for row in query_rows]),
        "order_counterfactual_accuracy": rate(order_checks),
    }
    return metrics, prediction


def aggregate(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    queries = [row for metric in metrics for row in metric["query_rows"]]
    order_checks = [value for metric in metrics for value in metric["order_checks"]]

    def grouped(field: str, metric: str) -> dict[str, float]:
        return {
            str(value): rate([row[metric] for row in queries if row[field] == value])
            for value in sorted({row[field] for row in queries}, key=str)
        }

    prefix_rows = defaultdict(list)
    for metric in metrics:
        for prefix in metric["prefixes"]:
            prefix_rows[prefix["prefix"]].append(prefix)
    return {
        "mechanics": len(metrics),
        "queries": len(queries),
        "oracle_program_validation": rate([row["program_validation"] for row in metrics]),
        "stateful_target_retention": rate([row["target_retained"] for row in metrics]),
        "stateful_schema_recovery": rate([row["schema_recovered"] for row in metrics]),
        "stateful_empty_version_space": rate([row["empty_version_space"] for row in metrics]),
        "stateful_next_state_exact": rate([row["step_exact"] for row in queries]),
        "stateful_final_observation_exact": rate([row["final_exact"] for row in queries]),
        "stateful_complete_mechanic_exact": rate([row["complete_mechanic_exact"] for row in metrics]),
        "stateful_by_family_final_exact": grouped("family", "final_exact"),
        "stateful_by_sequence_length_final_exact": grouped("sequence_length", "final_exact"),
        "stateful_by_split_final_exact": grouped("split", "final_exact"),
        "stateful_partial_initial_final_exact": grouped("partial_initial_state", "final_exact"),
        "order_counterfactual_pairs": len(order_checks),
        "order_counterfactual_accuracy": rate(order_checks),
        "memoryless_final_observation_exact": rate([row["memoryless_final_exact"] for row in queries]),
        "literal_lookup_final_observation_exact": rate([row["lookup_final_exact"] for row in queries]),
        "median_final_version_space": float(np.median([row["version_space"] for row in metrics])),
        "support_prefix": {
            str(prefix): {
                "mechanics": len(rows),
                "target_retention": rate([row["target_retained"] for row in rows]),
                "median_version_space": float(np.median([row["version_space"] for row in rows])),
            }
            for prefix, rows in sorted(prefix_rows.items())
        },
    }


def qualification(metrics: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    checks = {
        "oracle_program_validation": metrics["oracle_program_validation"] >= gates["minimumOracleProgramValidation"],
        "stateful_target_retention": metrics["stateful_target_retention"] >= gates["minimumStatefulTargetRetention"],
        "stateful_schema_recovery": metrics["stateful_schema_recovery"] >= gates["minimumStatefulSchemaRecovery"],
        "stateful_next_state_exact": metrics["stateful_next_state_exact"] >= gates["minimumStatefulNextStateExact"],
        "stateful_final_observation_exact": metrics["stateful_final_observation_exact"] >= gates["minimumStatefulFinalObservationExact"],
        "stateful_every_family": min(metrics["stateful_by_family_final_exact"].values()) >= gates["minimumStatefulEveryFamilyExact"],
        "stateful_every_sequence_length": min(metrics["stateful_by_sequence_length_final_exact"].values()) >= gates["minimumStatefulEverySequenceLengthExact"],
        "order_counterfactual_accuracy": metrics["order_counterfactual_accuracy"] >= gates["minimumOrderCounterfactualAccuracy"],
        "memoryless_inadequate": metrics["memoryless_final_observation_exact"] <= gates["maximumMemorylessFinalObservationExact"],
        "literal_lookup_inadequate": metrics["literal_lookup_final_observation_exact"] <= gates["maximumLiteralLookupFinalObservationExact"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-seal", default="configs/v42-corpus-seal.json")
    parser.add_argument("--output-dir", default="outputs/v42-sequential-state-foundation/development")
    args = parser.parse_args()
    seal_path = (PROJECT_ROOT / args.corpus_seal).resolve()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    attempt = output.parent / "development-attempt.json"
    if output.exists() or attempt.exists():
        raise RuntimeError("V42 oracle development already attempted")
    seal = json.loads(seal_path.read_text())
    implementation_path = PROJECT_ROOT / seal["implementation_lock"]
    implementation = json.loads(implementation_path.read_text())
    for path, expected in implementation["implementation"].items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"V42 implementation changed: {path}")
    records = []
    for artifact in seal["corpora"].values():
        path = PROJECT_ROOT / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"V42 sealed corpus changed: {artifact['path']}")
        records.extend(read(path))
    records.sort(key=lambda row: row["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt.write_text(json.dumps({
        "schema_version": 42, "status": "started", "oracle_development_run": 1,
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
        decision = "stateful_foundation_pass_authorize_sequential_language_grounding"
    elif not checks["stateful_next_state_exact"] or not checks["stateful_final_observation_exact"] or not checks["stateful_schema_recovery"]:
        decision = "repair_stateful_dsl_or_executor_before_language"
    elif not checks["memoryless_inadequate"]:
        decision = "redesign_benchmark_to_require_persistent_state"
    else:
        decision = "redesign_generalization_split_before_claim"
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "mechanic-predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in predictions))
    result = {
        "schema_version": 42,
        "experiment": implementation["config_payload"]["experiment"],
        "corpus_seal": str(seal_path.relative_to(PROJECT_ROOT)),
        "corpus_seal_sha256": file_sha256(seal_path),
        "oracle_development_run_number": 1,
        "metrics": metrics,
        "qualification": qualified,
        "decision": decision,
        "predictions": str(prediction_path.relative_to(PROJECT_ROOT)),
        "predictions_sha256": file_sha256(prediction_path),
        "runtime_seconds": time.perf_counter() - started,
        "data_access": {
            "oracle_development_runs": 1,
            "mechanics_scored": len(records),
            "development_fit_mechanics": sum(row["split"] == "development_fit" for row in records),
            "development_evaluation_mechanics": sum(row["split"] == "development_evaluation" for row in records),
            "selection_on_development_evaluation": 0,
            "language_model_forward_passes": 0,
            "adapter_training_runs": 0,
            "v41_records_read": 0,
        },
        "authorization": {
            "preregister_sequential_language_grounding": qualified["passed"],
            "construct_sequential_language_grounding": False,
            "add_stochasticity_or_delay": False,
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
